use std::sync::Arc;

use luminal::prelude::*;
use luminal_rocm_lite::{rocmrc::HipContext, runtime::RocmRuntime};

use crate::transformer::IN_CHANNELS;
use crate::{DecodeStage, DitStage, EncodeStage, TextStage};

pub struct StageDims {
    pub s_txt: usize,
    pub s_img: usize,
    pub width: usize,
    pub height: usize,
}

fn measure(
    rt: &mut RocmRuntime,
    cx: &Graph,
    warmup: usize,
    repeats: usize,
    mut reseed: impl FnMut(&mut RocmRuntime),
) -> (f64, usize, u64) {
    for _ in 0..warmup {
        reseed(rt);
        rt.execute(&cx.dyn_map);
    }
    let mut total_us = 0.0;
    for _ in 0..repeats.max(1) {
        reseed(rt);
        rt.execute(&cx.dyn_map);
        total_us += rt.last_total_time_us;
    }
    (
        total_us / repeats.max(1) as f64,
        rt.last_kernel_launches,
        rt.last_flops,
    )
}

pub fn report_metrics(
    ctx: &Arc<HipContext>,
    text_stage: &mut TextStage,
    encode_stage: &mut EncodeStage,
    dit_stage: &mut DitStage,
    decode_stage: &mut DecodeStage,
    steps: usize,
    dims: StageDims,
    cold_ms: [f64; 4],
    warmup: usize,
    repeats: usize
) {
    println!(
        "\n=== GPU performance (warm: {warmup} warmup + {repeats} timed executes) ===\n"
    );

    let StageDims {
        s_txt,
        s_img,
        width,
        height,
    } = dims;

    let (t_ids, t_pos, t_mask) = (text_stage.ids, text_stage.pos, text_stage.mask);
    let text = measure(&mut text_stage.rt, &text_stage.cx, warmup, repeats, |rt| {
        rt.set_data(t_ids, vec![0i32; s_txt]);
        rt.set_data(t_pos, vec![0i32; s_txt]);
        rt.set_data(t_mask, vec![0.0f32; s_txt]);
    });
    let e_image = encode_stage.image;
    let enc = measure(&mut encode_stage.rt, &encode_stage.cx, warmup, repeats, |rt| {
        rt.set_data(e_image, vec![0.0f32; 3 * width * height]);
    });
    let (d_latent, d_ts) = (dit_stage.latent, dit_stage.timestep);
    let dit = measure(&mut dit_stage.rt, &dit_stage.cx, warmup, repeats, |rt| {
        rt.set_data(d_latent, vec![0.0f32; 2 * s_img * IN_CHANNELS]);
        rt.set_data(d_ts, vec![0.0f32]);
    });
    let d_packed = decode_stage.packed;
    let dec = measure(&mut decode_stage.rt, &decode_stage.cx, warmup, repeats, |rt| {
        rt.set_data(d_packed, vec![0.0f32; s_img * IN_CHANNELS]);
    });

    let dit_passes = 2 * steps;
    let rows: [(&str, (f64, usize, u64), usize, f64); 4] = [
        ("text-encode", text, 2, cold_ms[0]),
        ("vae-encode", enc, 1, cold_ms[1]),
        ("DiT", dit, dit_passes, cold_ms[2]),
        ("vae-decode", dec, 1, cold_ms[3]),
    ];

    println!(
        "{:<13} {:>7} {:>12} {:>12} {:>12} {:>12}",
        "component", "passes", "ms/pass", "total ms", "launches", "GFLOP"
    );
    println!("{}", "-".repeat(72));
    let (mut tot_ms, mut tot_launch, mut tot_flops) = (0.0_f64, 0usize, 0u64);
    for (name, (us, launch, flops), passes, _cold) in rows {
        let ms_pass = us / 1e3;
        let total_ms = ms_pass * passes as f64;
        let launch_total = launch * passes;
        let flops_total = flops * passes as u64;
        tot_ms += total_ms;
        tot_launch += launch_total;
        tot_flops += flops_total;
        println!(
            "{:<13} {:>7} {:>12.3} {:>12.2} {:>12} {:>12.2}",
            name,
            passes,
            ms_pass,
            total_ms,
            launch_total,
            flops_total as f64 / 1e9,
        );
    }
    println!("{}", "-".repeat(72));
    let tot_s = tot_ms / 1e3;
    let achieved_tf = if tot_s > 0.0 {
        tot_flops as f64 / tot_s / 1e12
    } else {
        0.0
    };
    println!(
        "{:<13} {:>7} {:>12} {:>12.2} {:>12} {:>12.2}",
        "end-to-end",
        "",
        "",
        tot_ms,
        tot_launch,
        tot_flops as f64 / 1e9,
    );

    println!("\nend-to-end (GPU execute only): {tot_ms:.2} ms");
    println!("total GPU kernel launches:    {tot_launch}");
    println!(
        "total work:                   {:.2} GFLOP ({:.3} TFLOP)",
        tot_flops as f64 / 1e9,
        tot_flops as f64 / 1e12
    );
    // Klein's matmuls run in BF16, so MFU is measured against the BF16
    // matrix-core peak (fall back to the f32 peak, then nothing).
    match luminal_rocm_lite::rocm_compute_bf16_tflops(ctx) {
        Some(peak) => println!(
            "achieved throughput:          {achieved_tf:.2} TFLOP/s  ({:.1}% of {peak} TFLOP/s BF16 matrix peak)",
            achieved_tf / peak as f64 * 100.0
        ),
        None => match luminal_rocm_lite::rocm_compute_f32_tflops(ctx) {
            Some(peak) => println!(
                "achieved throughput:          {achieved_tf:.2} TFLOP/s  ({:.1}% of {peak} TFLOP/s f32 peak)",
                achieved_tf / peak as f64 * 100.0
            ),
            None => println!("achieved throughput:          {achieved_tf:.2} TFLOP/s"),
        },
    }

    println!(
        "\nfirst-run wall incl. graph instantiation (one-time, excl. compile):\n  \
         text {:.1} ms | encode {:.1} ms | DiT {:.1} ms | decode {:.1} ms",
        cold_ms[0], cold_ms[1], cold_ms[2], cold_ms[3]
    );

    println!("\n--- DiT compute by op type (one pass) ---");
    let stats = &dit_stage.rt.last_kernel_stats;
    let stage_flops: u64 = stats.iter().map(|s| s.flops as u64).sum();
    let mut by_type: std::collections::BTreeMap<&str, (usize, u64)> =
        std::collections::BTreeMap::new();
    for s in stats {
        let e = by_type.entry(s.name).or_insert((0, 0));
        e.0 += 1;
        e.1 += s.flops as u64;
    }
    println!("{:<12} {:>10} {:>14} {:>9}", "op type", "exec ops", "GFLOP", "% flops");
    for (name, (count, flops)) in &by_type {
        println!(
            "{:<12} {:>10} {:>14.2} {:>8.1}%",
            name,
            count,
            *flops as f64 / 1e9,
            if stage_flops > 0 {
                *flops as f64 / stage_flops as f64 * 100.0
            } else {
                0.0
            }
        );
    }
}