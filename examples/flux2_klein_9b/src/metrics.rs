use std::sync::Arc;

use luminal_rocm_lite::rocmrc::HipContext;

/// One stage execute's measurement: (GPU time µs, kernel launches, FLOPs).
pub type Sample = (f64, usize, u64);

/// Run `warmup` throwaway executes then average `repeats` timed executes of a
/// stage closure. The closure runs one real stage execute and returns
/// `(output, sample)`; we average the sample's time and take the (constant)
/// launches/flops. The output is ignored here — it's returned so the same
/// closure can also drive the real pipeline.
pub fn measure(
    warmup: usize,
    repeats: usize,
    mut run: impl FnMut() -> (Vec<f32>, Sample),
) -> Sample {
    for _ in 0..warmup {
        run();
    }
    let n = repeats.max(1);
    let (mut total_us, mut launches, mut flops) = (0.0_f64, 0usize, 0u64);
    for _ in 0..n {
        let (_out, (t, l, f)) = run();
        total_us += t;
        launches = l;
        flops = f;
    }
    (total_us / n as f64, launches, flops)
}

/// Format the collected per-stage samples into the GPU performance report. Pure
/// calculation + printing — all measurement happens in the caller.
///
/// `rows` is `(component name, per-pass sample, passes/image, cold-start wall
/// ms)`; `dit_by_type` is the DiT's `(op type, exec ops, flops)` breakdown for
/// one pass.
pub fn report_metrics(
    ctx: &Arc<HipContext>,
    warmup: usize,
    repeats: usize,
    rows: &[(&'static str, Sample, usize, f64)],
    dit_by_type: Vec<(&'static str, usize, u64)>,
) {
    println!("\n=== GPU performance (warm: {warmup} warmup + {repeats} timed executes) ===\n");
    println!(
        "{:<13} {:>7} {:>12} {:>12} {:>12} {:>12}",
        "component", "passes", "ms/pass", "total ms", "launches", "GFLOP"
    );
    println!("{}", "-".repeat(72));
    let (mut tot_ms, mut tot_launch, mut tot_flops) = (0.0_f64, 0usize, 0u64);
    for &(name, (us, launch, flops), passes, _cold) in rows {
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

    print!("\nfirst-run wall incl. graph instantiation (one-time, excl. compile):\n ");
    for (i, &(name, _, _, cold)) in rows.iter().enumerate() {
        print!("{}{name} {cold:.1} ms", if i == 0 { " " } else { " | " });
    }
    println!();

    println!("\n--- DiT compute by op type (one pass) ---");
    let stage_flops: u64 = dit_by_type.iter().map(|(_, _, f)| f).sum();
    println!("{:<12} {:>10} {:>14} {:>9}", "op type", "exec ops", "GFLOP", "% flops");
    for (name, count, flops) in &dit_by_type {
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
