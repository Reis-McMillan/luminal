#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod hf;
mod io;
mod metrics;
mod scheduler;
mod text_encoder;
mod transformer;
mod util;
mod vae;

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use luminal::graph::CompileOptions;
use luminal::prelude::*;
use luminal_rocm_lite::{rocmrc::HipContext, runtime::RocmRuntime};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use io::{load_png, save_png};
use metrics::{Sample, measure, report_metrics};
use scheduler::{SchedulerConfig, compute_mu, euler_step, make_schedule};
use tokenizers::Tokenizer;
use transformer::{HEAD_DIM, IN_CHANNELS};
use util::Materialize;
use vae::{VAE_DOWNSAMPLE, VaeDecoder, VaeEncoder};

const BN_EPS: f32 = 1e-4;

const _: () = assert!(text_encoder::OUTPUT_DIM == transformer::JOINT_ATTENTION_DIM);

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    text_len: usize,
) -> Result<(Vec<i32>, usize), Box<dyn std::error::Error>> {
    // Format the Qwen3 chat template (user message only, no system prompt) the
    // way Klein's pipeline does, then tokenize. The `<|im_start|>` / `<|im_end|>`
    // / `<think>` tags are single ids in the Qwen2 tokenizer, so we pass
    // `add_special_tokens = false` (no BOS is added; `add_bos_token = false`).
    let formatted = text_encoder::format_chat(prompt);
    let encoded = tokenizer
        .encode(formatted, false)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let mut ids: Vec<i32> = encoded.get_ids().iter().map(|&i| i as i32).collect();
    let real_len = ids.len();
    if real_len > text_len {
        ids.truncate(text_len);
    } else {
        // Right-pad to `text_len` with the Qwen2 pad token `<|endoftext|>`
        // (id 151643), matching diffusers' `padding="max_length"`.
        ids.resize(text_len, 151643);
    }
    Ok((ids, real_len.min(text_len)))
}

fn forward_bridge(latent: GraphTensor, bn_mean: GraphTensor, bn_var: GraphTensor) -> GraphTensor {
    let x = latent
        .split_dims(1, 2)
        .split_dims(3, 2)
        .permute(&[0, 2, 4, 1, 3])
        .merge_dims(0, 1)
        .merge_dims(0, 1);

    let h = x.dims()[1];
    let w = x.dims()[2];
    let mean = bn_mean.cast(DType::F32).expand_dim(1, h).expand_dim(2, w);
    let std = (bn_var.cast(DType::F32) + BN_EPS)
        .sqrt()
        .expand_dim(1, h)
        .expand_dim(2, w);
    let x = (x - mean) / std;

    x.merge_dims(1, 2).transpose(0, 1)
}


fn inverse_bridge(
    packed: GraphTensor,
    bn_mean: GraphTensor,
    bn_var: GraphTensor,
    w_pack: impl Into<Expression>,
) -> GraphTensor {
    let x = packed.transpose(0, 1).split_dims(1, w_pack);

    let h = x.dims()[1];
    let w = x.dims()[2];
    let mean = bn_mean.cast(DType::F32).expand_dim(1, h).expand_dim(2, w);
    let std = (bn_var.cast(DType::F32) + BN_EPS)
        .sqrt()
        .expand_dim(1, h)
        .expand_dim(2, w);
    let x = x * std + mean;

    x.split_dims(0, 4)
        .split_dims(1, 2)
        .permute(&[0, 3, 1, 4, 2])
        .merge_dims(1, 2)
        .merge_dims(2, 3)
}


fn bn_buffers(cx: &mut Graph) -> (GraphTensor, GraphTensor) {
    let bn_mean = cx
        .named_tensor("bn.running_mean", IN_CHANNELS)
        .as_dtype(DType::Bf16)
        .persist();
    let bn_var = cx
        .named_tensor("bn.running_var", IN_CHANNELS)
        .as_dtype(DType::Bf16)
        .persist();
    (bn_mean, bn_var)
}


fn compile_stage(
    cx: &mut Graph,
    ctx: &Arc<HipContext>,
    weight_files: &[std::path::PathBuf],
    seed_inputs: impl FnOnce(&mut RocmRuntime),
    search_iters: usize
) -> RocmRuntime {
    cx.build_search_space::<RocmRuntime>(CompileOptions::default());
    let mut rt = RocmRuntime::initialize(ctx.default_stream());
    for path in weight_files {
        rt.load_safetensors(cx, path.to_str().unwrap());
    }
    seed_inputs(&mut rt);
    cx.search(rt, CompileOptions::default().search_graph_limit(search_iters))
}

pub struct TextStage {
    pub cx: Graph,
    pub rt: RocmRuntime,
    pub ids: GraphTensor,
    pub pos: GraphTensor,
    pub mask: GraphTensor,
    pub out: GraphTensor,
}

impl TextStage {
    fn build(
        ctx: &Arc<HipContext>,
        weights: &[std::path::PathBuf],
        text_len: usize,
        search_iters: usize,
    ) -> Self {
        let mut cx = Graph::default();
        let ids = cx.named_tensor("__input_ids", text_len).as_dtype(DType::Int);
        let pos = cx.named_tensor("__pos_ids", text_len).as_dtype(DType::Int);
        let mask = cx
            .named_tensor("__attention_mask", text_len)
            .as_dtype(DType::F32);
        let out = text_encoder::Qwen3TextEncoder::init(&mut cx)
            .forward(ids, pos, mask)
            .output();
        let rt = compile_stage(
            &mut cx,
            ctx,
            weights,
            |rt| {
                rt.set_data(ids, vec![1i32; text_len]);
                rt.set_data(pos, (0..text_len as i32).collect::<Vec<_>>());
                rt.set_data(mask, vec![1.0_f32; text_len]);
            },
            search_iters,
        );
        Self { cx, rt, ids, pos, mask, out }
    }
}

pub struct EncodeStage {
    pub cx: Graph,
    pub rt: RocmRuntime,
    pub image: GraphTensor,
    pub packed: GraphTensor,
}

impl EncodeStage {
    fn build(
        ctx: &Arc<HipContext>,
        vae: &std::path::Path,
        height: usize,
        width: usize,
        search_iters: usize,
    ) -> Self {
        let mut cx = Graph::default();
        let image = cx.named_tensor("__image", (3usize, height, width));
        let (bn_mean, bn_var) = bn_buffers(&mut cx);
        let latent = VaeEncoder::new(&mut cx).encode_mean(image);
        let packed = forward_bridge(latent, bn_mean, bn_var).output();
        let files = [vae.to_path_buf()];
        let rt = compile_stage(
            &mut cx,
            ctx,
            &files,
            |rt| {
                rt.set_data(image, vec![0.0_f32; 3 * height * width]);
            },
            search_iters,
        );
        Self { cx, rt, image, packed }
    }
}

pub struct DitStage {
    pub cx: Graph,
    pub rt: RocmRuntime,
    pub latent: GraphTensor,
    pub text: GraphTensor,
    pub cos: GraphTensor,
    pub sin: GraphTensor,
    pub timestep: GraphTensor,
    pub velocity: GraphTensor,
}

impl DitStage {
    fn build(
        ctx: &Arc<HipContext>,
        weights: &[std::path::PathBuf],
        s_img: usize,
        s_txt: usize,
        with_reference: bool,
        search_iters: usize,
    ) -> Self {
        // Edit mode feeds [generated ++ reference] image tokens; text-to-image
        // feeds just the generated tokens.
        let s_img_tokens = if with_reference { 2 * s_img } else { s_img };
        let s_total = s_txt + s_img_tokens;
        let mut cx = Graph::default();
        let latent = cx.named_tensor("__latent", (s_img_tokens, IN_CHANNELS));
        let timestep = cx.named_tensor("__timestep", 1);
        let text = cx
            .named_tensor("__text", (s_txt, text_encoder::OUTPUT_DIM))
            .persist();
        let cos = cx.named_tensor("__rope_cos", (s_total, HEAD_DIM)).persist();
        let sin = cx.named_tensor("__rope_sin", (s_total, HEAD_DIM)).persist();
        // Keep only the generated tokens' velocity. `..s_img` is a contiguous
        // row prefix, so `materialize()` forces a copy (an un-materialized
        // contiguous slice would be dropped, leaking the reference rows).
        let velocity = transformer::Flux2Transformer::init(&mut cx)
            .forward(latent, text, cos, sin, timestep)
            .slice((..s_img, ..))
            .materialize()
            .output();
        let rt = compile_stage(
            &mut cx,
            ctx,
            weights,
            |rt| {
                rt.set_data(latent, vec![0.0_f32; s_img_tokens * IN_CHANNELS]);
                rt.set_data(timestep, vec![0.0_f32]);
                rt.set_data(text, vec![0.0_f32; s_txt * text_encoder::OUTPUT_DIM]);
                rt.set_data(cos, vec![0.0_f32; s_total * HEAD_DIM]);
                rt.set_data(sin, vec![0.0_f32; s_total * HEAD_DIM]);
            },
            search_iters,
        );
        Self { cx, rt, latent, text, cos, sin, timestep, velocity }
    }
}

pub struct DecodeStage {
    pub cx: Graph,
    pub rt: RocmRuntime,
    pub packed: GraphTensor,
    pub out: GraphTensor,
}

impl DecodeStage {
    fn build(
        ctx: &Arc<HipContext>,
        vae: &std::path::Path,
        h_lat: usize,
        w_lat: usize,
        search_iters: usize,
    ) -> Self {
        let s_img = (h_lat / 2) * (w_lat / 2);
        let mut cx = Graph::default();
        let packed = cx.named_tensor("__packed", (s_img, IN_CHANNELS));
        let (bn_mean, bn_var) = bn_buffers(&mut cx);
        let latent = inverse_bridge(packed, bn_mean, bn_var, w_lat / 2);
        let out = VaeDecoder::new(&mut cx).forward(latent).output();
        let files = [vae.to_path_buf()];
        let rt = compile_stage(
            &mut cx,
            ctx,
            &files,
            |rt| {
                rt.set_data(packed, vec![0.0_f32; s_img * IN_CHANNELS]);
            },
            search_iters,
        );
        Self { cx, rt, packed, out }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Text prompt.
    prompt: String,

    /// Input image for image-to-image / edit mode. Omit for text-to-image
    /// (generates a fresh 1024×1024 image from the prompt alone).
    #[arg(short = 'i', long)]
    image: Option<String>,

    /// Output PNG path.
    #[arg(short = 'o', long, default_value = "out.png")]
    output: String,

    #[arg(short = 'n', long, default_value_t = 5)]
    search_iters: usize,

    #[arg(short = 't', long, default_value_t = 512)]
    text_length: usize,

    #[arg(short = 'r', long, default_value_t = 0)]
    random_seed: usize,

    #[arg(short = 'g', long, default_value_t = 4.0)]
    guidance_scale: f32,

    #[arg(short = 'w', long, default_value_t = 2)]
    warmup: usize,

    #[arg(short = 'k', long, default_value_t = 5)]
    repeat: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        prompt,
        image,
        output,
        search_iters,
        text_length,
        random_seed,
        guidance_scale,
        warmup,
        repeat
    } = Args::parse();

    // Edit / image-to-image when an input image is given; otherwise text-to-image
    // at a fixed 1024×1024. `image_data` holds the loaded pixels (edit mode only).
    let with_reference = image.is_some();
    let (image_data, width, height) = match &image {
        Some(path) => {
            println!("Mode: image-to-image");
            println!("Input image: {path}");
            let (px, w, h) = load_png(path)?;
            (px, w, h)
        }
        None => {
            println!("Mode: text-to-image (1024×1024)");
            (Vec::new(), 1024, 1024)
        }
    };
    println!("Prompt: {prompt}");
    assert!(
        width.is_multiple_of(16) && height.is_multiple_of(16),
        "image dims must be multiples of 16 (got {width}x{height})",
    );
    let (h_lat, w_lat) = (height / VAE_DOWNSAMPLE, width / VAE_DOWNSAMPLE);
    let (h_pack, w_pack) = (h_lat / 2, w_lat / 2);
    let s_img = h_pack * w_pack;
    let s_txt = text_length;

    println!("\n[1/6] Resolving weights + tokenizer...");
    let tok_path = hf::fetch_tokenizer()?;
    let te_paths = hf::fetch_sharded("text_encoder")?;
    let tx_paths = hf::fetch_sharded("transformer")?;
    let vae_path = hf::fetch_vae()?;
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;

    let ctx = HipContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();

    println!("[2/6] Compiling text encoder ({s_txt} tokens)...");
    let t0 = Instant::now();
    let mut text_stage = TextStage::build(&ctx, &te_paths, s_txt, search_iters);
    println!("  done in {:.1}s", t0.elapsed().as_secs_f64());

    // VAE-encode stage only exists in edit mode (produces the reference latent).
    let mut encode_stage = if with_reference {
        println!("[3/6] Compiling VAE encode ({width}x{height})...");
        let t0 = Instant::now();
        let s = EncodeStage::build(&ctx, &vae_path, height, width, search_iters);
        println!("  done in {:.1}s", t0.elapsed().as_secs_f64());
        Some(s)
    } else {
        None
    };

    println!("[4/6] Compiling diffusion step (s_img={s_img}, s_txt={s_txt})...");
    let t0 = Instant::now();
    let mut dit_stage = DitStage::build(&ctx, &tx_paths, s_img, s_txt, with_reference, search_iters);
    println!("  done in {:.1}s", t0.elapsed().as_secs_f64());

    println!("[5/6] Compiling VAE decode...");
    let t0 = Instant::now();
    let mut decode_stage = DecodeStage::build(&ctx, &vae_path, h_lat, w_lat, search_iters);
    println!("  done in {:.1}s", t0.elapsed().as_secs_f64());

    println!("[6/6] Running pipeline...");
    let run_start = Instant::now();

    let (cond_ids, cond_len) = tokenize_prompt(&tokenizer, &prompt, s_txt)?;
    let cond_mask: Vec<f32> = (0..s_txt).map(|i| if i < cond_len { 1.0 } else { 0.0 }).collect();
    let (neg_ids, neg_len) = tokenize_prompt(&tokenizer, "", s_txt)?;
    let neg_mask: Vec<f32> = (0..s_txt).map(|i| if i < neg_len { 1.0 } else { 0.0 }).collect();
    let pos_ids: Vec<i32> = (0..s_txt as i32).collect();

    let steps = 4;
    let cfg = SchedulerConfig::default();
    let mu = compute_mu(&cfg, s_img);
    let (sigmas, timesteps_raw) = make_schedule(&cfg, steps, mu);
    let timesteps: Vec<f32> = timesteps_raw.iter().map(|t| t / 1000.0).collect();
    let mut rng = StdRng::seed_from_u64(random_seed as u64);
    let mut latent: Vec<f32> = (0..s_img * IN_CHANNELS)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();

    let mut run_text = |ids: &[i32], mask: &[f32]| -> (Vec<f32>, Sample) {
        text_stage.rt.set_data(text_stage.ids, ids.to_vec());
        text_stage.rt.set_data(text_stage.pos, pos_ids.clone());
        text_stage.rt.set_data(text_stage.mask, mask.to_vec());
        text_stage.rt.execute(&text_stage.cx.dyn_map);
        let out = text_stage.rt.get_f32(text_stage.out);
        let rt = &text_stage.rt;
        (out, (rt.last_total_time_us, rt.last_kernel_launches, rt.last_flops))
    };
    // Only reachable in edit mode (encode_stage is Some).
    let mut run_encode = || -> (Vec<f32>, Sample) {
        let es = encode_stage.as_mut().expect("encode stage exists in edit mode");
        es.rt.set_data(es.image, image_data.clone());
        es.rt.execute(&es.cx.dyn_map);
        let out = es.rt.get_f32(es.packed);
        let rt = &es.rt;
        (out, (rt.last_total_time_us, rt.last_kernel_launches, rt.last_flops))
    };

    // 1. Text-encode: conditional prompt + empty negative (for CFG).
    let t_cold = Instant::now();
    let (text_features, _) = run_text(&cond_ids, &cond_mask);
    let cold_text_ms = t_cold.elapsed().as_secs_f64() * 1e3;
    let (neg_text_features, _) = run_text(&neg_ids, &neg_mask);

    // 2. VAE-encode the input image → reference tokens (edit mode only). In
    //    text-to-image `reference` is empty, so the DiT sees just the generated
    //    tokens.
    let mut cold_enc_ms = 0.0_f64;
    let reference: Vec<f32> = if with_reference {
        let t_cold = Instant::now();
        let (r, _) = run_encode();
        cold_enc_ms = t_cold.elapsed().as_secs_f64() * 1e3;
        r
    } else {
        Vec::new()
    };

    // RoPE tables cover text ++ generated (++ reference in edit mode). Constant
    // across steps/passes → set once (persisted).
    let (cos, sin) = transformer::build_rope_tables(s_txt, h_pack, w_pack, with_reference);
    dit_stage.rt.set_data(dit_stage.cos, cos);
    dit_stage.rt.set_data(dit_stage.sin, sin);

    let mut run_dit = |text: &[f32], gen_latent: &[f32], t: f32| -> (Vec<f32>, Sample) {
        let mut model_input = Vec::with_capacity(gen_latent.len() + reference.len());
        model_input.extend_from_slice(gen_latent);
        model_input.extend_from_slice(&reference);
        dit_stage.rt.set_data(dit_stage.text, text.to_vec());
        dit_stage.rt.set_data(dit_stage.latent, model_input);
        dit_stage.rt.set_data(dit_stage.timestep, vec![t]);
        dit_stage.rt.execute(&dit_stage.cx.dyn_map);
        let v = dit_stage.rt.get_f32(dit_stage.velocity);
        let rt = &dit_stage.rt;
        (v, (rt.last_total_time_us, rt.last_kernel_launches, rt.last_flops))
    };
    let mut run_decode = |packed: &[f32]| -> (Vec<f32>, Sample) {
        decode_stage.rt.set_data(decode_stage.packed, packed.to_vec());
        decode_stage.rt.execute(&decode_stage.cx.dyn_map);
        let out = decode_stage.rt.get_f32(decode_stage.out);
        let rt = &decode_stage.rt;
        (out, (rt.last_total_time_us, rt.last_kernel_launches, rt.last_flops))
    };

    let mut cold_dit_ms = 0.0_f64;
    for i in 0..steps {
        let t_cold = Instant::now();
        let (v_cond, _) = run_dit(&text_features, &latent, timesteps[i]);
        if i == 0 {
            cold_dit_ms = t_cold.elapsed().as_secs_f64() * 1e3;
        }
        let (v_uncond, _) = run_dit(&neg_text_features, &latent, timesteps[i]);
        let velocity: Vec<f32> = v_uncond
            .iter()
            .zip(&v_cond)
            .map(|(u, c)| u + guidance_scale * (c - u))
            .collect();
        euler_step(&mut latent, &velocity, sigmas[i], sigmas[i + 1]);
        println!(
            "  step {}/{steps}  sigma {:.4} -> {:.4}",
            i + 1,
            sigmas[i],
            sigmas[i + 1]
        );
    }

    let t_cold = Instant::now();
    let (img, _) = run_decode(&latent);
    let cold_dec_ms = t_cold.elapsed().as_secs_f64() * 1e3;
    save_png(&output, &img, width, height)?;
    println!(
        "  \"{prompt}\" → {output} (run {:.1}s)",
        run_start.elapsed().as_secs_f64(),
    );

    // ── Steady-state measurement: re-run each stage warmup+repeats times ──────
    let s_text = measure(warmup, repeat, || run_text(&cond_ids, &cond_mask));
    let s_enc = if with_reference {
        Some(measure(warmup, repeat, || run_encode()))
    } else {
        None
    };
    let s_dit = measure(warmup, repeat, || run_dit(&text_features, &latent, timesteps[0]));
    let s_dec = measure(warmup, repeat, || run_decode(&latent));

    // DiT compute-by-op-type from the last DiT execute's kernel stats.
    let mut by_type: std::collections::BTreeMap<&'static str, (usize, u64)> =
        std::collections::BTreeMap::new();
    for s in &dit_stage.rt.last_kernel_stats {
        let e = by_type.entry(s.name).or_insert((0, 0));
        e.0 += 1;
        e.1 += s.flops as u64;
    }
    let dit_by_type: Vec<(&'static str, usize, u64)> =
        by_type.into_iter().map(|(n, (c, f))| (n, c, f)).collect();

    // (name, per-pass sample, passes/image, cold-start wall ms). VAE-encode only
    // appears in edit mode.
    let mut rows: Vec<(&'static str, Sample, usize, f64)> = Vec::new();
    rows.push(("text-encode", s_text, 2, cold_text_ms));
    if let Some(s_enc) = s_enc {
        rows.push(("vae-encode", s_enc, 1, cold_enc_ms));
    }
    rows.push(("DiT", s_dit, 2 * steps, cold_dit_ms));
    rows.push(("vae-decode", s_dec, 1, cold_dec_ms));

    report_metrics(&ctx, warmup, repeat, &rows, dit_by_type);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::vae::LATENT_CHANNELS;
    use super::*;

    fn ref_unpack(packed: &[f32], c: usize, h_pack: usize, w_pack: usize) -> Vec<f32> {
        let s_img = h_pack * w_pack;
        let mut out = vec![0.0_f32; c * s_img];
        for hi in 0..h_pack {
            for wi in 0..w_pack {
                let token = hi * w_pack + wi;
                for ci in 0..c {
                    out[ci * s_img + token] = packed[token * c + ci];
                }
            }
        }
        out
    }

    fn ref_bn_inverse(latent: &[f32], mean: &[f32], std: &[f32], c: usize) -> Vec<f32> {
        let hw = latent.len() / c;
        let mut out = vec![0.0_f32; latent.len()];
        for ci in 0..c {
            for i in 0..hw {
                out[ci * hw + i] = latent[ci * hw + i] * std[ci] + mean[ci];
            }
        }
        out
    }

    fn ref_unpatchify(packed: &[f32], c_out: usize, h_pack: usize, w_pack: usize) -> Vec<f32> {
        let (h_lat, w_lat) = (h_pack * 2, w_pack * 2);
        let mut out = vec![0.0_f32; c_out * h_lat * w_lat];
        for c in 0..c_out {
            for ph in 0..2 {
                for pw in 0..2 {
                    let in_c = c * 4 + ph * 2 + pw;
                    for hi in 0..h_pack {
                        for wi in 0..w_pack {
                            let in_idx = in_c * h_pack * w_pack + hi * w_pack + wi;
                            let out_idx =
                                c * h_lat * w_lat + (hi * 2 + ph) * w_lat + (wi * 2 + pw);
                            out[out_idx] = packed[in_idx];
                        }
                    }
                }
            }
        }
        out
    }

    fn one_search() -> CompileOptions {
        CompileOptions::default().search_graph_limit(1)
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-4, "mismatch at {i}: {a} vs {e}");
        }
    }

    #[test]
    fn inverse_bridge_matches_host_reference() {
        let (h_pack, w_pack) = (2usize, 3usize);
        let s_img = h_pack * w_pack;

        let mut cx = Graph::default();
        let packed_t = cx.named_tensor("packed", (s_img, IN_CHANNELS));
        let mean_t = cx.named_tensor("mean", IN_CHANNELS);
        let var_t = cx.named_tensor("var", IN_CHANNELS);
        let out = inverse_bridge(packed_t, mean_t, var_t, w_pack).output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());

        let packed: Vec<f32> = (0..s_img * IN_CHANNELS).map(|i| i as f32 * 0.01 - 3.0).collect();
        let mean: Vec<f32> = (0..IN_CHANNELS).map(|i| i as f32 * 0.02 - 1.0).collect();
        let var: Vec<f32> = (0..IN_CHANNELS).map(|i| 0.5 + (i as f32 % 5.0) * 0.1).collect();
        let std: Vec<f32> = var.iter().map(|v| (v + BN_EPS).sqrt()).collect();

        let mut rt = cx.search(ReferenceRuntime::default(), one_search());
        rt.set_data(packed_t, packed.clone());
        rt.set_data(mean_t, mean.clone());
        rt.set_data(var_t, var.clone());
        rt.execute(&cx.dyn_map);

        let unpacked = ref_unpack(&packed, IN_CHANNELS, h_pack, w_pack);
        let denormed = ref_bn_inverse(&unpacked, &mean, &std, IN_CHANNELS);
        let expected = ref_unpatchify(&denormed, LATENT_CHANNELS, h_pack, w_pack);
        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        let (h_pack, w_pack) = (2usize, 3usize);
        let (h_lat, w_lat) = (h_pack * 2, w_pack * 2);

        let mut cx = Graph::default();
        let latent_t = cx.named_tensor("latent", (LATENT_CHANNELS, h_lat, w_lat));
        let mean_t = cx.named_tensor("mean", IN_CHANNELS);
        let var_t = cx.named_tensor("var", IN_CHANNELS);
        let packed = forward_bridge(latent_t, mean_t, var_t);
        let out = inverse_bridge(packed, mean_t, var_t, w_pack).output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());

        let latent: Vec<f32> = (0..LATENT_CHANNELS * h_lat * w_lat)
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        let mean: Vec<f32> = (0..IN_CHANNELS).map(|i| i as f32 * 0.02 - 1.0).collect();
        let var: Vec<f32> = (0..IN_CHANNELS).map(|i| 0.5 + (i as f32 % 5.0) * 0.1).collect();

        let mut rt = cx.search(ReferenceRuntime::default(), one_search());
        rt.set_data(latent_t, latent.clone());
        rt.set_data(mean_t, mean);
        rt.set_data(var_t, var);
        rt.execute(&cx.dyn_map);

        assert_close(rt.get_f32(out.id), &latent);
    }
}
