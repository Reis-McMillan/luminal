use luminal::{dtype::DType, graph::Graph, prelude::*};

// ── Qwen3ForCausalLM architecture constants for FLUX.2-Klein-9B ───────────────
// Verified against black-forest-labs/FLUX.2-klein-9B `text_encoder/config.json`.
pub const HIDDEN: usize = 4096;
pub const NUM_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const KV_GROUPS: usize = NUM_HEADS / NUM_KV_HEADS; // 4
pub const HEAD_DIM: usize = 128;
pub const Q_DIM: usize = NUM_HEADS * HEAD_DIM; // 4096 == HIDDEN
pub const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024
pub const INTERMEDIATE: usize = 12288;
pub const RMS_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1.0e6;
pub const VOCAB_SIZE: usize = 151936;
/// Highest tapped layer — we only need to run layers 0..NUM_LAYERS_USED.
pub const NUM_LAYERS_USED: usize = 27;
/// Hidden-state layers tapped by diffusers' `Flux2KleinPipeline._get_qwen3_prompt_embeds`
/// (`hidden_states_layers = (9, 18, 27)`), stacked and concatenated into the channel axis.
pub const TAP_LAYERS: [usize; 3] = [9, 18, 27];
/// 3 × 4096 = 12288 = transformer `joint_attention_dim`.
pub const OUTPUT_DIM: usize = 3 * HIDDEN;

/// Storage dtype — the Qwen3 text encoder ships in BF16.
pub const WEIGHT_DTYPE: DType = DType::Bf16;

// =============================================================================
// Helpers (mirror the patterns in the existing `examples/qwen` & `gemma4_moe`)
// =============================================================================

fn linear_no_bias(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    if x.dtype == w.dtype {
        x.matmul(w.t())
    } else {
        x.cast(w.dtype).matmul(w.t()).cast(x.dtype)
    }
}

fn rmsnorm(x: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let w = if weight.dtype == DType::F32 {
        weight
    } else {
        weight.cast(DType::F32)
    };
    let x_rank = x.dims().len();
    let w_rank = w.dims().len();
    x.std_norm(x_rank - 1, eps) * w.expand_lhs(&x.dims()[..x_rank - w_rank])
}

/// Qwen3 QK-norm: per-head RMSNorm over the head-dim axis.
/// `x` is `(seq, n_heads, HEAD_DIM)`, `weight` is `(HEAD_DIM,)`. Computed in F32
/// (BF16 accumulation degrades the norm). Mirrors `examples/qwen`'s `qk_norm`.
fn qk_norm(x: GraphTensor, weight: GraphTensor, n_heads: usize) -> GraphTensor {
    let seq = x.dims()[0];
    let w = if weight.dtype == DType::F32 {
        weight
    } else {
        weight.cast(DType::F32)
    };
    let normed = x.cast(DType::F32).std_norm(2, RMS_EPS); // RMS over HEAD_DIM
    normed * w.expand_dim(0, n_heads).expand_dim(0, seq)
}

/// Rotary position embedding — half-rotation convention (`[x0, x1] →
/// [x0*cos - x1*sin, x1*cos + x0*sin]` where `x0`, `x1` are the first and
/// second halves of the head dim). Matches Llama / Mistral.
///
/// Inputs:
/// * `x`: `(seq, n_heads, head_dim)`
/// * `pos_ids`: `(seq,)` Int
/// * `theta`: RoPE base
fn apply_rope(x: GraphTensor, pos_ids: GraphTensor, n_heads: usize, theta: f32) -> GraphTensor {
    let cx = x.graph();
    let _seq = x.dims()[0];
    let half = HEAD_DIM / 2;

    // Frequencies: theta^(-2i/D) for i in 0..D/2 — represented as 1 / theta^(2i/D)
    let exponents = cx.arange_options(0, HEAD_DIM, 2).cast(DType::F32) / HEAD_DIM as f32;
    use luminal::prelude::F32Pow;
    let inv_freqs = theta.pow(exponents).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1)); // (seq, half)

    let cos = emb.cos().expand_dim(1, n_heads); // (seq, n_heads, half)
    let sin = emb.sin().expand_dim(1, n_heads);

    let x0 = x.slice((.., .., ..half));
    let x1 = x.slice((.., .., half..));
    let r0 = x0.cast(DType::F32) * cos - x1.cast(DType::F32) * sin;
    let r1 = x1.cast(DType::F32) * cos + x0.cast(DType::F32) * sin;
    r0.concat_along(r1, 2)
}

/// Standard scaled dot-product attention over `(n_heads, seq_q, head_dim)`,
/// `(n_heads, seq_k, head_dim)`, `(n_heads, seq_k, head_dim)` with a causal
/// mask. Returns `(seq_q, n_heads * head_dim)`.
fn causal_sdpa(
    q: GraphTensor,
    k: GraphTensor,
    v: GraphTensor,
    attention_mask: GraphTensor,
) -> GraphTensor {
    let cx = q.graph();
    let n_heads = q.dims()[0];
    let seq = q.dims()[1];
    let scale = (HEAD_DIM as f32).sqrt().recip();
    // Materialize strided views from the upstream transpose / GQA-expand chain
    // before expressing attention as HLIR matmuls. Today the generic batched
    // matmul fallback can handle those arbitrary strides correctly, but the
    // full model becomes too memory-heavy unless cuBLASLt sees contiguous
    // per-head matrices.
    let q = q * 1.0_f32;
    let k = k * 1.0_f32;
    let v = v * 1.0_f32;
    // Q @ K^T: (heads, seq, head_dim) @ (heads, seq, head_dim)^T = (heads, seq, seq).
    let scores = q.matmul(k.transpose(1, 2)) * scale;
    // Causal mask: positions where k_pos > q_pos are masked.
    let q_pos = cx.arange(seq).cast(DType::F32);
    let k_pos = cx.arange(seq).cast(DType::F32);
    let causal = k_pos.expand_dim(0, seq).gt(q_pos.expand_dim(1, seq));
    let causal = causal.cast(DType::F32);
    // Padding mask: keys at positions where attention_mask == 0 (padding
    // tokens) are masked regardless of the causal relation. Without this,
    // padding queries attend to prior padding keys via causal alone, and
    // every padding hidden state diverges from diffusers — surfaces as
    // cos_sim ≈ 0.65 on `prompt_embeds` even though tokens 0..real_len-1
    // match exactly. attention_mask has shape (seq,) with 1 for real and
    // 0 for padding tokens; broadcast as a per-key column to all queries.
    // (1 - mask[k]) is 1 for padding keys, 0 for real keys → adds -1e10
    // to every (q, padding_k) score.
    let pad_key = (attention_mask.cast(DType::F32) * (-1.0_f32) + 1.0_f32) // (seq,)
        .expand_dim(0, seq); // (seq_q=seq, seq_k=seq) — broadcast over q.
    // Combine: anywhere either causal or padding masks → -1e10.
    let mask = causal + pad_key;
    let mask = mask.expand_dim(0, n_heads);
    let masked = scores + mask * (-1e10_f32);
    let weights = masked.softmax(2);
    // attn = weights @ v: (heads, seq, seq) @ (heads, seq, head_dim) = (heads, seq, head_dim).
    let attn = weights.matmul(v);
    // `transpose(0, 1).merge_dims(1, 2)` produces a non-contiguous K stride;
    // materialize before the downstream o_proj matmul.
    attn.transpose(0, 1).merge_dims(1, 2) * 1.0_f32 // (seq_q, n_heads*head_dim)
}

// =============================================================================
// One Qwen3 layer (RMSNorm → GQA self-attn w/ QK-norm + residual → RMSNorm →
// SwiGLU MLP + residual). Same shape as the existing `examples/qwen`'s
// `QwenLayer`.
// =============================================================================

struct Qwen3Layer {
    attn_rms: GraphTensor,  // (HIDDEN,)
    q_proj: GraphTensor,    // (Q_DIM, HIDDEN) — Q dim = 32*128 = 4096
    q_norm: GraphTensor,    // (HEAD_DIM,) — Qwen3 QK-norm
    k_proj: GraphTensor,    // (KV_DIM, HIDDEN)
    k_norm: GraphTensor,    // (HEAD_DIM,) — Qwen3 QK-norm
    v_proj: GraphTensor,    // (KV_DIM, HIDDEN)
    o_proj: GraphTensor,    // (HIDDEN, Q_DIM)
    mlp_rms: GraphTensor,   // (HIDDEN,)
    gate_proj: GraphTensor, // (INTERMEDIATE, HIDDEN)
    up_proj: GraphTensor,   // (INTERMEDIATE, HIDDEN)
    down_proj: GraphTensor, // (HIDDEN, INTERMEDIATE)
}

impl Qwen3Layer {
    fn new(idx: usize, cx: &mut Graph) -> Self {
        let prefix = format!("model.layers.{idx}");
        let mk = |name: &str, shape: (usize, usize), cx: &mut Graph| -> GraphTensor {
            cx.named_tensor(format!("{prefix}.{name}"), shape)
                .as_dtype(WEIGHT_DTYPE)
                .persist()
        };
        let mk1 = |name: &str, n: usize, cx: &mut Graph| -> GraphTensor {
            cx.named_tensor(format!("{prefix}.{name}"), n)
                .as_dtype(WEIGHT_DTYPE)
                .persist()
        };
        Self {
            attn_rms: mk1("input_layernorm.weight", HIDDEN, cx),
            q_proj: mk("self_attn.q_proj.weight", (Q_DIM, HIDDEN), cx),
            q_norm: mk1("self_attn.q_norm.weight", HEAD_DIM, cx),
            k_proj: mk("self_attn.k_proj.weight", (KV_DIM, HIDDEN), cx),
            k_norm: mk1("self_attn.k_norm.weight", HEAD_DIM, cx),
            v_proj: mk("self_attn.v_proj.weight", (KV_DIM, HIDDEN), cx),
            o_proj: mk("self_attn.o_proj.weight", (HIDDEN, Q_DIM), cx),
            mlp_rms: mk1("post_attention_layernorm.weight", HIDDEN, cx),
            gate_proj: mk("mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN), cx),
            up_proj: mk("mlp.up_proj.weight", (INTERMEDIATE, HIDDEN), cx),
            down_proj: mk("mlp.down_proj.weight", (HIDDEN, INTERMEDIATE), cx),
        }
    }

    fn forward(
        &self,
        x: GraphTensor,
        pos_ids: GraphTensor,
        attention_mask: GraphTensor,
    ) -> GraphTensor {
        let h = rmsnorm(x, self.attn_rms, RMS_EPS);
        let q = linear_no_bias(h, self.q_proj);
        let k = linear_no_bias(h, self.k_proj);
        let v = linear_no_bias(h, self.v_proj);

        // (seq, dim) → (seq, n_heads, head_dim) → ... → (n_heads, seq, head_dim)
        let q = q.split_dims(1, HEAD_DIM); // (seq, NUM_HEADS, HEAD_DIM)
        let k = k.split_dims(1, HEAD_DIM); // (seq, NUM_KV_HEADS, HEAD_DIM)
        let v = v.split_dims(1, HEAD_DIM);

        // Qwen3 QK-norm: per-head RMSNorm on q and k before RoPE.
        let q = qk_norm(q, self.q_norm, NUM_HEADS);
        let k = qk_norm(k, self.k_norm, NUM_KV_HEADS);

        let q = apply_rope(q, pos_ids, NUM_HEADS, ROPE_THETA);
        let k = apply_rope(k, pos_ids, NUM_KV_HEADS, ROPE_THETA);

        // GQA expand: tile k, v along the kv_groups axis to match num_heads.
        let k = k
            .transpose(0, 1) // (NUM_KV_HEADS, seq, HEAD_DIM)
            .expand_dim(1, KV_GROUPS) // (NUM_KV_HEADS, KV_GROUPS, seq, HEAD_DIM)
            .merge_dims(0, 1); // (NUM_HEADS, seq, HEAD_DIM)
        let v = v.transpose(0, 1).expand_dim(1, KV_GROUPS).merge_dims(0, 1);
        let q = q.transpose(0, 1); // (NUM_HEADS, seq, HEAD_DIM)

        let attn = causal_sdpa(q, k, v, attention_mask); // (seq, Q_DIM)
        let attn_out = linear_no_bias(attn, self.o_proj); // (seq, HIDDEN)
        let x = x + attn_out;

        let h = rmsnorm(x, self.mlp_rms, RMS_EPS);
        let gate = linear_no_bias(h, self.gate_proj).silu();
        let up = linear_no_bias(h, self.up_proj);
        let mlp = linear_no_bias(gate * up, self.down_proj);
        x + mlp
    }
}

// =============================================================================
// Top-level text encoder
// =============================================================================

pub struct Qwen3TextEncoder {
    pub embed_tokens: GraphTensor, // (VOCAB_SIZE, HIDDEN) — used as a gather table
    layers: Vec<Qwen3Layer>,
}

impl Qwen3TextEncoder {
    pub fn init(cx: &mut Graph) -> Self {
        let embed_tokens = cx
            .named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
            .as_dtype(WEIGHT_DTYPE)
            .persist();
        let layers = (0..NUM_LAYERS_USED)
            .map(|i| Qwen3Layer::new(i, cx))
            .collect();
        Self {
            embed_tokens,
            layers,
        }
    }

    /// Run the prompt through the (truncated) text encoder and return the
    /// **stacked-and-flattened** `(seq, OUTPUT_DIM=12288)` text features the
    /// Flux 2 transformer's `context_embedder` consumes.
    ///
    /// Steps mirror diffusers' `Flux2KleinPipeline._get_qwen3_prompt_embeds`:
    ///   1. Gather `embed_tokens[input_ids]` → `(seq, HIDDEN)`.
    ///   2. Run layers; capture `hidden_states[9/18/27]` (in HF convention,
    ///      = post-residual after layers 8, 17, 26, all pre-final-norm).
    ///   3. Stack along a new "tap" axis: `(seq, 3, HIDDEN)`.
    ///   4. Flatten the tap axis into the channel axis: `(seq, 3*HIDDEN)`.
    pub fn forward(
        &self,
        input_ids: GraphTensor,
        pos_ids: GraphTensor,
        attention_mask: GraphTensor,
    ) -> GraphTensor {
        let seq = input_ids.dims1();
        // Token embedding lookup via gather. Mirror the qwen / llama pattern:
        // build a flat index table (id * HIDDEN + col) that picks the right
        // row from the embed_tokens (VOCAB_SIZE × HIDDEN) buffer. The source
        // is BF16 so the gathered slice is BF16 too — cast to F32 immediately
        // so the rest of the network runs in F32 with BF16 weights upcast at
        // each matmul (see `linear_no_bias`).
        let mut x = self.embed_tokens.gather(
            (input_ids * HIDDEN).expand_dim(1, HIDDEN)
                + input_ids.graph().arange(HIDDEN).expand_dim(0, seq),
        );
        x = x.cast(DType::F32);

        // Run layers, taking snapshots at the right HF-convention layer indices.
        // hidden_states[9] = post-residual after running 9 layers (idx 0..8), so
        // we capture AFTER running layer idx 8. Same for 18 and 27.
        let mut taps: Vec<GraphTensor> = Vec::with_capacity(TAP_LAYERS.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, pos_ids, attention_mask);
            // Map: TAP_LAYERS = [9, 18, 27] meaning "after running k layers".
            if TAP_LAYERS.iter().any(|&k| idx + 1 == k) {
                taps.push(x);
            }
        }

        // Stack as (seq, n_taps, HIDDEN) then flatten last two dims.
        let mut stacked = taps[0].expand_dim(1, 1_usize); // (seq, 1, HIDDEN)
        for t in &taps[1..] {
            stacked = stacked.concat_along(t.expand_dim(1, 1_usize), 1);
        }
        // (seq, 3, HIDDEN) → (seq, 3*HIDDEN)
        stacked.merge_dims(1, 2)
    }
}

// =============================================================================
// Chat-template formatting (text-only path) — produces the byte string that
// then gets fed to the Qwen2 tokenizer. Matches the template applied by
// diffusers' `Flux2KleinPipeline._get_qwen3_prompt_embeds`: a single user
// message, `add_generation_prompt=True`, `enable_thinking=False`, and NO
// system prompt.
// =============================================================================

/// Format a user prompt into the wire-format string the Qwen2 tokenizer
/// expects. Renders the Qwen3 chat template for `[{"role": "user", ...}]` with
/// `add_generation_prompt=True` and `enable_thinking=False` (the empty
/// `<think></think>` block). The tokenizer adds no BOS (`add_bos_token=false`),
/// and the `<|im_start|>` / `<|im_end|>` / `<think>` tags round-trip as single
/// ids. Mirrors `examples/qwen`'s `qwen3_chat_prompt`.
pub fn format_chat(user_prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

#[cfg(test)]
mod tests {
    use luminal::hlir::CustomOpKind;

    use super::*;

    fn assert_no_custom_ops(cx: &Graph) {
        assert!(
            cx.custom_ops.is_empty(),
            "Flux2 text encoder helpers should use pure HLIR, not registered CustomOp wrappers"
        );
        let custom_nodes: Vec<_> = cx
            .graph
            .node_indices()
            .filter(|&node| cx.try_get_op::<CustomOpKind>(node).is_some())
            .collect();
        assert!(
            custom_nodes.is_empty(),
            "Flux2 text encoder graph contains CustomOpKind nodes: {custom_nodes:?}"
        );
    }

    #[test]
    fn chat_template_matches_jinja_output() {
        // Sanity check: the result is the deterministic Qwen3 rendering we
        // expect for a user-only prompt (add_generation_prompt, no thinking).
        let s = format_chat("make a cat");
        assert_eq!(
            s,
            "<|im_start|>user\nmake a cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn architecture_constants_consistent() {
        assert_eq!(NUM_HEADS * HEAD_DIM, Q_DIM);
        assert_eq!(NUM_KV_HEADS * HEAD_DIM, KV_DIM);
        assert!(NUM_HEADS.is_multiple_of(NUM_KV_HEADS));
        assert_eq!(KV_GROUPS, NUM_HEADS / NUM_KV_HEADS);
        assert_eq!(OUTPUT_DIM, TAP_LAYERS.len() * HIDDEN);
        // hidden_states[30] requires running 30 layers (0..29 inclusive).
        assert_eq!(NUM_LAYERS_USED, *TAP_LAYERS.iter().max().unwrap());
    }

    #[test]
    fn text_encoder_helpers_use_no_custom_ops() {
        let mut cx = Graph::default();

        let x = cx.named_tensor("x", (2usize, 3usize));
        let w = cx
            .named_tensor("w", (4usize, 3usize))
            .as_dtype(WEIGHT_DTYPE);
        let _ = linear_no_bias(x, w).output();

        let q = cx.named_tensor("q", (1usize, 2usize, HEAD_DIM));
        let k = cx.named_tensor("k", (1usize, 2usize, HEAD_DIM));
        let v = cx.named_tensor("v", (1usize, 2usize, HEAD_DIM));
        let mask = cx.named_tensor("attention_mask", 2usize);
        let _ = causal_sdpa(q, k, v, mask).output();

        assert_no_custom_ops(&cx);
    }
}
