// Shared fmha types for the ck_tile-based ROCm attention kernels.
//
// Role: the compile-time + runtime contract shared by prefill.hpp and
// decode.hpp — dtype configuration, the mask enum, and the runtime `fmha_args`
// (pointers, shapes, strides) the launchers consume.
//
// This mirrors the *structure* of CK's example fmha_fwd.hpp (FmhaFwdTypeConfig
// / fmha_fwd_args), trimmed to luminal's needs: fp16/bf16 storage with fp32
// accumulation, causal mask, GQA. Bias / dropout / fp8 paths are intentionally
// omitted.
//
// FlashInfer reference: include/flashinfer/attention/default_decode_params.cuh,
// default_prefill_params.cuh, variants.cuh.

#pragma once

// ── ck_tile core types ──
// Provides ck_tile::fp16_t, ck_tile::bf16_t, ck_tile::index_t.
// TODO(confirm vs fmha_fwd_kernel.hpp): exact umbrella header name/path.
#include <ck_tile/core.hpp>

namespace luminal_fmha {

// ── HEAD_DIM (compile-time, injected via -DLUMINAL_HEAD_DIM=N, like wrapper.cu) ──
#ifndef LUMINAL_HEAD_DIM
#error "LUMINAL_HEAD_DIM must be defined (e.g. -DLUMINAL_HEAD_DIM=128)"
#endif
inline constexpr ck_tile::index_t kHeadDim = LUMINAL_HEAD_DIM;

// ── Input/storage dtype (fp16 default; -DLUMINAL_FMHA_BF16 selects bf16) ──
#if defined(LUMINAL_FMHA_BF16)
using InputDataType = ck_tile::bf16_t;
#else
using InputDataType = ck_tile::fp16_t;
#endif

// ── Warp tile (one matrix-core MMA instruction, M×N×K) ──
// CDNA (gfx9, the production target) uses MFMA 32×32×16. RDNA3/RDNA4 (gfx11/12)
// have no 32×32×16 instruction — they use WMMA 16×16×16, and an MFMA-shaped tile
// silently produces zeros there. jit.rs passes -DLUMINAL_FMHA_WMMA when the
// detected arch is a WMMA arch, so the same source is correct on both: tuned for
// CDNA, testable on a gfx11 dev card.
#if defined(LUMINAL_FMHA_WMMA)
inline constexpr ck_tile::index_t kWarpTileM = 16;
inline constexpr ck_tile::index_t kWarpTileN = 16;
inline constexpr ck_tile::index_t kWarpTileK = 16;
#else
inline constexpr ck_tile::index_t kWarpTileM = 32;
inline constexpr ck_tile::index_t kWarpTileN = 32;
inline constexpr ck_tile::index_t kWarpTileK = 16;
#endif

// ── Dtype configuration ──
// Storage (Q/K/V/O/probs) is 16-bit; the Q·Kᵀ accumulator, softmax math, output
// accumulator, and LSE are fp32. This split is the concrete expression of
// "tensor cores take 16-bit inputs but accumulate in 32-bit".
struct FmhaTypeConfig {
    using QDataType           = InputDataType;
    using KDataType           = InputDataType;
    using VDataType           = InputDataType;
    using SaccDataType        = float;          // Q·Kᵀ accumulator
    using SMPLComputeDataType = float;          // softmax compute
    using PDataType           = InputDataType;  // probabilities feeding ·V
    using OaccDataType        = float;          // output accumulator
    using ODataType           = InputDataType;  // output storage
    using LSEDataType         = float;          // log-sum-exp (decode splits / optional)
};

// ── Mask ──
enum class MaskKind : int {
    None   = 0,
    Causal = 1,
};

// ── Runtime arguments ──
// Everything a launcher needs to build a ck_tile kernel's MakeKargs. Group mode
// (variable-length batches): the tensors are RAGGED — all sequences packed
// contiguously along the seqlen axis as (total_tokens, nhead, hdim), with
// per-sequence boundaries given by the cumulative seqstart_* arrays (a.k.a.
// CSR indptr, length batch+1). There is no batch stride; sequence b's rows live
// at [seqstart[b], seqstart[b+1]). GQA is expressed via nhead_q / nhead_k.
struct fmha_args {
    // Device pointers. q/k/v/o are 16-bit (InputDataType); lse is fp32, optional.
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void*       o_ptr;
    void*       lse_ptr;        // nullptr when unused

    // Cumulative per-sequence offsets (device, int32, length batch+1). These
    // replace batch strides in group mode: seqstart_q indexes the packed Q/O,
    // seqstart_k indexes the packed K/V.
    const int32_t* seqstart_q_ptr;
    const int32_t* seqstart_k_ptr;

    // Problem shape.
    ck_tile::index_t batch;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;   // GQA group size = nhead_q / nhead_k
    ck_tile::index_t max_seqlen_q;    // longest seqlen_q in the batch; sizes the grid
    ck_tile::index_t total_q_tokens;  // sum of seqlen_q (packed q rows); sizes partials
    ck_tile::index_t hdim_q;    // == hdim_v == kHeadDim for now
    ck_tile::index_t hdim_v;

    // Softmax scale (1 / sqrt(hdim)). Subsumes the graph's separate scale op.
    float scale_s;

    // Strides, in elements (no batch stride in group mode — see seqstart_*):
    //   stride_*       — step along the seqlen axis
    //   nhead_stride_*  — step between heads
    ck_tile::index_t stride_q, nhead_stride_q;
    ck_tile::index_t stride_k, nhead_stride_k;
    ck_tile::index_t stride_v, nhead_stride_v;
    ck_tile::index_t stride_o, nhead_stride_o;

    // Mask.
    MaskKind mask;

    // Split-KV decode only (see decode.hpp). num_splits == 0 ⇒ unused (prefill).
    // lse_acc/o_acc point at fp32 partials scratch of decode_partials_bytes().
    void*            lse_acc_ptr = nullptr;
    void*            o_acc_ptr   = nullptr;
    ck_tile::index_t num_splits  = 0;
};

} // namespace luminal_fmha
