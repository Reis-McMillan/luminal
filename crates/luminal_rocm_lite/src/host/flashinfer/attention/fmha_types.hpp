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
// Everything a launcher needs to build a ck_tile kernel's MakeKargs. Layout is
// NHD: each tensor is logically (batch, seqlen, nhead, hdim), addressed by the
// element strides below. GQA is expressed via nhead_q / nhead_k.
//
// TODO(confirm vs fmha_fwd_kernel.hpp): the exact stride set MakeKargs expects
// (we finalize this when implementing prefill.hpp).
struct fmha_args {
    // Device pointers. q/k/v/o are 16-bit (InputDataType); lse is fp32, optional.
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void*       o_ptr;
    void*       lse_ptr;        // nullptr when unused

    // Problem shape.
    ck_tile::index_t batch;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;   // GQA group size = nhead_q / nhead_k
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t hdim_q;    // == hdim_v == kHeadDim for now
    ck_tile::index_t hdim_v;

    // Softmax scale (1 / sqrt(hdim)). Subsumes the graph's separate scale op.
    float scale_s;

    // Strides, in elements:
    //   stride_*       — step along the seqlen axis
    //   nhead_stride_*  — step between heads
    //   batch_stride_*  — step between batch items
    ck_tile::index_t stride_q, nhead_stride_q, batch_stride_q;
    ck_tile::index_t stride_k, nhead_stride_k, batch_stride_k;
    ck_tile::index_t stride_v, nhead_stride_v, batch_stride_v;
    ck_tile::index_t stride_o, nhead_stride_o, batch_stride_o;

    // Mask.
    MaskKind mask;
};

} // namespace luminal_fmha
