// Shared fmha types for the ck_tile-based ROCm attention kernels.

#pragma once

#include <ck_tile/core.hpp>

namespace luminal_fmha {

#ifndef LUMINAL_HEAD_DIM
#error "LUMINAL_HEAD_DIM must be defined (e.g. -DLUMINAL_HEAD_DIM=128)"
#endif
inline constexpr ck_tile::index_t kHeadDim = LUMINAL_HEAD_DIM;

#if defined(LUMINAL_FMHA_BF16)
using InputDataType = ck_tile::bf16_t;
#else
using InputDataType = ck_tile::fp16_t;
#endif

struct FmhaTypeConfig {
    using QDataType           = InputDataType;
    using KDataType           = InputDataType;
    using VDataType           = InputDataType;
    using SaccDataType        = float;          // Q x K^T accumulator
    using SMPLComputeDataType = float;          // softmax compute
    using PDataType           = InputDataType;  // probabilities feeding x V
    using OaccDataType        = float;          // output accumulator
    using ODataType           = InputDataType;  // output storage
    using LSEDataType         = float;          // log-sum-exp (decode splits / optional)
};

enum class MaskKind : int {
    None   = 0,
    Causal = 1,
};

struct fmha_args {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void*       o_ptr;
    void*       lse_ptr;        // nullptr when unused

    const int32_t* seqstart_q_ptr;
    const int32_t* seqstart_k_ptr;

    ck_tile::index_t batch;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;   // GQA group size = nhead_q / nhead_k
    ck_tile::index_t max_seqlen_q;    // longest seqlen_q in the batch; sizes the grid
    ck_tile::index_t total_q_tokens;  // sum of seqlen_q (packed q rows); sizes partials
    ck_tile::index_t hdim_q;    // == hdim_v == kHeadDim for now
    ck_tile::index_t hdim_v;

    float scale_s;

    ck_tile::index_t stride_q, nhead_stride_q;
    ck_tile::index_t stride_k, nhead_stride_k;
    ck_tile::index_t stride_v, nhead_stride_v;
    ck_tile::index_t stride_o, nhead_stride_o;

    MaskKind mask;

    void*            lse_acc_ptr = nullptr;
    void*            o_acc_ptr   = nullptr;
    ck_tile::index_t num_splits  = 0;

    // ── Paged-KV (in-kernel page navigation; mirrors CUDA FlashInfer's paged_kv_t).
    // When block_table_ptr != nullptr, k_ptr/v_ptr are the KV *pool* and the kernel
    // walks pages itself instead of reading a pre-gathered contiguous buffer. ──
    const int32_t*   block_table_ptr          = nullptr; // [batch, batch_stride_block_table] physical page ids
    ck_tile::index_t batch_stride_block_table = 0;       // row stride of block_table (>= max pages per seq)
    ck_tile::index_t page_block_size          = 1;       // tokens per page (luminal cache uses 1)
    ck_tile::index_t batch_stride_k           = 0;       // pool stride between pages = page_block_size*nhead_k*hdim
    ck_tile::index_t batch_stride_v           = 0;
};

} // namespace luminal_fmha
