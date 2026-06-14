// Decode (seqlen_q == 1) attention — ck_tile split-KV flash-decoding.

#pragma once

#include "fmha_types.hpp"

#include <ck_tile/host.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>

namespace luminal_fmha {
namespace decode_detail {

using Cfg = FmhaTypeConfig;


namespace tile {
    constexpr ck_tile::index_t kWarpM = 16;
    constexpr ck_tile::index_t kWarpN = 16;
    constexpr ck_tile::index_t kWarpK = 16;
    constexpr ck_tile::index_t kM0 = 16;
    constexpr ck_tile::index_t kN0 = 128;
    constexpr ck_tile::index_t kK0 = 16;
    constexpr ck_tile::index_t kN1 = kHeadDim;
    constexpr ck_tile::index_t kK1 = 16;
    constexpr ck_tile::index_t kQKHeadDim = kHeadDim;
}

// see: https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp
using FmhaShape = ck_tile::TileFmhaShape<
    ck_tile::sequence<tile::kM0, tile::kN0, tile::kK0, tile::kN1, tile::kK1, tile::kQKHeadDim>,
    ck_tile::sequence<1, 4, 1>, ck_tile::sequence<tile::kWarpM, tile::kWarpN, tile::kWarpK>,
    ck_tile::sequence<1, 4, 1>, ck_tile::sequence<tile::kWarpM, tile::kWarpN, tile::kWarpK>,
    /*IsVLayoutRowMajor=*/true>;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/block/block_masking.hpp#L330
using FmhaMask    = ck_tile::SimplifiedGenericAttentionMask</*IsMasking=*/true>;
// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/block/variants.hpp#L137
using FmhaVariant = ck_tile::StandardAttention;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp#L152
using SplitKVTraits = ck_tile::TileFmhaFwdSplitKVTraits<
    /*kPadSeqLenQ=*/true, /*kPadSeqLenK=*/true,
    /*kPadHeadDimQ=*/true, /*kPadHeadDimV=*/true,
    /*kHasLogitsSoftCap=*/false,
    /*BiasEnum=*/ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    /*kHasBiasGrad=*/false,
    /*kStoreLSE=*/true,
    /*kDoFp8StaticQuant=*/false,
    /*kIsPagedKV=*/false,
    /*kHasUnevenSplits=*/true>;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp#L258
using SplitKVProblem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    typename Cfg::QDataType, typename Cfg::KDataType, typename Cfg::VDataType,
    typename Cfg::SaccDataType, typename Cfg::SMPLComputeDataType,
    /*BiasDataType=*/typename Cfg::ODataType,
    typename Cfg::LSEDataType, typename Cfg::PDataType,
    /*OaccDataType=*/typename Cfg::OaccDataType,
    /*ODataType=*/typename Cfg::OaccDataType,
    FmhaShape, /*kIsGroupMode=*/true, FmhaVariant, FmhaMask, SplitKVTraits>;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp#L16
using SplitKVPipeline = ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<SplitKVProblem>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/epilogue/default_2d_epilogue.hpp
using SplitKVEpilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename Cfg::OaccDataType, typename Cfg::OaccDataType,
                                      /*kPadM=*/true, /*kPadN=*/true>>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp#L23
using SplitKVKernel = ck_tile::FmhaFwdSplitKVKernel<SplitKVPipeline, SplitKVEpilogue>;

constexpr ck_tile::index_t kCombineN1 = 32;
constexpr ck_tile::index_t kLogMaxSplits = 5; // kMaxSplits = 32
constexpr ck_tile::index_t kMaxSplits = 1 << kLogMaxSplits;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp#L315
using CombineTraits = ck_tile::TileFmhaFwdSplitKVCombineTraits<
    /*kPadSeqLenQ=*/true, /*kPadHeadDimV=*/true,
    /*kStoreLSE=*/false, /*kDoFp8StaticQuant=*/false, kLogMaxSplits>;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp#L315
using CombineProblem = ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
    typename Cfg::LSEDataType, typename Cfg::OaccDataType, typename Cfg::ODataType,
    kHeadDim, /*kIsGroupMode=*/true, kCombineN1, CombineTraits>;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp#L47
using CombinePipeline = ck_tile::BlockFmhaFwdSplitKVCombinePipeline<CombineProblem>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/epilogue/default_2d_epilogue.hpp
using CombineEpilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename Cfg::OaccDataType, typename Cfg::ODataType,
                                      /*kPadM=*/true, /*kPadN=*/true>>;

// see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp#L9
using CombineKernel = ck_tile::FmhaFwdSplitKVCombineKernel<CombinePipeline, CombineEpilogue>;

inline int num_splits_heuristic(int batch_nhead_mblocks, int num_sms, int max_splits) {
    if (batch_nhead_mblocks >= 0.8f * num_sms) return 1;
    max_splits = std::min(max_splits, num_sms);
    float max_eff = 0.f;
    std::vector<float> eff;
    eff.reserve(max_splits);
    for (int s = 1; s <= max_splits; ++s) {
        float n_waves = float(batch_nhead_mblocks * s) / num_sms;
        float e = n_waves / std::ceil(n_waves);
        if (e > max_eff) max_eff = e;
        eff.push_back(e);
    }
    for (int s = 1; s <= max_splits; ++s)
        if (eff[s - 1] >= 0.85f * max_eff) return s;
    return 1;
}

inline int choose_num_splits(int batch, int nhead_q, int max_seqlen_q) {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return 1;
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess) return 1;
    const int num_m_blocks = ck_tile::integer_divide_ceil(max_seqlen_q, tile::kM0);
    const int n = num_splits_heuristic(batch * nhead_q * num_m_blocks,
                                       props.multiProcessorCount * 2, kMaxSplits);
    return std::clamp(n, 1, (int)kMaxSplits);
}

} // namespace decode_detail

// Bytes of fp32 split-KV partials (o_acc + lse_acc) for a given problem. The
// caller carves this from the float workspace before launch_decode.
inline size_t decode_partials_bytes(int total_q_tokens, int nhead_q, int hdim_v, int num_splits) {
    const size_t o_acc   = (size_t)nhead_q * num_splits * total_q_tokens * hdim_v;
    const size_t lse_acc = (size_t)nhead_q * num_splits * total_q_tokens;
    return (o_acc + lse_acc) * sizeof(float);
}

inline void launch_decode(const fmha_args& a, hipStream_t stream) {
    using namespace decode_detail;

    const ck_tile::index_t window_left  = (a.mask == MaskKind::Causal) ? -1 : -1;
    const ck_tile::index_t window_right = (a.mask == MaskKind::Causal) ?  0 : -1;
    const ck_tile::index_t total_q = a.total_q_tokens; // group-mode shape_seqlen_q (packed q rows)

    // o_acc layout [nhead, split, q, hdim_v]; lse_acc layout [nhead, split, q].
    // (Strides match ck_tile/ops/fmha_fwd_runner.hpp exactly.)
    const ck_tile::index_t stride_o_acc          = a.hdim_v;
    const ck_tile::index_t nhead_stride_lse_acc  = a.num_splits * total_q;
    const ck_tile::index_t nhead_stride_o_acc    = a.num_splits * total_q * a.hdim_v;
    const ck_tile::index_t split_stride_lse_acc  = total_q;
    const ck_tile::index_t split_stride_o_acc    = total_q * a.hdim_v;

    auto sk_kargs = SplitKVKernel::MakeKargs(
        a.q_ptr, a.k_ptr, a.v_ptr, /*bias_ptr=*/nullptr,
        a.lse_acc_ptr, a.o_acc_ptr,
        a.batch, a.seqstart_q_ptr, a.seqstart_k_ptr, /*seqlen_k_ptr=*/nullptr,
        a.hdim_q, a.hdim_v, a.nhead_q, /*nhead_ratio_qk=*/a.nhead_q / a.nhead_k,
        a.num_splits,
        /*block_table_ptr=*/nullptr, /*batch_stride_block_table=*/0, /*page_block_size=*/0,
        /*is_gappy=*/false,
        a.scale_s, /*scale_p=*/1.0f, /*logits_soft_cap=*/0.0f,
        a.stride_q, a.stride_k, a.stride_v, /*stride_bias=*/0, stride_o_acc,
        a.nhead_stride_q, a.nhead_stride_k, a.nhead_stride_v, /*nhead_stride_bias=*/0,
        nhead_stride_lse_acc, nhead_stride_o_acc,
        /*batch_stride_k=*/0, /*batch_stride_v=*/0,
        split_stride_lse_acc, split_stride_o_acc,
        window_left, window_right, static_cast<ck_tile::index_t>(a.mask));

    const dim3 sk_grid = SplitKVKernel::GridSize(
        a.batch, a.nhead_q, a.nhead_k, a.max_seqlen_q, a.hdim_v, a.num_splits);
    const dim3 sk_block = SplitKVKernel::BlockSize();
    constexpr ck_tile::index_t sk_bpc = SplitKVKernel::kBlockPerCu;

    ck_tile::stream_config sc{stream, /*time_kernel=*/false};
    ck_tile::launch_kernel(
        sc, ck_tile::make_kernel<sk_bpc>(SplitKVKernel{}, sk_grid, sk_block, 0, sk_kargs));

    const ck_tile::index_t row_stride_o = a.stride_o;       // packed [q, nhead, hdim]
    const ck_tile::index_t nhead_stride_o = a.nhead_stride_o;
    auto cb_kargs = CombineKernel::MakeKargs(
        a.lse_acc_ptr, a.o_acc_ptr, /*lse_ptr=*/nullptr, a.o_ptr,
        a.batch, a.seqstart_q_ptr, a.hdim_v, a.num_splits, /*scale_o=*/1.0f,
        stride_o_acc, row_stride_o,
        nhead_stride_lse_acc, nhead_stride_o_acc, /*nhead_stride_lse=*/0, nhead_stride_o,
        split_stride_lse_acc, split_stride_o_acc);

    const dim3 cb_grid = CombineKernel::GridSize(a.batch, a.nhead_q, a.max_seqlen_q, a.hdim_v);
    const dim3 cb_block = CombineKernel::BlockSize();
    constexpr ck_tile::index_t cb_bpc = CombineKernel::kBlockPerCu;
    ck_tile::launch_kernel(
        sc, ck_tile::make_kernel<cb_bpc>(CombineKernel{}, cb_grid, cb_block, 0, cb_kargs));
}

} // namespace luminal_fmha
