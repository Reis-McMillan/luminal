#pragma once

#include "fmha_types.hpp"

// ── ck_tile includes ──
#include <ck_tile/host.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/epilogue.hpp>

namespace luminal_fmha {

namespace tile {
// ── Warp tile (one matrix-core MMA instruction, M×N×K) ──
// CDNA (gfx9, the production target) uses MFMA 32×32×16. RDNA3/RDNA4 (gfx11/12)
// have no 32×32×16 instruction — they use WMMA 16×16×16, and an MFMA-shaped tile
// silently produces zeros there. jit.rs passes -DLUMINAL_FMHA_WMMA when the
// detected arch is a WMMA arch, so the same source is correct on both: tuned for
// CDNA, testable on a gfx11 dev card.
#if defined(LUMINAL_FMHA_WMMA)
    constexpr ck_tile::index_t kWarpM = 16;
    constexpr ck_tile::index_t kWarpN = 16;
    constexpr ck_tile::index_t kWarpK = 16;
#else
    constexpr ck_tile::index_t kWarpM = 32;
    constexpr ck_tile::index_t kWarpN = 32;
    constexpr ck_tile::index_t kWarpK = 16;
#endif
    constexpr ck_tile::index_t kM0 = 128;          // q seqlen tile
    constexpr ck_tile::index_t kN0 = 128;          // kv seqlen tile
    constexpr ck_tile::index_t kK0 = 32;           // qk gemm K step
    constexpr ck_tile::index_t kN1 = kHeadDim;     // hdim_v tile
    constexpr ck_tile::index_t kK1 = 32;           // pv gemm K step
    constexpr ck_tile::index_t kQKHeadDim = kHeadDim;
} // namespace tile

using Cfg = FmhaTypeConfig;

// see: https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp
using FmhaShape = ck_tile::TileFmhaShape<
    ck_tile::sequence<tile::kM0, tile::kN0, tile::kK0, tile::kN1, tile::kK1, tile::kQKHeadDim>,
    ck_tile::sequence<4, 1, 1>,   ck_tile::sequence<tile::kWarpM, tile::kWarpN, tile::kWarpK>,   // gemm0 warps / warp-tile
    ck_tile::sequence<4, 1, 1>,   ck_tile::sequence<tile::kWarpM, tile::kWarpN, tile::kWarpK>,   // gemm1 warps / warp-tile
    /*IsVLayoutRowMajor=*/true>;

// see https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp
using FmhaTraits = ck_tile::TileFmhaTraits<
    /*kPadSeqLenQ=*/true,  /*kPadSeqLenK=*/true,
    /*kPadHeadDimQ=*/true, /*kPadHeadDimV=*/true,
    /*kHasLogitsSoftCap=*/false,
    /*BiasEnum=*/ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    /*kHasBiasGrad=*/false,
    /*kStoreLSE=*/false,
    /*kHasDropout=*/false,
    /*kDoFp8StaticQuant=*/false,
    /*kBlockPerCu=*/-1>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/block/block_masking.hpp#L330
using FmhaMask = ck_tile::SimplifiedGenericAttentionMask</*IsMasking=*/true>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/block/variants.hpp#L137
using FmhaVariant = ck_tile::StandardAttention;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp#L75
using FmhaPipelineProblem = ck_tile::BlockFmhaPipelineProblem<
    typename Cfg::QDataType, typename Cfg::KDataType, typename Cfg::VDataType,
    typename Cfg::SaccDataType, typename Cfg::SMPLComputeDataType,
    /*BiasDataType=*/typename Cfg::ODataType,
    /*RandValOutputDataType=*/uint8_t,
    typename Cfg::LSEDataType, typename Cfg::PDataType,
    typename Cfg::OaccDataType, typename Cfg::ODataType,
    FmhaShape, /*kIsGroupMode=*/true, FmhaVariant, FmhaMask,
    /*kUseTrLoad=*/false, FmhaTraits>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp
using FmhaPipeline = ck_tile::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/epilogue/default_2d_epilogue.hpp
using FmhaEpilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename Cfg::OaccDataType, typename Cfg::ODataType,
                                      /*kPadM=*/true, /*kPadN=*/true>>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp#L113
using FmhaKernel = ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

inline void launch_prefill(const fmha_args& a, hipStream_t stream) {
    // Causal window. For pure causal: left = -1 (unbounded past), right = 0.
    const ck_tile::index_t window_left  = (a.mask == MaskKind::Causal) ? -1 : -1;
    const ck_tile::index_t window_right = (a.mask == MaskKind::Causal) ?  0 : -1;

    auto kargs = FmhaKernel::MakeKargs(
        a.q_ptr, a.k_ptr, a.v_ptr, /*bias_ptr=*/nullptr, /*rand_val_ptr=*/nullptr,
        a.lse_ptr, a.o_ptr,
        a.seqstart_q_ptr, a.seqstart_k_ptr,
        /*seqlen_q_ptr=*/nullptr, /*seqlen_k_ptr=*/nullptr,
        a.hdim_q, a.hdim_v,
        a.nhead_q, /*nhead_ratio_qk=*/a.nhead_q / a.nhead_k,
        a.scale_s, /*scale_p=*/1.0f, /*scale_o=*/1.0f, /*logits_soft_cap=*/0.0f,
        a.stride_q, a.stride_k, a.stride_v, /*stride_bias=*/0, /*stride_randval=*/0, a.stride_o,
        a.nhead_stride_q, a.nhead_stride_k, a.nhead_stride_v,
        /*nhead_stride_bias=*/0, /*nhead_stride_randval=*/0, /*nhead_stride_lse=*/0,
        a.nhead_stride_o,
        window_left, window_right,
        /*mask_type=*/static_cast<ck_tile::index_t>(a.mask),
        /*min_seqlen_q=*/0,
        /*p_drop=*/0.0f, /*s_randval=*/false,
        /*drop_seed_offset=*/std::make_tuple<uint64_t, uint64_t>(0, 0));

    const dim3 grids = FmhaKernel::GridSize(a.batch, a.nhead_q, a.max_seqlen_q, a.hdim_v);
    const dim3 blocks = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    // see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/host/stream_config.hpp#L29
    ck_tile::stream_config sc{stream, /*time_kernel=*/false};
    // see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/host/kernel_launch.hpp#L303
    ck_tile::launch_kernel(
        sc, ck_tile::make_kernel<kBlockPerCu>(FmhaKernel{}, grids, blocks, 0, kargs));
}

} // namespace luminal_fmha
