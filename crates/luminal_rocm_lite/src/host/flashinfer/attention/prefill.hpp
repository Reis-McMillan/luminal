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
// Block tile (kM0×kN0) and warp tile must both match a CK-generated config for
// the arch. CK's fmha_fwd_pagedkv generated set uses 128×128 block + 32×32×16
// MFMA on gfx9, and 64×64 block + 16×16×16 WMMA on gfx11.
#if defined(LUMINAL_FMHA_WMMA)
    constexpr ck_tile::index_t kWarpM = 16;
    constexpr ck_tile::index_t kWarpN = 16;
    constexpr ck_tile::index_t kWarpK = 16;
    constexpr ck_tile::index_t kM0 = 64;           // q seqlen tile  (gfx11 WMMA known-good)
    constexpr ck_tile::index_t kN0 = 64;           // kv seqlen tile
#else
    constexpr ck_tile::index_t kWarpM = 32;
    constexpr ck_tile::index_t kWarpN = 32;
    constexpr ck_tile::index_t kWarpK = 16;
    constexpr ck_tile::index_t kM0 = 128;          // q seqlen tile  (gfx9 MFMA known-good)
    constexpr ck_tile::index_t kN0 = 128;          // kv seqlen tile
#endif
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
using FmhaTraits = ck_tile::TileFmhaFwdPagedKVTraits<
    /*kPadSeqLenQ=*/true,  /*kPadSeqLenK=*/true,
    /*kPadHeadDimQ=*/true, /*kPadHeadDimV=*/true,
    /*kHasLogitsSoftCap=*/false,
    /*BiasEnum=*/ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    /*kHasBiasGrad=*/false,
    /*kStoreLSE=*/false,
    /*kIsPagedKV_=*/true,
    /*kDoFp8StaticQuant_=*/false,
    /*kBlockPerCu=*/-1,
    /*kSkipMinSeqlenQ_=*/false,
    /*kHasSink_=*/false>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/block/block_masking.hpp#L330
using FmhaMask = ck_tile::SimplifiedGenericAttentionMask</*IsMasking=*/true>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/block/variants.hpp#L137
using FmhaVariant = ck_tile::ComposedAttention<false * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp#L75
// NOTE: BlockFmhaFwdPagedKVPipelineProblem has a DIFFERENT param list than the plain
// BlockFmhaPipelineProblem — it has NO RandValOutputDataType and NO kUseTrLoad (it
// has no dropout path). Order: 10 dtypes, shape, kIsGroupMode, variant, mask, traits.
using FmhaPipelineProblem = ck_tile::BlockFmhaFwdPagedKVPipelineProblem<
    typename Cfg::QDataType, typename Cfg::KDataType, typename Cfg::VDataType,
    typename Cfg::SaccDataType, typename Cfg::SMPLComputeDataType,
    /*BiasDataType=*/typename Cfg::ODataType,
    typename Cfg::LSEDataType, typename Cfg::PDataType,
    typename Cfg::OaccDataType, typename Cfg::ODataType,
    FmhaShape, /*kIsGroupMode=*/true, FmhaVariant, FmhaMask,
    FmhaTraits>;

// Paged-KV kernel requires the paged-KV pipeline (it does the in-kernel page-block
// navigation); the plain BlockFmhaPipelineQRKSVS won't pair with FmhaFwdPagedKVKernel.
// see https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp
using FmhaPipeline = ck_tile::BlockFmhaFwdPagedKVPipelineQRKSVS<FmhaPipelineProblem>;

// see https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/ops/epilogue/default_2d_epilogue.hpp
using FmhaEpilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename Cfg::OaccDataType, typename Cfg::ODataType,
                                      /*kPadM=*/true, /*kPadN=*/true>>;

// see https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp
using FmhaKernel = ck_tile::FmhaFwdPagedKVKernel<FmhaPipeline, FmhaEpilogue>;

inline void launch_prefill(const fmha_args& a, hipStream_t stream) {
    // Causal window. For pure causal: left = -1 (unbounded past), right = 0.
    const ck_tile::index_t window_left  = (a.mask == MaskKind::Causal) ? -1 : -1;
    const ck_tile::index_t window_right = (a.mask == MaskKind::Causal) ?  0 : -1;

    // Group-mode paged-KV MakeKargs. Arg order mirrors CK's own
    // fmha_fwd_pagedkv_create_kargs_and_grids (example/ck_tile/01_fmha/fmha_fwd.hpp):
    //   …, nhead_ratio_qk, block_table_ptr, batch_stride_block_table, page_block_size,
    //   is_gappy, scale_s, scale_p, scale_o, logits_soft_cap, stride_q/k/v/bias/o,
    //   nhead_stride_q/k/v/bias/lse/o, batch_stride_k, batch_stride_v, window_l/r,
    //   sink_size, mask_type, min_seqlen_q [, sink_ptr=nullptr].
    // Note: the paged group-mode signature has NO seqlen_q_ptr and NO *_randval args.
    auto kargs = FmhaKernel::MakeKargs(
        a.q_ptr, a.k_ptr, a.v_ptr, /*bias_ptr=*/nullptr,
        a.lse_ptr, a.o_ptr,
        a.seqstart_q_ptr, a.seqstart_k_ptr, /*seqlen_k_ptr=*/nullptr,
        a.hdim_q, a.hdim_v,
        a.nhead_q, /*nhead_ratio_qk=*/a.nhead_q / a.nhead_k,
        a.block_table_ptr, a.batch_stride_block_table, a.page_block_size, /*is_gappy=*/false,
        a.scale_s, /*scale_p=*/1.0f, /*scale_o=*/1.0f, /*logits_soft_cap=*/0.0f,
        a.stride_q, a.stride_k, a.stride_v, /*stride_bias=*/0, a.stride_o,
        a.nhead_stride_q, a.nhead_stride_k, a.nhead_stride_v,
        /*nhead_stride_bias=*/0, /*nhead_stride_lse=*/0, a.nhead_stride_o,
        a.batch_stride_k, a.batch_stride_v,
        window_left, window_right,
        /*sink_size=*/0,
        /*mask_type=*/static_cast<ck_tile::index_t>(a.mask),
        /*min_seqlen_q=*/0);

    // Paged kernel GridSize takes a 5th has_padded_seqlen_k flag (= seqlen_k_ptr !=
    // nullptr); we pass per-sequence lengths via seqstart_k, not seqlen_k_ptr, so false.
    const dim3 grids =
        FmhaKernel::GridSize(a.batch, a.nhead_q, a.max_seqlen_q, a.hdim_v,
                             /*has_padded_seqlen_k=*/false);
    const dim3 blocks = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    // see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/host/stream_config.hpp#L29
    ck_tile::stream_config sc{stream, /*time_kernel=*/false};
    // see: https://github.com/ROCm/composable_kernel/blob/01cca38c8eccc490a64e631289736edda1eda720/include/ck_tile/host/kernel_launch.hpp#L303
    ck_tile::launch_kernel(
        sc, ck_tile::make_kernel<kBlockPerCu>(FmhaKernel{}, grids, blocks, 0, kargs));
}

} // namespace luminal_fmha
