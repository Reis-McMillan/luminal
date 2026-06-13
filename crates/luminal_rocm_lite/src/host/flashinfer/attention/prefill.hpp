// Prefill (batch, uniform-length) attention kernel — ck_tile primitives.
//
// Role: the plain fmha forward pass. Assembles the ck_tile "type tower"
// (type config → tile shape → traits → pipeline problem → pipeline → epilogue
// → kernel) for one config and exposes a host launcher. A plain forward kernel
// handles any seqlen_q (including 1), so it also serves the decode path until
// the dedicated splitkv kernel in decode.hpp exists.
//
// FlashInfer reference: include/flashinfer/attention/prefill.cuh.
//
// Reconciled against Composable Kernel 1.2.0 (the ck_tile headers shipped in
// ROCm 7.2.1, /opt/rocm/include/ck_tile). Every template parameter list and the
// MakeKargs/GridSize/make_kernel arg orders below were verified field-by-field
// against those headers. Tile sizes are still a generic fp16 config (see below).

#pragma once

#include "fmha_types.hpp"

// ── ck_tile includes ──
// TODO(confirm vs include/ck_tile/ops/fmha/...): exact umbrella headers.
#include <ck_tile/host.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/epilogue.hpp>

namespace luminal_fmha {

// ── Tile sizes ─────────────────────────────────────────────────────────────
// Block-tile lengths + warp/MMA arrangement. BlockTile is sequence<kM0, kN0,
// kK0, kN1, kK1, kQKHeaddim> per TileFmhaShape (CK 1.2.0). 32x32x16 is the
// typical fp16 CDNA MFMA warp tile; this is a generic config, not per-kHeadDim
// tuned. NOTE: this warp tile targets CDNA matrix cores — RDNA3 (gfx11) uses
// WMMA and may need a different warp-tile/arch config; revisit if the gfx11
// compile rejects this.
namespace tile {
    constexpr ck_tile::index_t kM0 = 128;          // q seqlen tile
    constexpr ck_tile::index_t kN0 = 128;          // kv seqlen tile
    constexpr ck_tile::index_t kK0 = 32;           // qk gemm K step
    constexpr ck_tile::index_t kN1 = kHeadDim;     // hdim_v tile
    constexpr ck_tile::index_t kK1 = 32;           // pv gemm K step
    constexpr ck_tile::index_t kQKHeadDim = kHeadDim;
} // namespace tile

// ── Type tower (one config: fp16/bf16, kHeadDim, causal, batch mode) ─────────
using Cfg = FmhaTypeConfig;

// (1) Shape: block tile + per-gemm warp arrangement.
// TileFmhaShape<BlockTile, Gemm0BlockWarps, Gemm0WarpTile, Gemm1BlockWarps,
//               Gemm1WarpTile, IsVLayoutRowMajor>.
using FmhaShape = ck_tile::TileFmhaShape<
    ck_tile::sequence<tile::kM0, tile::kN0, tile::kK0, tile::kN1, tile::kK1, tile::kQKHeadDim>,
    ck_tile::sequence<4, 1, 1>,   ck_tile::sequence<kWarpTileM, kWarpTileN, kWarpTileK>,   // gemm0 warps / warp-tile
    ck_tile::sequence<4, 1, 1>,   ck_tile::sequence<kWarpTileM, kWarpTileN, kWarpTileK>,   // gemm1 warps / warp-tile
    /*IsVLayoutRowMajor=*/true>;

// (2) Traits: padding + feature switches. Bias/dropout/fp8 off (see fmha_types).
// TileFmhaTraits<kPadSeqLenQ, kPadSeqLenK, kPadHeadDimQ, kPadHeadDimV,
//   kHasLogitsSoftCap, BiasEnum, kHasBiasGrad, kStoreLSE, kHasDropout,
//   kDoFp8StaticQuant, kBlockPerCu=-1, kSkipMinSeqlenQ=false>.
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

// (3) Mask: causal ⇒ a masking mask type; None ⇒ non-masking.
using FmhaMask = ck_tile::SimplifiedGenericAttentionMask</*IsMasking=*/true>;

// (3b) Attention variant: plain scaled-dot-product (no bias/softcap). CK 1.2.0
// added AttentionVariant_ and kUseTrLoad_ to the pipeline problem param list.
using FmhaVariant = ck_tile::StandardAttention;

// (4) Pipeline problem: bundles dtypes + shape + mode + variant + mask + traits.
// BlockFmhaPipelineProblem<Q,K,V, Sacc, SMPLCompute, Bias, RandValOut, LSE, P,
//   Oacc, O, BlockFmhaShape, kIsGroupMode, AttentionVariant, FmhaMask,
//   kUseTrLoad, Traits>.
using FmhaPipelineProblem = ck_tile::BlockFmhaPipelineProblem<
    typename Cfg::QDataType, typename Cfg::KDataType, typename Cfg::VDataType,
    typename Cfg::SaccDataType, typename Cfg::SMPLComputeDataType,
    /*BiasDataType=*/typename Cfg::ODataType,
    /*RandValOutputDataType=*/uint8_t,
    typename Cfg::LSEDataType, typename Cfg::PDataType,
    typename Cfg::OaccDataType, typename Cfg::ODataType,
    FmhaShape, /*kIsGroupMode=*/true, FmhaVariant, FmhaMask,
    /*kUseTrLoad=*/false, FmhaTraits>;

// (5) Pipeline: the QR-KS-VS forward dataflow (default policy).
using FmhaPipeline = ck_tile::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;

// (6) Epilogue: write fp32 accumulator out as 16-bit.
// Default2DEpilogueProblem<AccDataType, ODataType, kPadM, kPadN, ...defaults>.
using FmhaEpilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename Cfg::OaccDataType, typename Cfg::ODataType,
                                      /*kPadM=*/true, /*kPadN=*/true>>;

// (7) Kernel: FmhaFwdKernel<FmhaPipeline, EpiloguePipeline>. The tile
// partitioner is built internally — there is no separate partitioner param.
using FmhaKernel = ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

// ── Host launcher ────────────────────────────────────────────────────────────
// Translates our fmha_args into the kernel's MakeKargs and launches it.
inline void launch_prefill(const fmha_args& a, hipStream_t stream) {
    // Causal window. For pure causal: left = -1 (unbounded past), right = 0.
    const ck_tile::index_t window_left  = (a.mask == MaskKind::Causal) ? -1 : -1;
    const ck_tile::index_t window_right = (a.mask == MaskKind::Causal) ?  0 : -1;

    // (a) Build kernel args (group mode — ragged batches, seqstart offsets).
    // Arg order matches the kIsGroupMode MakeKargs overload in
    // fmha_fwd_kernel.hpp (CK 1.2.0): seqstart_q/k + seqlen_q/k pointers replace
    // the scalar seqlens, there are NO batch strides (seqstart does the per-batch
    // addressing), and min_seqlen_q precedes p_drop. Passing seqlen_q/k_ptr =
    // nullptr makes the kernel derive each sequence length from seqstart (the
    // non-padded-seqlen_k tile mapping). Bias / randval / dropout / softcap off.
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

    // (b) Launch geometry: GridSize(batch, nhead, max_seqlen_q, hdim_v). The grid
    // covers the longest sequence; the kernel early-exits M-tiles past each
    // sequence's own seqstart-derived length (seqlen_q <= i_m0 ⇒ return).
    const dim3 grids = FmhaKernel::GridSize(a.batch, a.nhead_q, a.max_seqlen_q, a.hdim_v);
    const dim3 blocks = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    // (c) Launch. make_kernel's first template arg is MinBlockPerCu. The
    // dynamic-LDS byte size is 0: the kernel allocates its shared memory
    // statically (__shared__ char[GetSmemSize()]) in device context — calling
    // GetSmemSize() on the host would force CK's device-only arch dispatch
    // (get_n_lds_banks) to evaluate at host-constexpr time and fail to compile.
    ck_tile::stream_config sc{stream, /*time_kernel=*/false};
    ck_tile::launch_kernel(
        sc, ck_tile::make_kernel<kBlockPerCu>(FmhaKernel{}, grids, blocks, 0, kargs));
}

} // namespace luminal_fmha
