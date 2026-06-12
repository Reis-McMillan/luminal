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
// ⚠️ FIRST-DRAFT STATUS: the class names and assembly *order* below are real
// ck_tile, but every template PARAMETER LIST and the MakeKargs ARG ORDER are
// marked // TODO(confirm) — these drift between CK commits and must be checked
// against the fmha headers at CK_GIT_REV before this compiles. Treat the tile
// sizes as placeholders to replace with the codegen table values for kHeadDim.

#pragma once

#include "fmha_types.hpp"

// ── ck_tile includes ──
// TODO(confirm vs include/ck_tile/ops/fmha/...): exact umbrella headers.
#include <ck_tile/host.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/epilogue.hpp>

namespace luminal_fmha {

// ── Tile sizes ─────────────────────────────────────────────────────────────
// Block-tile lengths + warp/MFMA arrangement. THESE ARE PLACEHOLDERS: pull the
// tuned values for kHeadDim from CK's codegen tables (example/ck_tile/01_fmha/
// codegen). 32x32x16 is the typical fp16 CDNA MFMA warp tile.
// TODO(confirm): sequence layout/length expected by TileFmhaShape.
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
// TODO(confirm): TileFmhaShape parameter list (block tile seq, gemm0/gemm1
// block-warps + warp-tile seqs, IsVLayoutRowMajor).
using FmhaShape = ck_tile::TileFmhaShape<
    ck_tile::sequence<tile::kM0, tile::kN0, tile::kK0, tile::kN1, tile::kK1, tile::kQKHeadDim>,
    ck_tile::sequence<4, 1, 1>,   ck_tile::sequence<32, 32, 16>,   // gemm0 warps / warp-tile
    ck_tile::sequence<4, 1, 1>,   ck_tile::sequence<32, 32, 16>,   // gemm1 warps / warp-tile
    /*IsVLayoutRowMajor=*/true>;

// (2) Traits: padding + feature switches. Bias/dropout/fp8 off (see fmha_types).
// TODO(confirm): TileFmhaTraits parameter list & order.
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
// TODO(confirm): mask class name (SimplifiedGenericAttentionMask vs GenericAttentionMask).
using FmhaMask = ck_tile::SimplifiedGenericAttentionMask</*IsMasking=*/true>;

// (4) Pipeline problem: bundles dtypes + shape + mode + traits + mask.
// TODO(confirm): BlockFmhaPipelineProblem parameter list & order.
using FmhaPipelineProblem = ck_tile::BlockFmhaPipelineProblem<
    typename Cfg::QDataType, typename Cfg::KDataType, typename Cfg::VDataType,
    typename Cfg::SaccDataType, typename Cfg::SMPLComputeDataType,
    /*BiasDataType=*/typename Cfg::ODataType,
    /*RandValOutputDataType=*/uint8_t,
    typename Cfg::LSEDataType, typename Cfg::PDataType,
    typename Cfg::OaccDataType, typename Cfg::ODataType,
    FmhaShape, /*kIsGroupMode=*/false, FmhaMask, FmhaTraits>;

// (5) Pipeline: the QR-KS-VS forward dataflow.
// TODO(confirm): pipeline class (QRKSVS vs QRKSVSAsync) for the chosen config.
using FmhaPipeline = ck_tile::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;

// (6) Epilogue: write fp32 accumulator out as 16-bit.
// TODO(confirm): Default2DEpilogue / its problem parameter list.
using FmhaEpilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename Cfg::OaccDataType, typename Cfg::ODataType,
                                      /*kPadSeqLenQ=*/true, /*kPadHeadDimV=*/true>>;

// (7) Kernel: tile partitioner + pipeline + epilogue.
// TODO(confirm): FmhaFwdTilePartitioner / FmhaFwdKernel template params.
using FmhaKernel = ck_tile::FmhaFwdKernel<
    ck_tile::FmhaFwdTilePartitioner<FmhaShape>, FmhaPipeline, FmhaEpilogue>;

// ── Host launcher ────────────────────────────────────────────────────────────
// Translates our fmha_args into the kernel's MakeKargs and launches it.
inline void launch_prefill(const fmha_args& a, hipStream_t stream) {
    // Causal window. For pure causal: left = -1 (unbounded past), right = 0.
    const ck_tile::index_t window_left  = (a.mask == MaskKind::Causal) ? -1 : -1;
    const ck_tile::index_t window_right = (a.mask == MaskKind::Causal) ?  0 : -1;

    // (a) Build kernel args (batch mode — uniform seqlen, per-batch strides).
    // TODO(confirm): exact MakeKargs signature & arg ORDER (this is the most
    // error-prone line — verify field-by-field against fmha_fwd_kernel.hpp).
    auto kargs = FmhaKernel::MakeKargs(
        a.q_ptr, a.k_ptr, a.v_ptr, /*bias_ptr=*/nullptr, /*rand_val_ptr=*/nullptr,
        a.lse_ptr, a.o_ptr,
        a.seqlen_q, a.seqlen_k, a.batch,
        /*max_seqlen_q=*/a.seqlen_q,
        a.hdim_q, a.hdim_v, a.nhead_q, a.nhead_q / a.nhead_k,
        a.scale_s, /*scale_p=*/1.0f, /*scale_o=*/1.0f,
        a.stride_q, a.stride_k, a.stride_v, /*stride_bias=*/0, a.stride_o,
        a.nhead_stride_q, a.nhead_stride_k, a.nhead_stride_v,
        /*nhead_stride_bias=*/0, /*nhead_stride_lse=*/0, a.nhead_stride_o,
        a.batch_stride_q, a.batch_stride_k, a.batch_stride_v,
        /*batch_stride_bias=*/0, /*batch_stride_lse=*/0, a.batch_stride_o,
        window_left, window_right,
        /*mask_type=*/static_cast<ck_tile::index_t>(a.mask));

    // (b) Launch geometry.
    // TODO(confirm): GridSize argument list.
    const dim3 grids = FmhaKernel::GridSize(a.batch, a.nhead_q, a.seqlen_q, a.hdim_v);
    constexpr dim3 blocks = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    // (c) Launch.
    ck_tile::stream_config sc{stream, /*time_kernel=*/false};
    ck_tile::launch_kernel(
        sc, ck_tile::make_kernel<blocks.x, kBlockPerCu>(FmhaKernel{}, grids, blocks, 0, kargs));
}

} // namespace luminal_fmha
