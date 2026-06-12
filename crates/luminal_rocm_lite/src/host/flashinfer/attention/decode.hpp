// Decode (batch, seqlen_q == 1) attention kernel — ck_tile primitives.
//
// Role: the flash-decoding path. Uses the split-KV pipeline to parallelize a
// skinny query across KV chunks, plus a combine kernel that reduces the partial
// (o_acc, lse_acc) results across splits. Includes the num_splits heuristic
// (this is what FlashInfer's scheduler/plan does for us).
//
// FlashInfer reference: include/flashinfer/attention/decode.cuh (+ scheduler.cuh).
//
// NOTE: skeleton only — nothing implemented yet. Can be deferred: the prefill
// forward kernel handles seqlen_q == 1 functionally until this lands.

#pragma once

#include "fmha_types.hpp"

// ── ck_tile includes ──
// TODO(confirm vs fmha_fwd_kernel.hpp):
// #include <ck_tile/host.hpp>
// #include <ck_tile/ops/fmha.hpp>
// #include <ck_tile/ops/epilogue.hpp>

// ── Type tower (splitkv) ──
// TODO: TileFmhaShape (decode-tuned tile sizes)
// TODO: BlockFmhaPipelineProblem (split-kv variant)
// TODO: split-kv pipeline
// TODO: FmhaFwdSplitKVKernel
// TODO: FmhaFwdSplitKVCombineKernel

// ── num_splits heuristic ──
// TODO: choose split count from batch/heads/context vs GPU occupancy.

// ── Partials workspace ──
// TODO: size/layout for o_acc + lse_acc across splits (carved from float_ws).

// ── Host launcher ──
// TODO: launch_decode(const fmha_args&, workspace, hipStream_t)
//   - launch splitkv kernel, then combine kernel
