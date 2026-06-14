// ck_tile-based ROCm attention wrapper — implements the C ABI in wrapper.hpp.
//
// Orchestration only: the kernels live in attention/*.hpp. run() bridges
// luminal's paged fp32 inputs to ck_tile's contiguous fp16 fmha:
//   plan_info[0] (total KV slots), plan_info[1] (max_seqlen_q) ── from plan() ──┐
//   gather paged KV pool -> contiguous fp16 (helpers::gather_cast_kv)
//   cast Q fp32 -> fp16
//   prefill -> launch_prefill (prefill.hpp);  decode -> launch_decode (decode.hpp)
//   cast O fp16 -> fp32   (caller transposes separately)
//
// ⚠️ SCOPE / ASSUMPTIONS:
//   - TWO distinct kernels: prefill uses the plain forward kernel; decode uses the
//     split-KV flash-decoding kernel + combine reduction (selected by is_decode).
//   - VARIABLE context length across the batch (ck_tile group mode): per-sequence
//     boundaries come from qo_indptr / kv_indptr (seqstart pointers); the decode
//     path synthesizes its trivial seqstart_q = [0,1,..,batch].
//   - fp16 scratch + fp32 split-KV partials are bump-allocated from float_workspace.
//   - Compiles against Composable Kernel 1.2.0 (ROCm 7.2.1 ck_tile headers) for
//     gfx1100. Runtime correctness is not yet validated against a reference.

#include "wrapper.hpp"

#include "attention/fmha_types.hpp"
#include "attention/helpers.hpp"
#include "attention/prefill.hpp"
#include "attention/decode.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>

using namespace luminal_fmha;

namespace {

// Bump allocator over a raw device byte workspace (the caller's float_workspace).
struct Bump {
    char*  base;
    size_t cap;
    size_t off = 0;

    explicit Bump(void* p, size_t c) : base(static_cast<char*>(p)), cap(c) {}

    // 256B-aligned slice of `n` elements of T; nullptr on overflow.
    template <typename T>
    T* take(size_t n) {
        size_t bytes = n * sizeof(T);
        size_t start = (off + 255) & ~size_t(255);
        if (start + bytes > cap) return nullptr;
        off = start + bytes;
        return reinterpret_cast<T*>(base + start);
    }
};

// Single-thread CSR-from-mask kernel
__global__ void derive_indptr_kernel(const float* mask, int32_t* indptr, int s, int c) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    indptr[0] = 0;
    for (int i = 0; i < s; i++) {
        int count = 0;
        for (int j = 0; j < c; j++)
            if (mask[i * c + j] > -1e9f) count++;
        indptr[i + 1] = indptr[i] + count;
    }
}

int run_fmha(
    void* float_ws, size_t float_ws_size,
    const float* q, const float* k_pool, const float* v_pool,
    const int32_t* slot_indices, float* output,
    const int32_t* seqstart_q, const int32_t* seqstart_k,
    int total_q_tokens, int total_slots, int batch_size, int max_seqlen_q,
    int num_qo_heads, int num_kv_heads, int head_dim, bool is_decode,
    hipStream_t stream) {
    if (batch_size <= 0 || total_q_tokens <= 0) return 0; // nothing to do
    if (total_slots <= 0) return 0;
    if (!seqstart_k) return -3;                           // KV indptr is required

    const int kv_dim   = num_kv_heads * head_dim;
    const int q_elems  = total_q_tokens * num_qo_heads * head_dim;
    const int kv_elems = total_slots * kv_dim;

    // fp16 scratch carved from the float workspace.
    Bump bump(float_ws, float_ws_size);
    __half* q_f16 = bump.take<__half>(q_elems);
    __half* k_f16 = bump.take<__half>(kv_elems);
    __half* v_f16 = bump.take<__half>(kv_elems);
    __half* o_f16 = bump.take<__half>(q_elems);
    if (!q_f16 || !k_f16 || !v_f16 || !o_f16) return -2; // workspace too small

    // Decode has no qo_indptr (one query token per sequence): build the trivial
    // seqstart_q = [0,1,..,batch] in scratch.
    if (!seqstart_q) {
        int32_t* sq = bump.take<int32_t>(batch_size + 1);
        if (!sq) return -2;
        fill_iota(sq, batch_size + 1, stream);
        seqstart_q = sq;
    }

    // Bridge paged fp32 -> contiguous fp16.
    cast_f32_to_f16(q, q_f16, (size_t)q_elems, stream);
    gather_cast_kv(k_pool, slot_indices, k_f16, total_slots, kv_dim, stream);
    gather_cast_kv(v_pool, slot_indices, v_f16, total_slots, kv_dim, stream);

    fmha_args a{};
    a.q_ptr = q_f16; a.k_ptr = k_f16; a.v_ptr = v_f16; a.o_ptr = o_f16;
    a.lse_ptr = nullptr;
    a.seqstart_q_ptr = seqstart_q;
    a.seqstart_k_ptr = seqstart_k;
    a.batch = batch_size;
    a.nhead_q = num_qo_heads;
    a.nhead_k = num_kv_heads;
    a.max_seqlen_q = max_seqlen_q;
    a.total_q_tokens = total_q_tokens;
    a.hdim_q = head_dim;
    a.hdim_v = head_dim;
    a.scale_s = 1.0f / std::sqrt((float)head_dim);

    // Contiguous NHD strides for the packed scratch buffers (elements). No batch
    // stride in group mode — seqstart_* does the per-sequence addressing.
    a.stride_q = num_qo_heads * head_dim; a.nhead_stride_q = head_dim;
    a.stride_k = kv_dim;                  a.nhead_stride_k = head_dim;
    a.stride_v = kv_dim;                  a.nhead_stride_v = head_dim;
    a.stride_o = num_qo_heads * head_dim; a.nhead_stride_o = head_dim;

    // Decode: the single new query attends to ALL cached KV (all in the past) — full
    // attention, NO causal mask. With a causal mask the group-mode query sits at
    // position 0 and collapses to attending only key 0 (degenerate softmax = V[0]).
    // Prefill (varlen multi-token) keeps the causal mask.
    a.mask = is_decode ? MaskKind::None : MaskKind::Causal;

    // On RDNA (WMMA) there is no dedicated WMMA prefill pipeline — the plain QR-KS-VS
    // forward kernel can't bridge the gemm0-C -> gemm1-A fragment mismatch. The split-KV
    // path (nwarp_sshuffle, P routed through LDS) handles any seqlen_q via group mode, so
    // route BOTH decode and prefill through it. CDNA keeps the dedicated prefill kernel.
    bool route_splitkv = is_decode;
#if defined(LUMINAL_FMHA_WMMA)
    route_splitkv = true;
#endif

    if (route_splitkv) {
        // Split-KV flash-decoding: pick a split count, carve fp32 partials
        // (o_acc + lse_acc) from the workspace, then run splitkv + combine.
        const int num_splits =
            decode_detail::choose_num_splits(batch_size, num_qo_heads, max_seqlen_q);
        const size_t o_acc_elems =
            (size_t)num_qo_heads * num_splits * total_q_tokens * head_dim;
        const size_t lse_acc_elems = (size_t)num_qo_heads * num_splits * total_q_tokens;
        float* o_acc   = bump.take<float>(o_acc_elems);
        float* lse_acc = bump.take<float>(lse_acc_elems);
        if (!o_acc || !lse_acc) return -2; // workspace too small for partials
        a.num_splits   = num_splits;
        a.o_acc_ptr    = o_acc;
        a.lse_acc_ptr  = lse_acc;
        launch_decode(a, stream);
    }
#if !defined(LUMINAL_FMHA_WMMA)
    else {
        // launch_prefill instantiates the MFMA-only forward kernel; compile it only on
        // non-WMMA archs so its gemm0-C -> gemm1-A register reuse never instantiates here.
        launch_prefill(a, stream);
    }
#endif

    // fp16 output -> fp32 (batch, heads, dim); caller transposes to (heads, batch, dim).
    cast_f16_to_f32(o_f16, output, (size_t)q_elems, stream);
    return 0;
}

} // namespace

extern "C" {

int flashinfer_batch_decode_plan(
    void*, size_t, void*, size_t, void*,
    int32_t* indptr_h, int batch_size,
    int, int, int, int,
    hipStream_t,
    int64_t* plan_info_out, int* plan_info_len_out) {
    // Stash total KV slots (CSR end) for run(); no scheduler needed. Decode has
    // exactly one query token per sequence, so max_seqlen_q is 1.
    plan_info_out[0] = (batch_size > 0) ? (int64_t)indptr_h[batch_size] : 0;
    plan_info_out[1] = 1; // max_seqlen_q
    *plan_info_len_out = 2;
    return 0;
}

int flashinfer_batch_decode_run(
    void* float_workspace, size_t float_ws_size,
    void*,
    int64_t* plan_info_vec, int plan_info_len,
    float* q, float* k_cache, float* v_cache,
    int32_t* kv_indptr, int32_t* kv_indices, int32_t* /*kv_last_page_len*/,
    float* output,
    int batch_size,
    int num_qo_heads, int num_kv_heads, int /*page_size*/, int head_dim,
    hipStream_t stream) {
    if (plan_info_len < 2) return -1; // plan() must run first
    const int total_slots   = (int)plan_info_vec[0];
    const int max_seqlen_q  = (int)plan_info_vec[1];
    // Decode: one query token per sequence -> seqstart_q synthesized in run_fmha;
    // kv_indptr is the per-sequence KV offset array (seqstart_k).
    return run_fmha(float_workspace, float_ws_size, q, k_cache, v_cache,
                    kv_indices, output,
                    /*seqstart_q=*/nullptr, /*seqstart_k=*/kv_indptr,
                    /*total_q_tokens=*/batch_size, total_slots, batch_size, max_seqlen_q,
                    num_qo_heads, num_kv_heads, head_dim, /*is_decode=*/true, stream);
}

void flashinfer_extract_slot_indices(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim, hipStream_t stream) {
    extract_slot_indices(flat_idx, out, c, kv_dim, stream);
}

void flashinfer_derive_indptr_from_mask(
    const float* mask, int32_t* indptr, int s, int c, hipStream_t stream) {
    if (s <= 0) return;
    derive_indptr_kernel<<<1, 1, 0, stream>>>(mask, indptr, s, c);
}

void flashinfer_transpose_output(
    const float* src, float* dst, int batch, int heads, int dim, hipStream_t stream) {
    transpose_bhd_to_hbd(src, dst, batch, heads, dim, stream);
}

int flashinfer_batch_prefill_plan(
    void*, size_t, void*, size_t, void*,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h,
    int, int batch_size,
    int, int, int, int,
    hipStream_t,
    int64_t* plan_info_out, int* plan_info_len_out) {
    plan_info_out[0] = (batch_size > 0) ? (int64_t)kv_indptr_h[batch_size] : 0;
    // Longest query segment across the batch - sizes the launch grid in run().
    int max_seqlen_q = 0;
    for (int i = 0; i < batch_size; ++i) {
        const int len = qo_indptr_h[i + 1] - qo_indptr_h[i];
        if (len > max_seqlen_q) max_seqlen_q = len;
    }
    plan_info_out[1] = max_seqlen_q;
    *plan_info_len_out = 2;
    return 0;
}

int flashinfer_batch_prefill_run(
    void* float_workspace, size_t float_ws_size,
    void*,
    int64_t* plan_info_vec, int plan_info_len,
    float* q, float* k_cache, float* v_cache,
    int32_t* qo_indptr, int32_t* kv_indptr, int32_t* kv_indices,
    int32_t* /*kv_last_page_len*/,
    float* output,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int /*page_size*/, int head_dim,
    hipStream_t stream) {
    if (plan_info_len < 2) return -1;
    const int total_slots  = (int)plan_info_vec[0];
    const int max_seqlen_q = (int)plan_info_vec[1];
    // Varlen prefill: qo_indptr / kv_indptr are the device seqstart arrays.
    return run_fmha(float_workspace, float_ws_size, q, k_cache, v_cache,
                    kv_indices, output,
                    /*seqstart_q=*/qo_indptr, /*seqstart_k=*/kv_indptr,
                    total_num_rows, total_slots, batch_size, max_seqlen_q,
                    num_qo_heads, num_kv_heads, head_dim, /*is_decode=*/false, stream);
}

} // extern "C"
