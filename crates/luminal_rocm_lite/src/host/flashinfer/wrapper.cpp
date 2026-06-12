// ck_tile-based ROCm attention wrapper — implements the C ABI in wrapper.hpp.
//
// Orchestration only: the kernels live in attention/*.hpp. run() bridges
// luminal's paged fp32 inputs to ck_tile's contiguous fp16 fmha:
//   plan_info[0] (total KV slots)  ── from plan() ──┐
//   gather paged KV pool -> contiguous fp16 (helpers::gather_cast_kv)
//   cast Q fp32 -> fp16
//   launch_prefill (attention/prefill.hpp)
//   cast O fp16 -> fp32   (caller transposes separately)
//
// ⚠️ v1 SCOPE / ASSUMPTIONS:
//   - One forward kernel serves both decode and prefill (seqlen_q == 1 is fine).
//   - UNIFORM context length across the batch: seqlen_k = total_slots / batch.
//     Correct for batch_size == 1; variable-length batches need ck_tile group
//     mode (seqstart pointers) — see TODO(group-mode).
//   - fp16 scratch is bump-allocated from the caller's float_workspace.
//   - Depends on prefill.hpp, whose ck_tile signatures are still being
//     reconciled — this file will not compile until that lands.

#include "wrapper.hpp"

#include "attention/fmha_types.hpp"
#include "attention/helpers.hpp"
#include "attention/prefill.hpp"

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

// Single-thread CSR-from-mask kernel (port of wrapper.cu derive_indptr_kernel).
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

// Shared fmha orchestration for both decode and (uniform) prefill.
// total_q_tokens = query rows; total_slots = gathered KV rows (from plan_info).
int run_fmha(
    void* float_ws, size_t float_ws_size,
    const float* q, const float* k_pool, const float* v_pool,
    const int32_t* slot_indices, float* output,
    int total_q_tokens, int total_slots, int batch_size,
    int num_qo_heads, int num_kv_heads, int head_dim,
    hipStream_t stream) {
    if (batch_size <= 0 || total_q_tokens <= 0) return 0; // nothing to do
    if (total_slots <= 0) return 0;

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

    // Bridge paged fp32 -> contiguous fp16.
    cast_f32_to_f16(q, q_f16, (size_t)q_elems, stream);
    gather_cast_kv(k_pool, slot_indices, k_f16, total_slots, kv_dim, stream);
    gather_cast_kv(v_pool, slot_indices, v_f16, total_slots, kv_dim, stream);

    // Uniform per-sequence lengths. TODO(group-mode): for varlen batches, switch
    // to ck_tile group mode driven by qo_indptr / kv_indptr instead of this.
    const int seqlen_q = total_q_tokens / batch_size;
    const int seqlen_k = total_slots / batch_size;

    fmha_args a{};
    a.q_ptr = q_f16; a.k_ptr = k_f16; a.v_ptr = v_f16; a.o_ptr = o_f16;
    a.lse_ptr = nullptr;
    a.batch = batch_size;
    a.nhead_q = num_qo_heads;
    a.nhead_k = num_kv_heads;
    a.seqlen_q = seqlen_q;
    a.seqlen_k = seqlen_k;
    a.hdim_q = head_dim;
    a.hdim_v = head_dim;
    a.scale_s = 1.0f / std::sqrt((float)head_dim);

    // Contiguous NHD strides for the scratch buffers (elements).
    a.stride_q = num_qo_heads * head_dim; a.nhead_stride_q = head_dim;
    a.batch_stride_q = seqlen_q * num_qo_heads * head_dim;
    a.stride_k = kv_dim;                  a.nhead_stride_k = head_dim;
    a.batch_stride_k = seqlen_k * kv_dim;
    a.stride_v = kv_dim;                  a.nhead_stride_v = head_dim;
    a.batch_stride_v = seqlen_k * kv_dim;
    a.stride_o = num_qo_heads * head_dim; a.nhead_stride_o = head_dim;
    a.batch_stride_o = seqlen_q * num_qo_heads * head_dim;

    a.mask = MaskKind::Causal;

    launch_prefill(a, stream);

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
    // Stash total KV slots (CSR end) for run(); no scheduler needed.
    plan_info_out[0] = (batch_size > 0) ? (int64_t)indptr_h[batch_size] : 0;
    *plan_info_len_out = 1;
    return 0;
}

int flashinfer_batch_decode_run(
    void* float_workspace, size_t float_ws_size,
    void*,
    int64_t* plan_info_vec, int plan_info_len,
    float* q, float* k_cache, float* v_cache,
    int32_t* /*kv_indptr*/, int32_t* kv_indices, int32_t* /*kv_last_page_len*/,
    float* output,
    int batch_size,
    int num_qo_heads, int num_kv_heads, int /*page_size*/, int head_dim,
    hipStream_t stream) {
    if (plan_info_len < 1) return -1; // plan() must run first
    const int total_slots = (int)plan_info_vec[0];
    // Decode: one query token per sequence.
    return run_fmha(float_workspace, float_ws_size, q, k_cache, v_cache,
                    kv_indices, output,
                    /*total_q_tokens=*/batch_size, total_slots, batch_size,
                    num_qo_heads, num_kv_heads, head_dim, stream);
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
    int32_t*, int32_t* kv_indptr_h,
    int, int batch_size,
    int, int, int, int,
    hipStream_t,
    int64_t* plan_info_out, int* plan_info_len_out) {
    plan_info_out[0] = (batch_size > 0) ? (int64_t)kv_indptr_h[batch_size] : 0;
    *plan_info_len_out = 1;
    return 0;
}

int flashinfer_batch_prefill_run(
    void* float_workspace, size_t float_ws_size,
    void*,
    int64_t* plan_info_vec, int plan_info_len,
    float* q, float* k_cache, float* v_cache,
    int32_t* /*qo_indptr*/, int32_t* /*kv_indptr*/, int32_t* kv_indices,
    int32_t* /*kv_last_page_len*/,
    float* output,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int /*page_size*/, int head_dim,
    hipStream_t stream) {
    if (plan_info_len < 1) return -1;
    const int total_slots = (int)plan_info_vec[0];
    return run_fmha(float_workspace, float_ws_size, q, k_cache, v_cache,
                    kv_indices, output,
                    total_num_rows, total_slots, batch_size,
                    num_qo_heads, num_kv_heads, head_dim, stream);
}

} // extern "C"
