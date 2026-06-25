// Plain-HIP helper kernels for the ROCm attention path.
//
// Native 16-bit pipeline: the KV cache pool, Q, and the attention output are all
// stored in the kernel's element type (fp16 or bf16). The wrapper therefore never
// casts at the boundary — it moves raw 16-bit elements (uint16_t, dtype-agnostic).

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>

namespace luminal_fmha {

namespace detail {

// flat_idx[(i, kv_dim)] holds the same slot id repeated across kv_dim; recover
// the physical slot index by dividing the row's first element by kv_dim.
static __global__ void extract_slot_indices_kernel(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c) out[i] = flat_idx[i * kv_dim] / kv_dim;
}

// Gather paged rows from a 16-bit KV pool into a contiguous 16-bit buffer (decode
// path): out[row, :] = pool[slot_indices[row], :], row_dim = num_kv_heads*head_dim.
// Pure 2-byte move — dtype-agnostic (fp16/bf16). Materializes the scattered cache
// slots in CSR order so the non-paged split-KV kernel reads contiguous KV.
static __global__ void gather_rows_16_kernel(
    const uint16_t* pool, const int32_t* slot_indices, uint16_t* out,
    int num_rows, int row_dim) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)num_rows * row_dim;
    if (idx >= total) return;
    int row = (int)(idx / row_dim);
    int col = (int)(idx % row_dim);
    out[idx] = pool[(long long)slot_indices[row] * row_dim + col];
}

// (batch, heads, dim) -> (heads, batch, dim), 16-bit elements.
static __global__ void transpose_bhd_to_hbd_kernel(
    const uint16_t* src, uint16_t* dst, int batch, int heads, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * dim;
    if (idx >= total) return;
    int d = idx % dim;
    int h = (idx / dim) % heads;
    int b = idx / (heads * dim);
    dst[h * batch * dim + b * dim + d] = src[idx];
}

// Fill out[i] = i for i in [0, n). Used to synthesize the group-mode
// seqstart_q for the decode path (one query token per sequence ⇒ [0,1,..,n]).
static __global__ void fill_iota_kernel(int32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

// Build ck_tile's rectangular [batch, max_blocks] block table from luminal's ragged
// CSR page list (kv_indptr + kv_indices). For page_block_size=1, kv_indices already
// holds the per-position physical slot ids; row i of the table is sequence i's slots,
// padded with 0 past its length (the kernel only reads num_blocks_i = seqlen_k_i
// entries, so the padding is never touched). One block per batch row.
static __global__ void build_block_table_kernel(
    const int32_t* kv_indptr, const int32_t* kv_indices,
    int32_t* block_table, int batch, int max_blocks) {
    int i = blockIdx.x;
    if (i >= batch) return;
    const int start = kv_indptr[i];
    const int len   = kv_indptr[i + 1] - start;
    for (int j = threadIdx.x; j < max_blocks; j += blockDim.x) {
        block_table[(long long)i * max_blocks + j] = (j < len) ? kv_indices[start + j] : 0;
    }
}

constexpr int kThreads = 256;
static inline int blocks_for(long long total) {
    return (int)((total + kThreads - 1) / kThreads);
}

}

// Paged flat gather index -> physical slot indices (length c).
static inline void extract_slot_indices(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim, hipStream_t stream) {
    if (c <= 0) return;
    detail::extract_slot_indices_kernel<<<detail::blocks_for(c), detail::kThreads, 0, stream>>>(
        flat_idx, out, c, kv_dim);
}

// Gather a 16-bit paged KV pool into a contiguous 16-bit buffer (decode path).
static inline void gather_rows_16(
    const void* pool, const int32_t* slot_indices, void* out,
    int num_rows, int row_dim, hipStream_t stream) {
    long long total = (long long)num_rows * row_dim;
    if (total <= 0) return;
    detail::gather_rows_16_kernel<<<detail::blocks_for(total), detail::kThreads, 0, stream>>>(
        static_cast<const uint16_t*>(pool), slot_indices, static_cast<uint16_t*>(out),
        num_rows, row_dim);
}

// Fill an int32 device array with [0, 1, ..., n-1].
static inline void fill_iota(int32_t* out, int n, hipStream_t stream) {
    if (n <= 0) return;
    detail::fill_iota_kernel<<<detail::blocks_for(n), detail::kThreads, 0, stream>>>(out, n);
}

// Rectangular block table from ragged CSR (kv_indptr length batch+1, kv_indices on GPU).
static inline void build_block_table(
    const int32_t* kv_indptr, const int32_t* kv_indices,
    int32_t* block_table, int batch, int max_blocks, hipStream_t stream) {
    if (batch <= 0 || max_blocks <= 0) return;
    detail::build_block_table_kernel<<<batch, detail::kThreads, 0, stream>>>(
        kv_indptr, kv_indices, block_table, batch, max_blocks);
}

// Output transpose (batch, heads, dim) -> (heads, batch, dim), 16-bit. No-op shape
// for batch==1; explicit transpose otherwise, to match luminal's Sum layout.
static inline void transpose_bhd_to_hbd(
    const void* src, void* dst, int batch, int heads, int dim, hipStream_t stream) {
    long long total = (long long)batch * heads * dim;
    if (total <= 0) return;
    detail::transpose_bhd_to_hbd_kernel<<<detail::blocks_for(total), detail::kThreads, 0, stream>>>(
        static_cast<const uint16_t*>(src), static_cast<uint16_t*>(dst), batch, heads, dim);
}

} // namespace luminal_fmha
