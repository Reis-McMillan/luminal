// Plain-HIP helper kernels for the ROCm attention path.
//
// Role: the glue kernels around the fmha launch. Most are straight CUDA->HIP
// ports of the helpers at the bottom of the original wrapper.cu (the __global__
// syntax is identical; only the stream type and fp16 intrinsics namespace
// differ). The exception is gather_cast_kv, which is new: it bridges luminal's
// paged KV pool to the contiguous, 16-bit buffers ck_tile's fmha expects.
//
// These are INTERNAL helpers (namespace luminal_fmha), not the .so ABI — the
// extern "C" plan/run surface lives in wrapper.cpp. Kernels + launchers are
// `static` so the header is safe to include without ODR concerns.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>   // __half, __float2half, __half2float
#include <cstdint>
#include <cstddef>

namespace luminal_fmha {

namespace detail {

// flat_idx[(i, kv_dim)] holds the same slot id repeated across kv_dim; recover
// the physical slot index by dividing the row's first element by kv_dim.
// (port of wrapper.cu extract_slot_indices_kernel)
static __global__ void extract_slot_indices_kernel(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c) out[i] = flat_idx[i * kv_dim] / kv_dim;
}

// Gather paged rows from an fp32 KV pool into a contiguous fp16 buffer.
// out[row, :] = (fp16) pool[slot_indices[row], :], where each row is row_dim
// elements (row_dim == num_kv_heads * head_dim). This is the paging shim:
// CK's plain fmha wants contiguous KV, so we materialize it here (gather + the
// fp32->fp16 cast fused into one pass).
static __global__ void gather_cast_kv_kernel(
    const float* pool, const int32_t* slot_indices, __half* out,
    int num_rows, int row_dim) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)num_rows * row_dim;
    if (idx >= total) return;
    int row = (int)(idx / row_dim);
    int col = (int)(idx % row_dim);
    out[idx] = __float2half(pool[(long long)slot_indices[row] * row_dim + col]);
}

// (port of wrapper.cu cast_f32_to_f16_kernel)
static __global__ void cast_f32_to_f16_kernel(const float* src, __half* dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// (port of wrapper.cu cast_f16_to_f32_kernel)
static __global__ void cast_f16_to_f32_kernel(const __half* src, float* dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// (batch, heads, dim) -> (heads, batch, dim).
// (port of wrapper.cu transpose_bhd_to_hbd_kernel)
static __global__ void transpose_bhd_to_hbd_kernel(
    const float* src, float* dst, int batch, int heads, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * dim;
    if (idx >= total) return;
    int d = idx % dim;
    int h = (idx / dim) % heads;
    int b = idx / (heads * dim);
    dst[h * batch * dim + b * dim + d] = src[idx];
}

constexpr int kThreads = 256;
static inline int blocks_for(long long total) {
    return (int)((total + kThreads - 1) / kThreads);
}

} // namespace detail

// ── Host launchers ───────────────────────────────────────────────────────────

// Paged flat gather index -> physical slot indices (length c).
static inline void extract_slot_indices(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim, hipStream_t stream) {
    if (c <= 0) return;
    detail::extract_slot_indices_kernel<<<detail::blocks_for(c), detail::kThreads, 0, stream>>>(
        flat_idx, out, c, kv_dim);
}

// Gather + cast an fp32 paged KV pool into a contiguous fp16 buffer.
static inline void gather_cast_kv(
    const float* pool, const int32_t* slot_indices, __half* out,
    int num_rows, int row_dim, hipStream_t stream) {
    long long total = (long long)num_rows * row_dim;
    if (total <= 0) return;
    detail::gather_cast_kv_kernel<<<detail::blocks_for(total), detail::kThreads, 0, stream>>>(
        pool, slot_indices, out, num_rows, row_dim);
}

// fp32 -> fp16 (e.g. casting Q at the kernel boundary).
static inline void cast_f32_to_f16(
    const float* src, __half* dst, size_t n, hipStream_t stream) {
    if (n == 0) return;
    detail::cast_f32_to_f16_kernel<<<detail::blocks_for((long long)n), detail::kThreads, 0, stream>>>(
        src, dst, n);
}

// fp16 -> fp32 (e.g. casting the kernel output back to luminal's f32).
static inline void cast_f16_to_f32(
    const __half* src, float* dst, size_t n, hipStream_t stream) {
    if (n == 0) return;
    detail::cast_f16_to_f32_kernel<<<detail::blocks_for((long long)n), detail::kThreads, 0, stream>>>(
        src, dst, n);
}

// Output transpose (batch, heads, dim) -> (heads, batch, dim). No-op shape for
// batch==1; explicit transpose otherwise, to match luminal's Sum layout.
static inline void transpose_bhd_to_hbd(
    const float* src, float* dst, int batch, int heads, int dim, hipStream_t stream) {
    long long total = (long long)batch * heads * dim;
    if (total <= 0) return;
    detail::transpose_bhd_to_hbd_kernel<<<detail::blocks_for(total), detail::kThreads, 0, stream>>>(
        src, dst, batch, heads, dim);
}

} // namespace luminal_fmha
