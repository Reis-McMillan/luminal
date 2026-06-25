// ck_tile-based ROCm attention wrapper — implements the C ABI in wrapper.hpp.
//
// Native 16-bit pipeline: Q, the KV cache pool, and O are all stored in the kernel
// element type (fp16 or bf16, selected by -DLUMINAL_FMHA_BF16 at JIT time). The
// wrapper does NOT cast at the boundary. The two paths differ in how KV is reached:
//
//   decode  (seqlen_q == 1): gather paged slots -> contiguous 16-bit scratch, then
//           the split-KV flash-decoding kernel + combine reduction (decode.hpp).
//   prefill (varlen):        IN-KERNEL paging — k/v point straight at the 16-bit
//           pool and the kernel walks a block table built from kv_indptr/kv_indices
//           (no gather). Mirrors CUDA FlashInfer's paged_kv_t. (prefill.hpp)
//
// VARIABLE context length across the batch (ck_tile group mode): per-sequence
// boundaries come from qo_indptr / kv_indptr (seqstart pointers); the decode path
// synthesizes its trivial seqstart_q = [0,1,..,batch].
//
// Compiles against Composable Kernel (ROCm 7.2.1 ck_tile headers / pinned develop).
// Runtime correctness on RDNA/WMMA is not yet validated against a reference.

#include "wrapper.hpp"

#include "attention/fmha_types.hpp"
#include "attention/helpers.hpp"
#include "attention/prefill.hpp"
#include "attention/decode.hpp"

#include <hip/hip_runtime.h>
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

// Single-thread CSR-from-mask kernel (mask stays fp32; unused by the indptr path).
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

// Both paths now use IN-KERNEL paged KV (mirrors CUDA FlashInfer's paged_kv_t):
// k_pool/v_pool are the 16-bit cache pool and the kernel walks a block table built
// from the ragged CSR (kv_indptr + kv_indices). No host-side gather anywhere.
//   decode  → split-KV flash-decoding (seqlen_q == 1, synthesized seqstart_q)
//   prefill → plain paged forward kernel (varlen, causal)
int run_fmha(
    void* float_ws, size_t float_ws_size,
    const void* q, const void* k_pool, const void* v_pool,
    const int32_t* kv_indices,     // physical slot ids in CSR order (block-table source)
    const int32_t* kv_indptr_dev,  // device CSR offsets (block-table source)
    void* output,
    const int32_t* seqstart_q, const int32_t* seqstart_k,
    int total_q_tokens, int total_slots, int batch_size,
    int max_seqlen_q, int max_seqlen_k,
    int num_qo_heads, int num_kv_heads, int head_dim, bool is_decode,
    hipStream_t stream) {
    if (batch_size <= 0 || total_q_tokens <= 0) return 0; // nothing to do
    if (total_slots <= 0) return 0;
    if (!seqstart_k) return -3;                           // KV indptr is required

    const int kv_dim = num_kv_heads * head_dim;
    Bump bump(float_ws, float_ws_size);

    // Rectangular [batch, max_blocks] block table from the ragged CSR. page_block_size
    // = 1, so a physical slot id == a page id and the real per-row length is seqlen_k.
    //
    // Pad each row up to the kernel's KV-tile width: the paged navigator's last tile
    // spans a full kN0 columns even when seqlen_k isn't tile-aligned, so it reads
    // block_indices past the real length. Without padding that runs off the table into
    // workspace neighbours → garbage slot ids → OOB pool reads (the illegal address).
    // build_block_table zero-fills the tail; slot 0 is valid pool memory and those
    // positions are masked by kPadSeqLenK, so it's correct as well as safe. 128 covers
    // both archs' kN0 (gfx11 decode=64, gfx9 decode/prefill=128).
    const int real_max_blocks = (max_seqlen_k > 0) ? max_seqlen_k : 1;
    constexpr int kBlockTileN = 128;
    const int max_blocks =
        ((real_max_blocks + kBlockTileN - 1) / kBlockTileN) * kBlockTileN;
    int32_t* block_table = bump.take<int32_t>((size_t)batch_size * max_blocks);
    if (!block_table) return -2;
    build_block_table(kv_indptr_dev, kv_indices, block_table, batch_size, max_blocks, stream);

    fmha_args a{};
    a.q_ptr = q;
    a.k_ptr = k_pool;   // pool used directly — the kernel pages through it
    a.v_ptr = v_pool;
    a.o_ptr = output;
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

    // Q and O are contiguous packed [tokens, nhead, hdim] (NHD); no batch stride in
    // group mode (seqstart_* does the per-sequence addressing). K/V live in the paged
    // pool: stride_k is the within-page row stride, batch_stride_k the page stride.
    a.stride_q = num_qo_heads * head_dim; a.nhead_stride_q = head_dim;
    a.stride_o = num_qo_heads * head_dim; a.nhead_stride_o = head_dim;
    a.stride_k = kv_dim; a.nhead_stride_k = head_dim;
    a.stride_v = kv_dim; a.nhead_stride_v = head_dim;
    a.block_table_ptr = block_table;
    a.batch_stride_block_table = max_blocks;
    a.page_block_size = 1;
    a.batch_stride_k = kv_dim;   // stride between physical pages in the pool
    a.batch_stride_v = kv_dim;

    // Decode: the single new query attends to ALL cached KV (full attention, no causal
    // mask — a causal mask would collapse the group-mode position-0 query to key 0).
    // Prefill (varlen multi-token) keeps the causal mask.
    a.mask = is_decode ? MaskKind::None : MaskKind::Causal;

    // Stage Q into workspace scratch. The M-tile loads kM0 query rows even when
    // seqlen_q < kM0 (e.g. decode: 1 row, kM0=16), over-running a tight external Q
    // buffer → illegal address. Copying into the 128 MiB workspace gives the trailing
    // slack the over-read needs (the surplus rows are masked by kPadSeqLenQ). Q is
    // small (one token per sequence in decode), so the copy is cheap.
    {
        const size_t q_stride = (size_t)num_qo_heads * head_dim;
        const size_t q_elems = (size_t)total_q_tokens * q_stride;
        InputDataType* q_scratch = bump.take<InputDataType>(q_elems + 128 * q_stride);
        if (!q_scratch) return -2;
        hipMemcpyAsync(q_scratch, q, q_elems * sizeof(InputDataType),
                       hipMemcpyDeviceToDevice, stream);
        a.q_ptr = q_scratch;
    }

    if (is_decode) {
        // One query token per sequence: build the trivial seqstart_q = [0,1,..,batch].
        if (!a.seqstart_q_ptr) {
            int32_t* sq = bump.take<int32_t>(batch_size + 1);
            if (!sq) return -2;
            fill_iota(sq, batch_size + 1, stream);
            a.seqstart_q_ptr = sq;
        }
        // Split-KV partials (fp32 o_acc + lse_acc) carved from the workspace.
        int num_splits =
            decode_detail::choose_num_splits(batch_size, num_qo_heads, max_seqlen_q);
        // choose_num_splits sizes splits purely by occupancy and ignores seqlen_k, so
        // a short context can get more splits than it has KV tiles — the surplus splits
        // are empty and the split-KV navigation reads out of bounds (illegal address).
        // Cap by the KV-tile count, like FA/FlashInfer. A context below one kN0 tile
        // can't be split → num_splits = 1.
        const int num_kv_tiles = ck_tile::integer_divide_ceil(
            max_seqlen_k > 0 ? max_seqlen_k : 1, decode_detail::tile::kN0);
        if (num_splits > num_kv_tiles) num_splits = num_kv_tiles > 0 ? num_kv_tiles : 1;
        const size_t o_acc_elems =
            (size_t)num_qo_heads * num_splits * total_q_tokens * head_dim;
        const size_t lse_acc_elems = (size_t)num_qo_heads * num_splits * total_q_tokens;
        float* o_acc   = bump.take<float>(o_acc_elems);
        float* lse_acc = bump.take<float>(lse_acc_elems);
        if (!o_acc || !lse_acc) return -2; // workspace too small for partials
        a.num_splits  = num_splits;
        a.o_acc_ptr   = o_acc;
        a.lse_acc_ptr = lse_acc;
        launch_decode(a, stream);
    } else {
        launch_prefill(a, stream);
    }
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
    // exactly one query token per sequence, so max_seqlen_q is 1. max_seqlen_k sizes
    // the rectangular block table (page_block_size = 1).
    plan_info_out[0] = (batch_size > 0) ? (int64_t)indptr_h[batch_size] : 0;
    plan_info_out[1] = 1; // max_seqlen_q
    int max_seqlen_k = 0;
    for (int i = 0; i < batch_size; ++i) {
        const int lk = indptr_h[i + 1] - indptr_h[i];
        if (lk > max_seqlen_k) max_seqlen_k = lk;
    }
    plan_info_out[2] = max_seqlen_k;
    *plan_info_len_out = 3;
    return 0;
}

int flashinfer_batch_decode_run(
    void* float_workspace, size_t float_ws_size,
    void*,
    int64_t* plan_info_vec, int plan_info_len,
    void* q, void* k_cache, void* v_cache,
    int32_t* kv_indptr, int32_t* kv_indices, int32_t* /*kv_last_page_len*/,
    void* output,
    int batch_size,
    int num_qo_heads, int num_kv_heads, int /*page_size*/, int head_dim,
    hipStream_t stream) {
    if (plan_info_len < 3) return -1; // plan() must run first
    const int total_slots  = (int)plan_info_vec[0];
    const int max_seqlen_q = (int)plan_info_vec[1];
    const int max_seqlen_k = (int)plan_info_vec[2];
    // Decode: one query token per sequence -> seqstart_q synthesized in run_fmha;
    // kv_indptr is the per-sequence KV offset array (seqstart_k); kv_indices are the
    // physical slots the kernel pages through via the block table.
    return run_fmha(float_workspace, float_ws_size, q, k_cache, v_cache,
                    /*kv_indices=*/kv_indices, /*kv_indptr_dev=*/kv_indptr, output,
                    /*seqstart_q=*/nullptr, /*seqstart_k=*/kv_indptr,
                    /*total_q_tokens=*/batch_size, total_slots, batch_size,
                    max_seqlen_q, max_seqlen_k,
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
    const void* src, void* dst, int batch, int heads, int dim, hipStream_t stream) {
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
    // Longest query / KV segment across the batch. max_seqlen_q sizes the launch
    // grid; max_seqlen_k sizes the rectangular block table (page_block_size = 1).
    int max_seqlen_q = 0;
    int max_seqlen_k = 0;
    for (int i = 0; i < batch_size; ++i) {
        const int lq = qo_indptr_h[i + 1] - qo_indptr_h[i];
        const int lk = kv_indptr_h[i + 1] - kv_indptr_h[i];
        if (lq > max_seqlen_q) max_seqlen_q = lq;
        if (lk > max_seqlen_k) max_seqlen_k = lk;
    }
    plan_info_out[1] = max_seqlen_q;
    plan_info_out[2] = max_seqlen_k;
    *plan_info_len_out = 3;
    return 0;
}

int flashinfer_batch_prefill_run(
    void* float_workspace, size_t float_ws_size,
    void*,
    int64_t* plan_info_vec, int plan_info_len,
    void* q, void* k_cache, void* v_cache,
    int32_t* qo_indptr, int32_t* kv_indptr, int32_t* kv_indices,
    int32_t* /*kv_last_page_len*/,
    void* output,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int /*page_size*/, int head_dim,
    hipStream_t stream) {
    if (plan_info_len < 3) return -1;
    const int total_slots  = (int)plan_info_vec[0];
    const int max_seqlen_q = (int)plan_info_vec[1];
    const int max_seqlen_k = (int)plan_info_vec[2];
    // Varlen prefill: qo_indptr / kv_indptr are device seqstart arrays; kv_indices is
    // the physical slot list the kernel pages through via the block table.
    return run_fmha(float_workspace, float_ws_size, q, k_cache, v_cache,
                    /*kv_indices=*/kv_indices, /*kv_indptr_dev=*/kv_indptr, output,
                    /*seqstart_q=*/qo_indptr, /*seqstart_k=*/kv_indptr,
                    total_num_rows, total_slots, batch_size,
                    max_seqlen_q, max_seqlen_k,
                    num_qo_heads, num_kv_heads, head_dim, /*is_decode=*/false, stream);
}

} // extern "C"
