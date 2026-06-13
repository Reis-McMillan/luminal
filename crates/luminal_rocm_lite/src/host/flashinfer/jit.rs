//! JIT compilation and dynamic loading of the ck_tile attention kernels.
//!
//! Everything runs at compile / profiling time — there is no `build.rs`.
//! `wrapper.cpp` / `wrapper.hpp` and the `attention/*.hpp` kernels are embedded
//! via `include_str!()` and extracted to the cache directory on first use. The
//! Composable Kernel header tree is located by probing `LUMINAL_CK_DIR`, the
//! system ROCm install, and (as a last resort) by `git clone`-ing CK at a
//! pinned commit into the cache. `hipcc` is then invoked with the model's
//! actual `HEAD_DIM` and the resulting `.so` is `dlopen`'d.
//!
//! `ensure_compiled` is called from `FlashInferAttention::extract()`, i.e.
//! during luminal's compile / GA-profiling phase, not from `execute()`. After
//! the first call the `OnceLock` makes subsequent lookups free.

use std::{
    ffi::c_void,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

// ── Function pointer types matching wrapper.hpp ──

pub type PlanFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    int_ws_size: usize,
    page_locked_int_workspace: *mut c_void,
    indptr_h: *mut i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    stream: *mut c_void,
    plan_info_out: *mut i64,
    plan_info_len_out: *mut i32,
) -> i32;

pub type RunFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    q: *mut f32,
    k_cache: *mut f32,
    v_cache: *mut f32,
    kv_indptr: *mut i32,
    kv_indices: *mut i32,
    kv_last_page_len: *mut i32,
    output: *mut f32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> i32;

pub type ExtractFn = unsafe extern "C" fn(
    flat_idx: *const i32,
    out: *mut i32,
    c: i32,
    kv_dim: i32,
    stream: *mut c_void,
);

pub type DeriveIndptrFn =
    unsafe extern "C" fn(mask: *const f32, indptr: *mut i32, s: i32, c: i32, stream: *mut c_void);

pub type TransposeOutputFn = unsafe extern "C" fn(
    src: *const f32,
    dst: *mut f32,
    batch: i32,
    heads: i32,
    dim: i32,
    stream: *mut c_void,
);

pub type PrefillPlanFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    int_ws_size: usize,
    page_locked_int_workspace: *mut c_void,
    qo_indptr_h: *mut i32,
    kv_indptr_h: *mut i32,
    total_num_rows: i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    stream: *mut c_void,
    plan_info_out: *mut i64,
    plan_info_len_out: *mut i32,
) -> i32;

pub type PrefillRunFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    q: *mut f32,
    k_cache: *mut f32,
    v_cache: *mut f32,
    qo_indptr: *mut i32,
    kv_indptr: *mut i32,
    kv_indices: *mut i32,
    kv_last_page_len: *mut i32,
    output: *mut f32,
    total_num_rows: i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> i32;

// ── Embedded wrapper + kernel sources ──
// Self-contained: every source is include_str!'d into the binary and written to
// the cache dir at JIT time, so no source tree is needed on the target machine.

const WRAPPER_CPP: &str = include_str!("wrapper.cpp");
const WRAPPER_HPP: &str = include_str!("wrapper.hpp");
const ATTN_FMHA_TYPES: &str = include_str!("attention/fmha_types.hpp");
const ATTN_HELPERS: &str = include_str!("attention/helpers.hpp");
const ATTN_PREFILL: &str = include_str!("attention/prefill.hpp");
const ATTN_DECODE: &str = include_str!("attention/decode.hpp");

// ── Loaded library handle ──

pub struct FlashInferLib {
    // Keep the handle alive so the dlopen'd .so remains mapped.
    _lib: libloading::Library,
    pub plan: PlanFn,
    pub run: RunFn,
    pub extract_slot_indices: ExtractFn,
    pub derive_indptr_from_mask: DeriveIndptrFn,
    pub transpose_output: TransposeOutputFn,
    pub prefill_plan: PrefillPlanFn,
    pub prefill_run: PrefillRunFn,
}

// SAFETY: The library handle and function pointers are valid for the lifetime
// of the process. All functions are called with proper HIP stream serialization.
unsafe impl Send for FlashInferLib {}
unsafe impl Sync for FlashInferLib {}

static FLASHINFER_LIB: OnceLock<FlashInferLib> = OnceLock::new();

/// Ensure the FlashInfer library is compiled and loaded for the given HEAD_DIM.
/// Returns a reference to the loaded library. Thread-safe via OnceLock.
pub fn ensure_compiled(head_dim: usize) -> &'static FlashInferLib {
    FLASHINFER_LIB.get_or_init(|| {
        assert!(
            matches!(head_dim, 64 | 128 | 256),
            "FlashInfer: unsupported HEAD_DIM={} (must be 64, 128, or 256)",
            head_dim
        );
        let so_path = compile_or_cache(head_dim);
        unsafe {
            FlashInferLib::load(&so_path)
                .unwrap_or_else(|e| panic!("Failed to load FlashInfer library: {e}"))
        }
    })
}

impl FlashInferLib {
    /// Load a compiled FlashInfer .so and resolve function pointers.
    ///
    /// # Safety
    /// The .so must be a valid wrapper compiled from wrapper.cpp.
    unsafe fn load(path: &Path) -> Result<Self, libloading::Error> {
        let lib = unsafe { libloading::Library::new(path)? };
        let plan: PlanFn = unsafe { *lib.get::<PlanFn>(b"flashinfer_batch_decode_plan\0")? };
        let run: RunFn = unsafe { *lib.get::<RunFn>(b"flashinfer_batch_decode_run\0")? };
        let extract_slot_indices: ExtractFn =
            unsafe { *lib.get::<ExtractFn>(b"flashinfer_extract_slot_indices\0")? };
        let derive_indptr_from_mask: DeriveIndptrFn =
            unsafe { *lib.get::<DeriveIndptrFn>(b"flashinfer_derive_indptr_from_mask\0")? };
        let transpose_output: TransposeOutputFn =
            unsafe { *lib.get::<TransposeOutputFn>(b"flashinfer_transpose_output\0")? };
        let prefill_plan: PrefillPlanFn =
            unsafe { *lib.get::<PrefillPlanFn>(b"flashinfer_batch_prefill_plan\0")? };
        let prefill_run: PrefillRunFn =
            unsafe { *lib.get::<PrefillRunFn>(b"flashinfer_batch_prefill_run\0")? };
        Ok(Self {
            _lib: lib,
            plan,
            run,
            extract_slot_indices,
            derive_indptr_from_mask,
            transpose_output,
            prefill_plan,
            prefill_run,
        })
    }
}

/// Compile wrapper.cpp for the given HEAD_DIM, or return cached .so path.
fn compile_or_cache(head_dim: usize) -> PathBuf {
    let cache_dir = cache_directory();
    std::fs::create_dir_all(&cache_dir).expect("Failed to create FlashInfer cache directory");

    // Extract bundled wrapper + kernel sources to the cache so hipcc can compile them.
    let (wrapper_src_path, include_dir) = extract_wrapper_sources(&cache_dir);

    let arch = detect_rocm_arch();
    // Bake a hash of the embedded sources into the .so name so old caches are
    // discarded automatically when any embedded wrapper/kernel source changes.
    let wrapper_hash = wrapper_source_hash();
    let so_name = format!(
        "libflashinfer_hd{}_{}_w{:016x}.so",
        head_dim, arch, wrapper_hash
    );
    let so_path = cache_dir.join(&so_name);

    if so_path.exists() {
        eprintln!(
            "FlashInfer: using cached library for HEAD_DIM={} ({})",
            head_dim,
            so_path.display()
        );
        return so_path;
    }

    let Some(ck_include) = locate_ck_includes() else {
        panic!(
            "Composable Kernel: could not locate header tree. Set LUMINAL_CK_DIR to a CK \
             source/install root (the directory containing `include/ck_tile/`), or install \
             ROCm with the CK development headers under /opt/rocm/include."
        );
    };

    eprintln!(
        "FlashInfer: JIT compiling for HEAD_DIM={}, arch={} ...",
        head_dim, arch
    );
    let start = std::time::Instant::now();

    // The fmha warp tile differs by matrix-core ISA: CDNA (gfx9, the target) uses
    // MFMA 32x32x16; RDNA3/RDNA4 (gfx11/gfx12) use WMMA 16x16x16 (an MFMA-shaped
    // tile silently produces zeros there). The arch-conditional tile lives in
    // fmha_types.hpp behind LUMINAL_FMHA_WMMA.
    //
    // NOTE: the WMMA path does NOT yet build — ck_tile's QR-KS-VS forward pipeline
    // keeps the softmax probabilities P in registers between gemm0 and gemm1, and
    // that C→A distribution reuse is only implemented for MFMA, not WMMA (gemm1
    // fails to instantiate). So we do NOT pass -DLUMINAL_FMHA_WMMA: gfx11/12 builds
    // with the MFMA tile (compiles, but fmha numerics are only valid on CDNA).
    // Re-enable once a WMMA-compatible fwd config lands. TODO(wmma-fmha).
    let _wmma_arch = arch.starts_with("gfx11") || arch.starts_with("gfx12");
    let mut args: Vec<String> = vec![
        "-shared".into(),
        "-o".into(),
        so_path.to_str().unwrap().into(),
        format!("-DLUMINAL_HEAD_DIM={head_dim}"),
        // -x hip forces HIP (device) compilation of the .cpp; without it
        // hipcc treats .cpp as host-only C++ and the __global__ kernels fail.
        "-x".into(),
        "hip".into(),
        wrapper_src_path.to_str().unwrap().into(),
        "-I".into(),
        ck_include.to_str().unwrap().into(),
        "-I".into(),
        include_dir.to_str().unwrap().into(),
        "-std=c++17".into(),
        format!("--offload-arch={arch}"),
        "-O3".into(),
        "-w".into(),
        "-fPIC".into(),
    ];
    let output = Command::new("hipcc")
        .args(&args)
        .output()
        .expect("Failed to run hipcc. Is the ROCm toolkit installed and on PATH?");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = std::fs::remove_file(&so_path);
        panic!(
            "FlashInfer JIT compilation failed (HEAD_DIM={}, arch={}):\nstdout: {}\nstderr: {}",
            head_dim, arch, stdout, stderr
        );
    }

    let elapsed = start.elapsed();
    eprintln!(
        "FlashInfer: compiled in {:.1}s → {}",
        elapsed.as_secs_f64(),
        so_path.display()
    );

    so_path
}

/// Returns ~/.cache/luminal/flashinfer/
fn cache_directory() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("luminal")
        .join("flashinfer")
}

/// Drop the embedded wrapper + kernel sources into the cache dir so hipcc has files
/// on disk to compile. Recreates the attention/ subdir so wrapper.cpp's relative
/// includes resolve. Returns (wrapper.cpp path, include root dir).
fn extract_wrapper_sources(cache_dir: &Path) -> (PathBuf, PathBuf) {
    let cpp = cache_dir.join("wrapper.cpp");
    write_if_changed(&cpp, WRAPPER_CPP.as_bytes());
    write_if_changed(&cache_dir.join("wrapper.hpp"), WRAPPER_HPP.as_bytes());

    let attn = cache_dir.join("attention");
    std::fs::create_dir_all(&attn).expect("Failed to create attention cache dir");
    write_if_changed(&attn.join("fmha_types.hpp"), ATTN_FMHA_TYPES.as_bytes());
    write_if_changed(&attn.join("helpers.hpp"), ATTN_HELPERS.as_bytes());
    write_if_changed(&attn.join("prefill.hpp"), ATTN_PREFILL.as_bytes());
    write_if_changed(&attn.join("decode.hpp"), ATTN_DECODE.as_bytes());

    (cpp, cache_dir.to_path_buf())
}

fn write_if_changed(path: &Path, contents: &[u8]) {
    if let Ok(existing) = std::fs::read(path)
        && existing == contents
    {
        return;
    }
    std::fs::write(path, contents).unwrap_or_else(|e| {
        panic!(
            "FlashInfer: failed to write wrapper source to {}: {e}",
            path.display()
        )
    });
}

fn wrapper_source_hash() -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for src in [
        WRAPPER_CPP,
        WRAPPER_HPP,
        ATTN_FMHA_TYPES,
        ATTN_HELPERS,
        ATTN_PREFILL,
        ATTN_DECODE,
    ] {
        src.hash(&mut hasher);
    }
    hasher.finish()
}

const CK_GIT_URL: &str = "https://github.com/ROCm/composable_kernel.git";
const CK_GIT_REV: &str = "d7609923b6a2fd9c83e8c40d8bd510b2c483ff91";

/// Locate Composable Kernel's `include/` directory — the one containing `ck/`
/// and `ck_tile/`. Unlike FlashInfer, CK is self-contained (no CUTLASS
/// submodule), so only a single include path is needed.
///
/// Resolution order: `LUMINAL_CK_DIR` override → system ROCm install
/// (`$ROCM_PATH` / `/opt/rocm`, which ships CK headers) → git clone of the
/// pinned commit into the cache.
fn locate_ck_includes() -> Option<PathBuf> {
    // 1. Explicit override: LUMINAL_CK_DIR points at a CK source/install root.
    if let Ok(path) = std::env::var("LUMINAL_CK_DIR")
        && !path.is_empty()
    {
        let inc = PathBuf::from(&path).join("include");
        if is_ck_include_dir(&inc) {
            return Some(inc);
        }
        eprintln!(
            "Composable Kernel: LUMINAL_CK_DIR={path} did not contain include/ck_tile — \
             falling back to default locations"
        );
    }

    // 2. System ROCm install ships CK headers under <rocm>/include.
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let rocm_inc = PathBuf::from(rocm).join("include");
    if is_ck_include_dir(&rocm_inc) {
        return Some(rocm_inc);
    }

    // 3. Last resort: fetch the pinned commit into the cache directory.
    fetch_ck_source().ok().map(|root| root.join("include"))
}

/// A directory is a usable CK include root if it holds the `ck_tile/` (or
/// legacy `ck/`) header tree.
fn is_ck_include_dir(inc: &Path) -> bool {
    inc.join("ck_tile").exists() || inc.join("ck").exists()
}

/// Clone Composable Kernel at `CK_GIT_REV` into
/// `~/.cache/luminal/flashinfer/ck-src/<short_rev>/` if absent, then return the
/// CK root directory (the one containing `include/`). CK is self-contained — no
/// CUTLASS submodule is required. One-time download; subsequent calls
/// short-circuit on the directory check.
fn fetch_ck_source() -> Result<PathBuf, String> {
    let short = &CK_GIT_REV[..12];
    let cache_root = cache_directory().join("ck-src").join(short);

    if is_ck_include_dir(&cache_root.join("include")) {
        return Ok(cache_root);
    }

    let parent = cache_root.parent().unwrap();
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;

    // Clone into a staging dir, then atomic rename. Protects against multiple
    // processes racing to fetch the same source.
    let staging = parent.join(format!(".staging-{}-{}", short, std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);

    eprintln!(
        "Composable Kernel: cloning {CK_GIT_URL} @ {short} into {} (one-time fetch) …",
        cache_root.display()
    );

    run_git(&[
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        CK_GIT_URL,
        staging.to_str().unwrap(),
    ])?;
    run_git_in(&staging, &["checkout", CK_GIT_REV])?;

    if !is_ck_include_dir(&staging.join("include")) {
        return Err(format!(
            "Composable Kernel clone succeeded but include/ck_tile missing at {}",
            staging.display()
        ));
    }

    // Atomic-ish rename. If another process beat us to it, just keep theirs.
    match std::fs::rename(&staging, &cache_root) {
        Ok(()) => {}
        Err(_) if cache_root.exists() => {
            let _ = std::fs::remove_dir_all(&staging);
        }
        Err(e) => return Err(format!("rename to {} failed: {e}", cache_root.display())),
    }

    Ok(cache_root)
}

fn run_git(args: &[&str]) -> Result<(), String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn `git`: {e}. Is git installed?"))?;
    if !out.status.success() {
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn run_git_in(cwd: &Path, args: &[&str]) -> Result<(), String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("failed to spawn `git`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`git {}` in {} failed: {}",
            args.join(" "),
            cwd.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

/// Detect ROCm arch via env overrides → amd-smi → default gfx950.
///
/// Order matters: if `HSA_OVERRIDE_GFX_VERSION` is set (common on consumer cards
/// whose physical ISA the runtime is told to masquerade as another — e.g. a
/// gfx1102 RX 7600 XT overridden to 11.0.0/gfx1100), the *runtime* will only load
/// code objects for the overridden arch. amd-smi reports the PHYSICAL arch
/// (gfx1102), so trusting it would build a code object the runtime can't load and
/// every kernel launch faults. The override therefore wins over amd-smi.
fn detect_rocm_arch() -> String {
    if let Ok(arch) = std::env::var("FLASHINFER_ROCM_ARCH") {
        return arch;
    }

    if let Ok(v) = std::env::var("HSA_OVERRIDE_GFX_VERSION")
        && let Some(arch) = arch_from_hsa_override(&v)
    {
        return arch;
    }

    if let Ok(output) = Command::new("amd-smi")
        .args(["static", "-g", "0", "--asic", "--json"])
        .output()
        && output.status.success()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(gfx) = parse_target_graphics_version(&stdout) {
            return gfx;
        }
    }

    "gfx950".to_string()
}

/// Map an `HSA_OVERRIDE_GFX_VERSION` value ("major.minor.stepping", e.g.
/// "11.0.0") to a gfx arch string ("gfx1100"). minor/stepping are single hex
/// digits (so "9.0.10" → "gfx90a", "9.4.2" → "gfx942").
fn arch_from_hsa_override(v: &str) -> Option<String> {
    let mut parts = v.trim().split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    let stepping: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || minor > 0xf || stepping > 0xf {
        return None;
    }
    Some(format!("gfx{major}{minor:x}{stepping:x}"))
}

/// Pull `target_graphics_version` (e.g. "gfx1102") out of the
/// `amd-smi static --asic --json` output, without a JSON dependency.
fn parse_target_graphics_version(json: &str) -> Option<String> {
    const KEY: &str = "\"target_graphics_version\"";
    let after_key = &json[json.find(KEY)? + KEY.len()..];
    let after_colon = &after_key[after_key.find(':')? + 1..];
    let after_open_quote = &after_colon[after_colon.find('"')? + 1..];
    let value = &after_open_quote[..after_open_quote.find('"')?];
    (!value.trim().is_empty()).then(|| value.trim().to_string())
}
