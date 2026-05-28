pub mod dyn_backend;
pub mod host;
pub mod kernel;
mod memory_analysis;
pub mod runtime;
use std::{
    path::Path,
    sync::Arc,
};

pub use rocmrc;

use rocmrc::{driver::HipStream};

#[cfg(test)]
mod tests;

use rocmrc::{
    HipResult,
    driver::{HipContext, sys as driver_sys},
    hiprtc::{
        Hsaco,
        result::{self as hiprtc_result, HiprtcError},
        sys as hiprtc_sys,
    },
    hipblaslt::{HipBlasLt}
};
use luminal::dtype::DType;

fn rocm_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F64 => "double",
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Bf16 => "__nv_bfloat16",
        DType::TF32 => "float", // TF32 uses float storage, tensor cores handle the format
        DType::Int => "int",
        DType::I16 => "short",
        DType::U16 => "unsigned short",
        DType::I8 => "signed char",
        DType::U8 => "unsigned char",
        DType::Bool => "unsigned char",
        DType::F8E4M3 => "__nv_fp8_e4m3",
        DType::F8E5M2 => "__nv_fp8_e5m2",
        DType::F8UE8M0 => "__nv_fp8_e8m0",
        DType::F6E2M3 => "__nv_fp6_e2m3",
        DType::F6E3M2 => "__nv_fp6_e3m2",
        DType::F4E2M1 => "__nv_fp4_e2m1",
        DType::I4 | DType::U4 => "unsigned char", // Sub-byte, packed storage
    }
}

const ROCM_HIPRTC_INCLUDE_PATHS: [&str; 2] = ["/opt/rocm/include", "/usr/include"];

#[derive(Debug)]
pub(crate) enum RocmModuleImageCompileFailure {
    Hiprtc {
        stage: &'static str,
        error: HiprtcError,
    },
    NoModuleImageProduced,
}

#[derive(Debug)]
pub(crate) struct RocmModuleImageCompileError {
    pub target_arch: Option<String>,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub hiprtc_options: Vec<String>,
    pub hiprtc_log: Option<String>,
    pub failure: RocmModuleImageCompileFailure,
}

impl std::fmt::Display for RocmModuleImageCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to compile ROCm module image")?;
        if let Some(target_arch) = &self.target_arch {
            write!(f, " for {target_arch}")?;
        }
        match &self.failure {
            RocmModuleImageCompileFailure::Hiprtc { stage, error } => {
                write!(f, ": HipRTC {stage} failed: {error}")?;
            }
            RocmModuleImageCompileFailure::NoModuleImageProduced => {
                write!(f, ": HipRTC produced no ROCBIN for the selected target")?;
            }
        }
        if let Some(version) = self.driver_version {
            write!(f, " | driver {}", format_rocm_version(version))?;
        }
        if let Some(version) = self.runtime_version {
            write!(f, " | runtime {}", format_rocm_version(version))?;
        }
        if !self.hiprtc_options.is_empty() {
            write!(f, " | options {:?}", self.hiprtc_options)?;
        }
        if let Some(log) = &self.hiprtc_log {
            write!(f, " | log: {log}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RocmModuleImageCompileError {}

fn format_rocm_version(version: i32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn rocm_hiprtc_include_paths() -> Vec<String> {
    let mut include_paths = Vec::new();
    for env_var in ["ROCM_HOME", "ROCM_PATH", "ROCM_ROOT"] {
        if let Ok(root) = std::env::var(env_var) {
            let path = format!("{root}/include");
            if Path::new(&path).exists() && !include_paths.contains(&path) {
                include_paths.push(path);
            }
        }
    }
    for path in ROCM_HIPRTC_INCLUDE_PATHS {
        let path = path.to_string();
        if Path::new(&path).exists() && !include_paths.contains(&path) {
            include_paths.push(path);
        }
    }
    include_paths
}

fn rocm_driver_diagnostics() -> (Option<i32>, Option<i32>) {
    let mut driver_version = 0;
    let driver_version = unsafe { driver_sys::hipDriverGetVersion(&mut driver_version as *mut _) }
        .result()
        .ok()
        .map(|_| driver_version);

    // Avoid touching the HIP runtime loader here. On some environments it
    // eagerly resolves newer libamdhip64 symbols that may not exist in the
    // installed runtime.
    (driver_version, None)
}

pub(crate) fn try_create_hipblaslt(
    stream: Arc<HipStream>,
) -> std::result::Result<Arc<HipBlasLt>, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| HipBlasLt::new(stream))) {
        Ok(Ok(handle)) => Ok(handle),
        Ok(Err(err)) => Err(err.to_string()),
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else if let Some(message) = payload.downcast_ref::<&str>() {
                message.to_string()
            } else {
                "rocBLASLt initialization panicked".to_string()
            };
            Err(message)
        }
    }
}

fn rocm_hiprtc_compile_options(target_arch: &str) -> Vec<String> {
    let mut options = rocm_hiprtc_include_paths()
        .into_iter()
        .map(|path| format!("--include-path={path}"))
        .collect::<Vec<_>>();
    options.push(format!("--offload-arch={target_arch}"));
    options
}

fn build_module_image_compile_error(
    target_arch: Option<String>,
    driver_version: Option<i32>,
    runtime_version: Option<i32>,
    hiprtc_options: &[String],
    hiprtc_log: Option<String>,
    failure: RocmModuleImageCompileFailure,
) -> RocmModuleImageCompileError {
    RocmModuleImageCompileError {
        target_arch,
        driver_version,
        runtime_version,
        hiprtc_options: hiprtc_options.to_vec(),
        hiprtc_log,
        failure,
    }
}

fn read_hiprtc_log(program: hiprtc_sys::hiprtcProgram) -> Option<String> {
    let raw = hiprtc_result::get_program_log(program).ok()?;
    let log = raw.trim_end_matches('\0').trim().to_string();
    if log.is_empty() { None } else { Some(log) }
}

#[allow(clippy::slow_vector_initialization)]
fn get_rocbin(program: hiprtc_sys::hiprtcProgram) -> Result<Vec<u8>, HiprtcError> {
    let mut rocbin_size = 0usize;
    unsafe { hiprtc_sys::hiprtcGetBitcodeSize(program, &mut rocbin_size as *mut _) }.result()?;
    if rocbin_size == 0 {
        return Ok(Vec::new());
    }

    let mut cubin = Vec::with_capacity(rocbin_size);
    cubin.resize(rocbin_size, 0u8);
    unsafe { hiprtc_sys::hiprtcGetBitcodeSize(program, cubin.as_mut_ptr() as *mut _) }.result()?;
    Ok(cubin)
}

pub(crate) fn compile_module_image_for_current_device<S: AsRef<str>>(
    ctx: &Arc<HipContext>,
    src: S,
) -> Result<Hsaco, RocmModuleImageCompileError> {
    let (driver_version, runtime_version) = rocm_driver_diagnostics();
    let target_arch = ctx.gfx_arch().to_string();
    let hiprtc_options = rocm_hiprtc_compile_options(&target_arch);   // Vec<String>

    let program = hiprtc_result::create_program(src.as_ref(), "kernel.hip").map_err(|error| {
        build_module_image_compile_error(
            Some(target_arch.clone()),
            driver_version,
            runtime_version,
            &hiprtc_options,                              // &Vec<String> → &[String], fine
            None,
            RocmModuleImageCompileFailure::Hiprtc { stage: "create_program", error },
        )
    })?;

    let opt_refs: Vec<&str> = hiprtc_options.iter().map(String::as_str).collect();
    if let Err(error) = hiprtc_result::compile_program(program, &opt_refs) {
        let hiprtc_log = read_hiprtc_log(program);
        let _ = hiprtc_result::destroy_program(program);
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &hiprtc_options,
            hiprtc_log,
            RocmModuleImageCompileFailure::Hiprtc { stage: "compile_program", error },
        ));
    }

    let hiprtc_log = read_hiprtc_log(program);
    let rocbin = match get_rocbin(program) {
        Ok(cubin) => cubin,
        Err(error) => {
            let _ = hiprtc_result::destroy_program(program);
            return Err(build_module_image_compile_error(
                Some(target_arch),
                driver_version,
                runtime_version,
                &hiprtc_options,
                hiprtc_log,
                RocmModuleImageCompileFailure::Hiprtc {
                    stage: "get_cubin",
                    error,
                },
            ));
        }
    };

    if let Err(error) = hiprtc_result::destroy_program(program) {
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &hiprtc_options,
            hiprtc_log,
            RocmModuleImageCompileFailure::Hiprtc {
                stage: "destroy_program",
                error,
            },
        ));
    }

    if rocbin.is_empty() {
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &hiprtc_options,
            hiprtc_log,
            RocmModuleImageCompileFailure::NoModuleImageProduced,
        ));
    }

    Ok(Hsaco::from_bytes(rocbin))
}

/// Returns the bandwidth of the device in GB/s. Unknown devices return `None`
/// so callers can skip bandwidth-dependent decisions instead of guessing.
pub fn rocm_bandwidth_gbps(ctx: &Arc<HipContext>) -> Option<usize> {
    let name = ctx.name().ok()?;
    Some(match name.as_str() {
        n if n.contains("7900 XTX") => 960,
        n if n.contains("7600 XT") => 288,
        n if n.contains("R9600D") => 640,
        _ => return None,
    })
}

/// Returns the f32 compute throughput of the device in TFLOPs. Same unknown-
/// device semantics as [`rocm_bandwidth_gbps`].
pub fn rocm_compute_f32_tflops(ctx: &Arc<HipContext>) -> Option<usize> {
    let name = ctx.name().ok()?;
    Some(match name.as_str() {
        n if n.contains("7900 XTX") => 61,
        n if n.contains("7600 XT") => 23,
        n if n.contains("R9600D") => 25,
        _ => return None,
    })
}
