//! [`DynBackend`] implementation for the ROCm lite runtime.

use luminal::dtype::DType;
use luminal::dyn_backend::{BackendCompileArgs, DynBackend, compile_backend};
use luminal::prelude::*;

use crate::rocmrc::hip::HipContext;
use crate::runtime::RocmRuntime;

/// [`DynBackend`] wrapper for [`RocmRuntime`].
pub struct RocmLiteDynBackend {
    pub runtime: RocmRuntime,
}

impl DynBackend for RocmLiteDynBackend {
    fn name(&self) -> &str {
        "rocm_lite"
    }
    fn device_type(&self) -> &str {
        "rocm"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, _dtype: DType) {
        self.runtime.set_data(node, bytes);
    }
    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }
    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        self.runtime.get_f32(node)
    }
    fn get_output_i32(&self, node: NodeIndex) -> Vec<i32> {
        self.runtime.get_i32(node)
    }
    fn get_output_bool(&self, node: NodeIndex) -> Vec<bool> {
        self.runtime.get_bool(node)
    }
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.runtime.execute(dyn_map);
    }

    fn supports_device_ptrs(&self) -> bool {
        true
    }
    unsafe fn set_device_ptr(&mut self, node: NodeIndex, ptr: u64, n: usize) {
        unsafe { self.runtime.set_device_ptr(node, ptr, n) }
    }
    unsafe fn set_output_device_ptr(&mut self, node: NodeIndex, ptr: u64, n: usize) {
        unsafe { self.runtime.set_output_device_ptr(node, ptr, n) }
    }
    fn output_is_zero_copy(&self, node: NodeIndex) -> bool {
        self.runtime.output_is_zero_copy(node)
    }
    unsafe fn copy_output_to_device_ptr(&self, node: NodeIndex, ptr: u64, n: usize) {
        unsafe { self.runtime.copy_output_to_device_ptr(node, ptr, n) }
    }
}

pub fn rocm_lite_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    let hip_ctx = HipContext::new(0).map_err(|e| format!("HIP init failed: {e}"))?;
    let stream = hip_ctx.default_stream();
    compile_backend::<RocmRuntime>(
        graph,
        args,
        || Ok(RocmRuntime::initialize(stream)),
        |rt, node, bytes, _dtype| {
            rt.set_data(node, bytes);
        },
        Some(&|rt, node, ptr, n| unsafe { rt.set_device_ptr(node, ptr, n) }),
        |rt| Box::new(RocmLiteDynBackend { runtime: rt }),
    )
}
