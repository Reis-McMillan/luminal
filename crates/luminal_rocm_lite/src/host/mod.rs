use std::{fmt::Debug, sync::Arc};

use crate::rocmrc::hip::{HipStream, HipError, result};
use luminal::{op::EgglogOp, prelude::*};
mod rocblas;
mod hipblaslt;
pub mod flashinfer;

pub type Ops = (
    hipblaslt::HipblasLt,
    hipblaslt::HipblasLtScaled,
    flashinfer::FlashInferAttention,
    // moe (cuda_lite sibling) not yet ported to ROCm.
);

#[cfg(test)]
pub(crate) type HipblasLtTypeTuple = (
    luminal::dtype::DType,
    luminal::dtype::DType,
    luminal::dtype::DType,
    luminal::dtype::DType,
    &'static str,
    luminal::dtype::DType,
);

#[cfg(test)]
pub(crate) fn hipblaslt_type_tuple(op: &dyn HostOp) -> Option<HipblasLtTypeTuple> {
    op.as_any()
        .downcast_ref::<hipblaslt::HipblasLt>()
        .map(hipblaslt::HipblasLt::type_tuple)
}

#[cfg(test)]
pub(crate) type HipblasLtScaleValues = (f64, f64);

#[cfg(test)]
pub(crate) fn hipblaslt_scale_values(op: &dyn HostOp) -> Option<HipblasLtScaleValues> {
    op.as_any()
        .downcast_ref::<hipblaslt::HipblasLt>()
        .map(hipblaslt::HipblasLt::scale_values)
}

#[cfg(test)]
pub(crate) fn hipblaslt_epilogue(op: &dyn HostOp) -> Option<&'static str> {
    op.as_any()
        .downcast_ref::<hipblaslt::HipblasLt>()
        .map(hipblaslt::HipblasLt::epilogue)
}

#[cfg(test)]
pub(crate) type HipblasLtMatrixOrders = (&'static str, &'static str, &'static str, &'static str);

#[cfg(test)]
pub(crate) fn hipblaslt_matrix_orders(op: &dyn HostOp) -> Option<HipblasLtMatrixOrders> {
    op.as_any()
        .downcast_ref::<hipblaslt::HipblasLt>()
        .map(hipblaslt::HipblasLt::matrix_orders)
}

#[cfg(test)]
pub(crate) type HipblasLtTransposeOps = (&'static str, &'static str);

#[cfg(test)]
pub(crate) fn hipblaslt_transpose_ops(op: &dyn HostOp) -> Option<HipblasLtTransposeOps> {
    op.as_any()
        .downcast_ref::<hipblaslt::HipblasLt>()
        .map(hipblaslt::HipblasLt::transpose_ops)
}

#[cfg(test)]
pub(crate) fn hipblaslt_c_d_layouts_match(op: &dyn HostOp) -> Option<bool> {
    op.as_any()
        .downcast_ref::<hipblaslt::HipblasLt>()
        .map(hipblaslt::HipblasLt::c_d_layouts_match)
}

#[cfg(test)]
pub(crate) type HipblasLtTensorScaleInputs = (bool, bool);

#[cfg(test)]
pub(crate) fn hipblaslt_tensor_scale_inputs(op: &dyn HostOp) -> Option<HipblasLtTensorScaleInputs> {
    op.as_any()
        .downcast_ref::<hipblaslt::HipblasLt>()
        .map(hipblaslt::HipblasLt::tensor_scale_inputs)
}

/// Non-owning device buffer handle used by host operations.
///
/// Runtime-owned intermediates may be a whole `HipSlice`, a subregion inside
/// the reusable arena, or an external pointer. Host ops only need the pointer
/// and the logical byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    ptr: u64,
    len: usize,
}

impl DeviceBuffer {
    pub fn new(ptr: u64, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn ptr(self) -> u64 {
        self.ptr
    }

    pub fn len(self) -> usize {
        self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn clone_dtoh(self, stream: &Arc<HipStream>) -> Result<Vec<u8>, HipError> {
        let mut host = vec![0u8; self.len];
        unsafe {
            result::memcpy_dtoh_async(&mut host, self.ptr, stream.hip_stream())?;
        }
        stream.synchronize()?;
        Ok(host)
    }
}

/// Host operations that execute on the CPU but orchestrate GPU work.
///
/// This includes operations like cuBLAS calls and HIP graph executions.
pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    /// Execute the operation with access to buffers via a map.
    ///
    /// # Arguments
    /// * `stream` - The HIP stream to execute on
    /// * `self_node` - The NodeIndex of this op in the llir_graph (used as output buffer)
    /// * `inputs` - NodeIndices of input nodes (in edge order from the graph)
    /// * `buffers` - Map from NodeIndex to device buffer for all allocated nodes
    /// * `dyn_map` - Dynamic dimension values
    fn execute(
        &self,
        stream: &Arc<HipStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()>;

    /// Returns the output buffer size in elements.
    /// Return 0 if this op doesn't have a single output buffer (e.g., RocmGraphOp).
    fn output_size(&self) -> Expression;

    /// Returns the output buffer size in bytes (accounts for dtype).
    fn output_bytes(&self) -> Expression;

    /// Returns additional nodes (beyond graph edges) that this op needs buffers for.
    ///
    /// For most ops, this returns empty (buffers determined by graph edges).
    /// For RocmGraphOp, this returns all internal kernel nodes.
    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns relative lifetimes for extra buffer nodes within this host op.
    ///
    /// The tuple is `(node, first_step, last_step)`, where steps are local to
    /// this host op's execution. Returning `None` tells the runtime to treat
    /// every extra buffer as live for the whole host op.
    fn extra_buffer_lifetimes(&self) -> Option<Vec<(NodeIndex, usize, usize)>> {
        None
    }

    /// Returns buffer size requirements for extra nodes (node -> size in elements).
    ///
    /// Called during buffer allocation to ensure all required buffers exist.
    /// For RocmGraphOp, this returns sizes for all internal kernel output buffers.
    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        FxHashMap::default()
    }

    /// Returns the name of this host op for stats reporting, or None if not reportable.
    fn stats_name(&self) -> Option<&'static str> {
        None
    }

    /// Estimated floating-point operations this op issues to the GPU for the
    /// given dynamic-dim binding. Default 0 for ops that do no arithmetic
    /// (copies, reshapes). Matmul/kernel ops override with their analytical
    /// cost (e.g. `2·m·n·k`).
    fn flop_estimate(&self, _dyn_map: &FxHashMap<char, usize>) -> u64 {
        0
    }

    /// Number of distinct GPU kernels this op launches. Default 1 (a single
    /// BLAS/attention dispatch). `RocmGraphOp` overrides with the number of
    /// fused kernels in its HIP graph. Note: a BLAS call may issue more than
    /// one GPU kernel internally, which is invisible here — so this is a lower
    /// bound for BLAS-heavy graphs.
    fn kernel_launch_count(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        1
    }
}
