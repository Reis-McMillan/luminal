use std::sync::{Arc, OnceLock};

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND, STRING},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
};

use crate::{
    rocmrc::{
        rocblas::{
            RocBlas,
            sys::{rocblas_operation, rocblas_set_stream, rocblas_sgemm, rocblas_status},
        },
        hip::HipStream,
    },
    host::{DeviceBuffer, HostOp},
};

/// Global shared rocBLAS handle to avoid per-operation workspace allocation
static SHARED_ROCBLAS: OnceLock<Arc<RocBlas>> = OnceLock::new();

/// Parse rocBLAS operation from egglog string (e.g., "\"T\"" -> rocblas_operation_transpose)
pub fn parse_rocblas_op(s: &str) -> rocblas_operation {
    // Strip quotes if present (egglog strings are stored with quotes)
    let stripped = s.trim_matches('"');
    match stripped {
        "T" => rocblas_operation::rocblas_operation_transpose,
        "N" => rocblas_operation::rocblas_operation_none,
        "C" => rocblas_operation::rocblas_operation_conjugate_transpose,
        other => panic!("Unknown rocBLAS operation: '{other}' (original: '{s}')"),
    }
}

#[allow(dead_code)]
pub struct RocBlasSgemm {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: rocblas_operation,
    b_layout: rocblas_operation,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    /// Lazily initialized rocBLAS handle - created on first execute
    rocblas: OnceLock<Arc<RocBlas>>,
}

// `RocBlas` does not implement `Debug`, so derive can't cover the handle field.
impl std::fmt::Debug for RocBlasSgemm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocBlasSgemm")
            .field("m", &self.m)
            .field("n", &self.n)
            .field("k", &self.k)
            .field("a_layout", &self.a_layout)
            .field("b_layout", &self.b_layout)
            .field("lda", &self.lda)
            .field("ldb", &self.ldb)
            .field("ldc", &self.ldc)
            .finish_non_exhaustive()
    }
}

// Useless default for IntoEgglogOp
impl Default for RocBlasSgemm {
    fn default() -> Self {
        Self {
            m: Expression::default(),
            n: Expression::default(),
            k: Expression::default(),
            a_layout: rocblas_operation::rocblas_operation_none, // IGNORE NOT REAL
            b_layout: rocblas_operation::rocblas_operation_transpose, // IGNORE NOT REAL
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            rocblas: OnceLock::new(),
        }
    }
}

impl EgglogOp for RocBlasSgemm {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "rocblasSgemm",
            &[
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("a_layout", STRING),
                ("b_layout", STRING),
                ("lda", EXPRESSION),
                ("ldb", EXPRESSION),
                ("ldc", EXPRESSION),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(include_str!["sgemm_RmRm_rewrite.egg"]), // row row
            Rule::raw(include_str!["sgemm_RmCm_rewrite.egg"]), // row col
            Rule::raw(include_str!["sgemm_CmRm_rewrite.egg"]), // col row
            Rule::raw(include_str!["sgemm_CmCm_rewrite.egg"]), // col col
        ]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        // Extract dimensions from egglog
        let m = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let n = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();

        // Extract layout strings from egglog
        let a_layout_str = &egraph.enodes[kind_children[3]].0;
        let b_layout_str = &egraph.enodes[kind_children[4]].0;
        let a_layout = parse_rocblas_op(a_layout_str);
        let b_layout = parse_rocblas_op(b_layout_str);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, kind_children[5], expr_cache).unwrap();
        let ldb = extract_expr(egraph, kind_children[6], expr_cache).unwrap();
        let ldc = extract_expr(egraph, kind_children[7], expr_cache).unwrap();

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            lda,
            ldb,
            ldc,
            rocblas: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for RocBlasSgemm {
    fn execute(
        &self,
        stream: &Arc<HipStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // GEMM parameters
        let m = self.m.exec(dyn_map).unwrap() as i32;
        let n = self.n.exec(dyn_map).unwrap() as i32;
        let k = self.k.exec(dyn_map).unwrap() as i32;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = self.lda.exec(dyn_map).unwrap() as i32;
        let ldb = self.ldb.exec(dyn_map).unwrap() as i32;
        let ldc = self.ldc.exec(dyn_map).unwrap() as i32;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        // Get buffers: output is self_node, inputs are from graph edges
        let c_buf = buffers[&self_node];
        let a_buf = buffers[&inputs[0]];
        let b_buf = buffers[&inputs[1]];

        // Get device pointers
        let a_ptr = a_buf.ptr();
        let b_ptr = b_buf.ptr();
        let c_ptr = c_buf.ptr();

        // Debug: Check buffer sizes
        trace!(
            "buffer_validation {}=={},{}=={},{}=={}",
            a_buf.len(),
            m * k * 4,
            b_buf.len(),
            k * n * 4,
            c_buf.len(),
            m * n * 4
        );
        let _sgemm_span = span!(
            Level::TRACE,
            "rocBLAS_SGEMM",
            m,
            n,
            k,
            alpha,
            beta,
            lda,
            ldb,
            ldc,
            ?a_layout,
            ?b_layout,
        )
        .entered();

        // Use shared rocBLAS handle to avoid per-operation workspace allocation.
        // `RocBlas::new` returns `Self`, so wrap it for the shared `Arc` slot.
        let rocblas =
            SHARED_ROCBLAS.get_or_init(|| Arc::new(RocBlas::new(stream.clone()).unwrap()));

        // Set the stream for this operation (rocBLAS handle can work with any stream).
        // The stream types from rocblas::sys and hip::sys are compatible, just cast.
        unsafe {
            rocblas_set_stream(*rocblas.handle(), stream.hip_stream() as _);
        }

        let status = unsafe {
            rocblas_sgemm(
                *rocblas.handle(),
                a_layout,
                b_layout,
                m,
                n,
                k,
                &alpha as *const f32,
                a_ptr as *const f32,
                lda,
                b_ptr as *const f32,
                ldb,
                &beta as *const f32,
                c_ptr as *mut f32,
                ldc,
            )
        };
        stream.synchronize().unwrap();

        if status != rocblas_status::rocblas_status_success {
            return Err(anyhow::anyhow!(
                "rocBLAS SGEMM TN failed with status: {:?}",
                status
            ));
        }

        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn output_bytes(&self) -> Expression {
        // RocBlasSgemm is F32 only (Sgemm = Single precision)
        self.output_size() * 4
    }
}
