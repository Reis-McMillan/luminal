use std::sync::{Arc, OnceLock};

use half::{bf16, f16};
use luminal::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, F64, OP_KIND, STRING},
        extract_dtype, extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
};

use crate::{
    rocmrc::{
        HipResult,
        hipblaslt::{
            HipBlasLt,
            sys::{
                hipDataType, hipblasComputeType_t, hipblasLtEpilogue_t, hipblasLtMatmul,
                hipblasLtMatmulAlgoGetHeuristic, hipblasLtMatmulDesc_t,
                hipblasLtMatmulDescAttributes_t, hipblasLtMatmulDescCreate,
                hipblasLtMatmulDescDestroy, hipblasLtMatmulDescSetAttribute,
                hipblasLtMatmulHeuristicResult_t, hipblasLtMatmulPreference_t,
                hipblasLtMatmulPreferenceAttributes_t, hipblasLtMatmulPreferenceCreate,
                hipblasLtMatmulPreferenceDestroy, hipblasLtMatmulPreferenceSetAttribute,
                hipblasLtMatrixLayout_t, hipblasLtMatrixLayoutAttribute_t,
                hipblasLtMatrixLayoutCreate, hipblasLtMatrixLayoutDestroy,
                hipblasLtMatrixLayoutSetAttribute, hipblasLtOrder_t, hipblasOperation_t,
            },
        },
        driver::HipStream,
    },
    host::{DeviceBuffer, HostOp},
    try_create_hipblaslt,
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct HipblasLt {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: hipblasOperation_t,
    b_layout: hipblasOperation_t,
    a_order: hipblasLtOrder_t,
    b_order: hipblasLtOrder_t,
    c_order: hipblasLtOrder_t,
    d_order: hipblasLtOrder_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    ldd: Expression,
    batch_count: Expression,
    stride_a: Expression,
    stride_b: Expression,
    stride_c: Expression,
    stride_d: Expression,
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    d_dtype: DType,
    compute_type: hipblasComputeType_t,
    scale_dtype: DType,
    alpha: f64,
    beta: f64,
    epilogue: hipblasLtEpilogue_t,
    a_scale_input: bool,
    b_scale_input: bool,
    hipblas_lt: OnceLock<Arc<HipBlasLt>>,
}

// Useless default for IntoEgglogOp
impl Default for HipblasLt {
    fn default() -> Self {
        Self {
            m: Expression::default(),
            n: Expression::default(),
            k: Expression::default(),
            a_layout: hipblasOperation_t::HIPBLAS_OP_N,
            b_layout: hipblasOperation_t::HIPBLAS_OP_T,
            a_order: hipblasLtOrder_t::HIPBLASLT_ORDER_COL,
            b_order: hipblasLtOrder_t::HIPBLASLT_ORDER_COL,
            c_order: hipblasLtOrder_t::HIPBLASLT_ORDER_COL,
            d_order: hipblasLtOrder_t::HIPBLASLT_ORDER_COL,
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            ldd: Expression::default(),
            batch_count: 1.into(),
            stride_a: 0.into(),
            stride_b: 0.into(),
            stride_c: 0.into(),
            stride_d: 0.into(),
            a_dtype: DType::F32,
            b_dtype: DType::F32,
            c_dtype: DType::F32,
            d_dtype: DType::F32,
            compute_type: hipblasComputeType_t::HIPBLAS_COMPUTE_32F,
            scale_dtype: DType::F32,
            alpha: 1.0,
            beta: 0.0,
            epilogue: hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT,
            a_scale_input: false,
            b_scale_input: false,
            hipblas_lt: OnceLock::new(),
        }
    }
}

#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct HipblasLtScaled;

fn hipblaslt_sort(name: &'static str) -> SortDef {
    sort(
        OP_KIND,
        name,
        &[
            ("m", EXPRESSION),
            ("n", EXPRESSION),
            ("k", EXPRESSION),
            ("a_layout", STRING),
            ("b_layout", STRING),
            ("a_order", STRING),
            ("b_order", STRING),
            ("c_order", STRING),
            ("d_order", STRING),
            ("lda", EXPRESSION),
            ("ldb", EXPRESSION),
            ("ldc", EXPRESSION),
            ("ldd", EXPRESSION),
            ("batch_count", EXPRESSION),
            ("stride_a", EXPRESSION),
            ("stride_b", EXPRESSION),
            ("stride_c", EXPRESSION),
            ("stride_d", EXPRESSION),
            ("a_dtype", DTYPE),
            ("b_dtype", DTYPE),
            ("c_dtype", DTYPE),
            ("d_dtype", DTYPE),
            ("compute_type", STRING),
            ("scale_dtype", STRING),
            ("alpha", F64),
            ("beta", F64),
            ("epilogue", STRING),
        ],
    )
}

impl EgglogOp for HipblasLt {
    fn sort(&self) -> SortDef {
        hipblaslt_sort("hipblaslt")
    }

    fn n_inputs(&self) -> usize {
        let c_input = usize::from(self.beta != 0.0);
        let bias_input = usize::from(epilogue_uses_bias(self.epilogue));
        let scale_inputs = usize::from(self.a_scale_input) + usize::from(self.b_scale_input);
        2 + c_input + bias_input + scale_inputs
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(relation hipblasLt_base_dtype (DType))
                 (hipblasLt_base_dtype (F32))
                 (hipblasLt_base_dtype (F16))
                 (hipblasLt_base_dtype (Bf16))
                 (hipblasLt_base_dtype (TF32))
                 (relation hipblasLt_fp8_dtype (DType))
                 (hipblasLt_fp8_dtype (F8E4M3))
                 (hipblasLt_fp8_dtype (F8E5M2))
                 (relation hipblasLt_fp8_f32_output_pair (DType DType))
                 (hipblasLt_fp8_f32_output_pair (F8E4M3) (F8E4M3))
                 (hipblasLt_fp8_f32_output_pair (F8E4M3) (F8E5M2))
                 (hipblasLt_fp8_f32_output_pair (F8E5M2) (F8E4M3))",
            ),
            Rule::raw(include_str!["hipblaslt_RmRm_rewrite.egg"]), // row row
            Rule::raw(include_str!["hipblaslt_RmCm_rewrite.egg"]), // row col
            Rule::raw(include_str!["hipblaslt_CmRm_rewrite.egg"]), // col row
            Rule::raw(include_str!["hipblaslt_CmCm_rewrite.egg"]), // col col
            Rule::raw(include_str!["hipblaslt_fp8_rewrite.egg"]),
            Rule::raw(include_str!["hipblaslt_mixed_dtype_rewrite.egg"]),
            Rule::raw(include_str!["hipblaslt_scale_rewrite.egg"]),
            Rule::raw(include_str!["hipblaslt_beta_rewrite.egg"]),
            Rule::raw(include_str!["hipblaslt_epilogue_rewrite.egg"]),
            Rule::raw(include_str!["hipblaslt_row_order_rewrite.egg"]),
            // Delete the matmul-broadcast Mul eclass when the consuming Sum
            // eclass has a `hipblasLt` or `KernelBatchMatMul` alternative. The
            // hipblasLt / batched-matmul rewrite rules only union those enodes
            // into the Sum eclass after the broadcast pattern check passes,
            // so their presence is the matmul-broadcast signal — no further
            // stride-form check needed.
            //
            // Delete the HLIR `Mul` fallback from the Mul eclass. Emptying that
            // eclass lets the empty-eclass cascade prune the downstream Sum /
            // KernelSum fallback. cuBLAS, TileMatmulFullSplit, KernelBatchMatVec,
            // and KernelBatchMatMul all take original (a, b) inputs rather than
            // the Mul eclass, so they survive the cascade and remain as the
            // matmul output alternative.
            Rule::raw("(rule
                ((= ?mul (Op (Mul ?shape ?as ?bs ?os) ?inputs))
                 (= ?sum (Op (Sum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                 (= ?sum (Op (hipblasLt ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?ci)))
                ((delete (Op (Mul ?shape ?as ?bs ?os) ?inputs)))
                :ruleset cleanup
            )"),
            Rule::raw("(rule
                ((= ?mul (Op (Mul ?shape ?as ?bs ?os) ?inputs))
                 (= ?sum (Op (Sum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                 (= ?sum (Op (KernelBatchMatMul ?bos ?bk ?bas ?baks ?bbs ?bbks ?bouts ?bdt) ?bi)))
                ((delete (Op (Mul ?shape ?as ?bs ?os) ?inputs)))
                :ruleset cleanup
            )"),
            // Also remove any generic fusion wrapper that was unioned with the
            // broadcast Mul. This is deliberately a separate rule: requiring a
            // FusionEnd in the same eclass made cleanup miss valid hipblasLt
            // matmuls when fusion wrapping was absent.
            Rule::raw("(rule
                ((= ?mul (Op (FusionEnd ?fshape ?fos ?fdt) ?finputs))
                 (= ?sum (Op (Sum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                 (= ?sum (Op (hipblasLt ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?ci)))
                ((delete (Op (FusionEnd ?fshape ?fos ?fdt) ?finputs)))
                :ruleset cleanup
            )"),
            Rule::raw("(rule
                ((= ?mul (Op (FusionEnd ?fshape ?fos ?fdt) ?finputs))
                 (= ?sum (Op (Sum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                 (= ?sum (Op (KernelBatchMatMul ?bos ?bk ?bas ?baks ?bbs ?bbks ?bouts ?bdt) ?bi)))
                ((delete (Op (FusionEnd ?fshape ?fos ?fdt) ?finputs)))
                :ruleset cleanup
            )"),
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

        // Extract transpose/layout strings from egglog
        let a_layout_str = &egraph.enodes[kind_children[3]].0;
        let b_layout_str = &egraph.enodes[kind_children[4]].0;
        let a_layout = parse_hipblaslt_op(a_layout_str);
        let b_layout = parse_hipblaslt_op(b_layout_str);
        let a_order = parse_hipblaslt_order(&egraph.enodes[kind_children[5]].0);
        let b_order = parse_hipblaslt_order(&egraph.enodes[kind_children[6]].0);
        let c_order = parse_hipblaslt_order(&egraph.enodes[kind_children[7]].0);
        let d_order = parse_hipblaslt_order(&egraph.enodes[kind_children[8]].0);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, kind_children[9], expr_cache).unwrap();
        let ldb = extract_expr(egraph, kind_children[10], expr_cache).unwrap();
        let ldc = extract_expr(egraph, kind_children[11], expr_cache).unwrap();
        let ldd = extract_expr(egraph, kind_children[12], expr_cache).unwrap();

        // Extract batch parameters
        let batch_count = extract_expr(egraph, kind_children[13], expr_cache).unwrap();
        let stride_a = extract_expr(egraph, kind_children[14], expr_cache).unwrap();
        let stride_b = extract_expr(egraph, kind_children[15], expr_cache).unwrap();
        let stride_c = extract_expr(egraph, kind_children[16], expr_cache).unwrap();
        let stride_d = extract_expr(egraph, kind_children[17], expr_cache).unwrap();

        // Extract hipblasLt type tuple from egglog. Existing rewrites emit the
        // same dtype for A/B/C/D, but keeping these fields separate lets later
        // rewrites model mixed-input and mixed-output matmuls without changing
        // the host launch helper again.
        let a_dtype = extract_dtype(egraph, kind_children[18]);
        let b_dtype = extract_dtype(egraph, kind_children[19]);
        let c_dtype = extract_dtype(egraph, kind_children[20]);
        let d_dtype = extract_dtype(egraph, kind_children[21]);
        let compute_type_str = &egraph.enodes[kind_children[22]].0;
        let scale_dtype_str = &egraph.enodes[kind_children[23]].0;
        let compute_type = parse_hipblaslt_compute_type(compute_type_str, a_dtype);
        let scale_dtype = parse_hipblaslt_scale_dtype(scale_dtype_str, a_dtype);
        let alpha = parse_hipblaslt_scalar(&egraph.enodes[kind_children[24]].0);
        let beta = parse_hipblaslt_scalar(&egraph.enodes[kind_children[25]].0);
        let epilogue = parse_hipblaslt_epilogue(&egraph.enodes[kind_children[26]].0);

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            a_order,
            b_order,
            c_order,
            d_order,
            lda,
            ldb,
            ldc,
            ldd,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            stride_d,
            a_dtype,
            b_dtype,
            c_dtype,
            d_dtype,
            compute_type,
            scale_dtype,
            alpha,
            beta,
            epilogue,
            a_scale_input: false,
            b_scale_input: false,
            hipblas_lt: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl EgglogOp for HipblasLtScaled {
    fn sort(&self) -> SortDef {
        hipblaslt_sort("hipblaslt_scaled")
    }

    fn n_inputs(&self) -> usize {
        4
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
        let m = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let n = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();

        let a_layout = parse_hipblaslt_op(&egraph.enodes[kind_children[3]].0);
        let b_layout = parse_hipblaslt_op(&egraph.enodes[kind_children[4]].0);
        let a_order = parse_hipblaslt_order(&egraph.enodes[kind_children[5]].0);
        let b_order = parse_hipblaslt_order(&egraph.enodes[kind_children[6]].0);
        let c_order = parse_hipblaslt_order(&egraph.enodes[kind_children[7]].0);
        let d_order = parse_hipblaslt_order(&egraph.enodes[kind_children[8]].0);

        let lda = extract_expr(egraph, kind_children[9], expr_cache).unwrap();
        let ldb = extract_expr(egraph, kind_children[10], expr_cache).unwrap();
        let ldc = extract_expr(egraph, kind_children[11], expr_cache).unwrap();
        let ldd = extract_expr(egraph, kind_children[12], expr_cache).unwrap();

        let batch_count = extract_expr(egraph, kind_children[13], expr_cache).unwrap();
        let stride_a = extract_expr(egraph, kind_children[14], expr_cache).unwrap();
        let stride_b = extract_expr(egraph, kind_children[15], expr_cache).unwrap();
        let stride_c = extract_expr(egraph, kind_children[16], expr_cache).unwrap();
        let stride_d = extract_expr(egraph, kind_children[17], expr_cache).unwrap();

        let a_dtype = extract_dtype(egraph, kind_children[18]);
        let b_dtype = extract_dtype(egraph, kind_children[19]);
        let c_dtype = extract_dtype(egraph, kind_children[20]);
        let d_dtype = extract_dtype(egraph, kind_children[21]);
        let compute_type_str = &egraph.enodes[kind_children[22]].0;
        let scale_dtype_str = &egraph.enodes[kind_children[23]].0;
        let compute_type = parse_hipblaslt_compute_type(compute_type_str, a_dtype);
        let scale_dtype = parse_hipblaslt_scale_dtype(scale_dtype_str, a_dtype);
        let alpha = parse_hipblaslt_scalar(&egraph.enodes[kind_children[24]].0);
        let beta = parse_hipblaslt_scalar(&egraph.enodes[kind_children[25]].0);
        let epilogue = parse_hipblaslt_epilogue(&egraph.enodes[kind_children[26]].0);

        let extracted_state = HipblasLt {
            m,
            n,
            k,
            a_layout,
            b_layout,
            a_order,
            b_order,
            c_order,
            d_order,
            lda,
            ldb,
            ldc,
            ldd,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            stride_d,
            a_dtype,
            b_dtype,
            c_dtype,
            d_dtype,
            compute_type,
            scale_dtype,
            alpha,
            beta,
            epilogue,
            a_scale_input: true,
            b_scale_input: true,
            hipblas_lt: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

/// Convert DType to CUDA matrix/scale type for cuBLAS LT.
fn dtype_to_rocm_dtype(dtype: DType) -> hipDataType {
    match dtype {
        DType::F64 => hipDataType::HIP_R_64F,
        DType::F32 => hipDataType::HIP_R_32F,
        DType::F16 => hipDataType::HIP_R_16F,
        DType::Bf16 => hipDataType::HIP_R_16BF,
        // TF32 is a compute mode over f32 storage.
        DType::TF32 => hipDataType::HIP_R_32F,
        DType::F8E4M3 => hipDataType::HIP_R_8F_E4M3,
        DType::F8E5M2 => hipDataType::HIP_R_8F_E5M2,
        DType::Int => panic!("cuBLAS LT does not support integer matmul"),
        DType::Bool => panic!("cuBLAS LT does not support bool matmul"),
        other => todo!("cuBLAS LT matmul not yet implemented for {other}"),
    }
}

fn default_compute_and_scale_dtype(dtype: DType) -> (hipblasComputeType_t, DType) {
    match dtype {
        DType::F64 => (hipblasComputeType_t::HIPBLAS_COMPUTE_64F, DType::F64),
        DType::F32 => (hipblasComputeType_t::HIPBLAS_COMPUTE_32F, DType::F32),
        DType::F16 => (hipblasComputeType_t::HIPBLAS_COMPUTE_32F, DType::F32),
        DType::Bf16 => (
            hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_16BF,
            DType::F32,
        ),
        DType::TF32 => (
            hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_TF32,
            DType::F32,
        ),
        DType::F8E4M3 | DType::F8E5M2 => (hipblasComputeType_t::HIPBLAS_COMPUTE_32F, DType::F32),
        DType::Int => panic!("cuBLAS LT does not support integer matmul"),
        DType::Bool => panic!("cuBLAS LT does not support bool matmul"),
        other => todo!("cuBLAS LT matmul not yet implemented for {other}"),
    }
}

fn parse_hipblaslt_compute_type(s: &str, default_dtype: DType) -> hipblasComputeType_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "default" => default_compute_and_scale_dtype(default_dtype).0,
        "64F" => hipblasComputeType_t::HIPBLAS_COMPUTE_64F,
        "32F" => hipblasComputeType_t::HIPBLAS_COMPUTE_32F,
        "32F_FAST_16BF" => hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_16BF,
        "32F_FAST_TF32" => hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_TF32,
        other => panic!("Unknown hipblasLt compute type: '{other}' (original: '{s}')"),
    }
}

fn parse_hipblaslt_scale_dtype(s: &str, default_dtype: DType) -> DType {
    let stripped = s.trim_matches('"');
    match stripped {
        "default" => default_compute_and_scale_dtype(default_dtype).1,
        "F64" => DType::F64,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "Bf16" => DType::Bf16,
        "TF32" => DType::TF32,
        "F8E4M3" => DType::F8E4M3,
        "F8E5M2" => DType::F8E5M2,
        other => panic!("Unknown hipblasLt scale dtype: '{other}' (original: '{s}')"),
    }
}

fn parse_hipblaslt_scalar(s: &str) -> f64 {
    let stripped = s.trim_matches('"');
    stripped.parse::<f64>().unwrap_or_else(|_| {
        panic!("Unknown hipblasLt scalar literal: '{stripped}' (original: '{s}')")
    })
}

fn parse_hipblaslt_order(s: &str) -> hipblasLtOrder_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "COL" => hipblasLtOrder_t::HIPBLASLT_ORDER_COL,
        "ROW" => hipblasLtOrder_t::HIPBLASLT_ORDER_ROW,
        // COL32 / COL4_4R2_8C / COL32_2R_4R4 are cuBLASLt-only Tensor Core
        // layouts with no hipBLASLt equivalent. Don't accept them.
        other => panic!("Unknown hipblasLt matrix order: '{other}' (original: '{s}')"),
    }
}

/// Local hipBLASLt operation parser. `parse_hipblaslt_op` returns the rocBLAS
/// `rocblas_operation` enum, which is a distinct Rust type from hipBLASLt's
/// `hipblasOperation_t` despite identical N/T/C semantics.
fn parse_hipblaslt_op(s: &str) -> hipblasOperation_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "T" => hipblasOperation_t::HIPBLAS_OP_T,
        "N" => hipblasOperation_t::HIPBLAS_OP_N,
        "C" => hipblasOperation_t::HIPBLAS_OP_C,
        other => panic!("Unknown hipBLAS operation: '{other}' (original: '{s}')"),
    }
}

fn parse_hipblaslt_epilogue(s: &str) -> hipblasLtEpilogue_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "DEFAULT" => hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT,
        "BIAS" => hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_BIAS,
        "RELU" => hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_RELU,
        "RELU_BIAS" => hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_RELU_BIAS,
        "GELU" => hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_GELU,
        "GELU_BIAS" => hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_GELU_BIAS,
        other => panic!("Unknown hipblasLt epilogue: '{other}' (original: '{s}')"),
    }
}

#[cfg(test)]
fn compute_type_name(compute_type: hipblasComputeType_t) -> &'static str {
    match compute_type {
        hipblasComputeType_t::HIPBLAS_COMPUTE_64F => "64F",
        hipblasComputeType_t::HIPBLAS_COMPUTE_32F => "32F",
        hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_16BF => "32F_FAST_16BF",
        hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_TF32 => "32F_FAST_TF32",
        _ => "other",
    }
}

#[cfg(test)]
fn order_name(order: hipblasLtOrder_t) -> &'static str {
    match order {
        hipblasLtOrder_t::HIPBLASLT_ORDER_COL => "COL",
        hipblasLtOrder_t::HIPBLASLT_ORDER_ROW => "ROW",
        // rocm-07021 adds COL16_4R16/8/4/2; we never produce those in
        // parse_hipblaslt_order, but the match must be exhaustive.
        _ => "other",
    }
}

#[cfg(test)]
fn transpose_op_name(op: hipblasOperation_t) -> &'static str {
    match op {
        hipblasOperation_t::HIPBLAS_OP_N => "N",
        hipblasOperation_t::HIPBLAS_OP_T => "T",
        hipblasOperation_t::HIPBLAS_OP_C => "C",
    }
}

#[cfg(test)]
fn epilogue_name(epilogue: hipblasLtEpilogue_t) -> &'static str {
    match epilogue {
        hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT => "DEFAULT",
        hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_BIAS => "BIAS",
        hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_RELU => "RELU",
        hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_RELU_BIAS => "RELU_BIAS",
        hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_GELU => "GELU",
        hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_GELU_BIAS => "GELU_BIAS",
        _ => "other",
    }
}

#[derive(Debug, Clone, Copy)]
enum LtScalar {
    F64(f64),
    F32(f32),
    F16(f16),
    Bf16(bf16),
}

impl LtScalar {
    #[cfg(test)]
    fn one(dtype: DType) -> anyhow::Result<Self> {
        Self::from_f64(dtype, 1.0)
    }

    #[cfg(test)]
    fn zero(dtype: DType) -> anyhow::Result<Self> {
        Self::from_f64(dtype, 0.0)
    }

    fn from_f64(dtype: DType, value: f64) -> anyhow::Result<Self> {
        match dtype {
            DType::F64 => Ok(Self::F64(value)),
            DType::F32 => Ok(Self::F32(value as f32)),
            DType::F16 => Ok(Self::F16(f16::from_f32(value as f32))),
            DType::Bf16 => Ok(Self::Bf16(bf16::from_f32(value as f32))),
            other => Err(anyhow::anyhow!(
                "hipblasLt scale dtype {other} is not supported for host alpha/beta scalars"
            )),
        }
    }

    fn as_ptr(&self) -> *const std::ffi::c_void {
        match self {
            Self::F64(value) => value as *const _ as *const std::ffi::c_void,
            Self::F32(value) => value as *const _ as *const std::ffi::c_void,
            Self::F16(value) => value as *const _ as *const std::ffi::c_void,
            Self::Bf16(value) => value as *const _ as *const std::ffi::c_void,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LtMatmulProblem {
    m: u64,
    n: u64,
    k: u64,
    batch_count: i32,
}

#[derive(Debug, Clone, Copy)]
struct LtMatrixSpec {
    dtype: hipDataType,
    rows: u64,
    cols: u64,
    ld: i64,
    batch_stride: i64,
    order: hipblasLtOrder_t,
}

#[derive(Debug, Clone, Copy)]
struct LtComputeSpec {
    compute_type: hipblasComputeType_t,
    scale_dtype: hipDataType,
    alpha: LtScalar,
    beta: LtScalar,
    epilogue: hipblasLtEpilogue_t,
}

#[derive(Debug, Clone, Copy)]
struct LtMatmulSpec {
    problem: LtMatmulProblem,
    trans_a: hipblasOperation_t,
    trans_b: hipblasOperation_t,
    a: LtMatrixSpec,
    b: LtMatrixSpec,
    c: LtMatrixSpec,
    d: LtMatrixSpec,
    compute: LtComputeSpec,
    workspace_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct LtMatmulPointers {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    bias: Option<u64>,
    a_scale: Option<u64>,
    b_scale: Option<u64>,
}

struct LtRawDescriptors {
    matmul_desc: hipblasLtMatmulDesc_t,
    a_desc: hipblasLtMatrixLayout_t,
    b_desc: hipblasLtMatrixLayout_t,
    c_desc: hipblasLtMatrixLayout_t,
    d_desc: hipblasLtMatrixLayout_t,
    preference: hipblasLtMatmulPreference_t,
}

impl Default for LtRawDescriptors {
    fn default() -> Self {
        Self {
            matmul_desc: std::ptr::null_mut(),
            a_desc: std::ptr::null_mut(),
            b_desc: std::ptr::null_mut(),
            c_desc: std::ptr::null_mut(),
            d_desc: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
        }
    }
}

impl Drop for LtRawDescriptors {
    fn drop(&mut self) {
        unsafe {
            if !self.preference.is_null() {
                let _ = hipblasLtMatmulPreferenceDestroy(self.preference);
            }
            if !self.d_desc.is_null() {
                let _ = hipblasLtMatrixLayoutDestroy(self.d_desc);
            }
            if !self.c_desc.is_null() {
                let _ = hipblasLtMatrixLayoutDestroy(self.c_desc);
            }
            if !self.b_desc.is_null() {
                let _ = hipblasLtMatrixLayoutDestroy(self.b_desc);
            }
            if !self.a_desc.is_null() {
                let _ = hipblasLtMatrixLayoutDestroy(self.a_desc);
            }
            if !self.matmul_desc.is_null() {
                let _ = hipblasLtMatmulDescDestroy(self.matmul_desc);
            }
        }
    }
}

fn create_matrix_layout(
    desc: &mut hipblasLtMatrixLayout_t,
    spec: LtMatrixSpec,
) -> anyhow::Result<()> {
    unsafe {
        hipblasLtMatrixLayoutCreate(desc, spec.dtype, spec.rows, spec.cols, spec.ld).result()?;
        hipblasLtMatrixLayoutSetAttribute(
            *desc,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
            &spec.order as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<hipblasLtOrder_t>(),
        )
        .result()?;
    }
    Ok(())
}

fn clamp_ld_for_order(ld: i64, rows: u64, cols: u64, order: hipblasLtOrder_t) -> i64 {
    let min_ld = match order {
        hipblasLtOrder_t::HIPBLASLT_ORDER_COL => rows,
        hipblasLtOrder_t::HIPBLASLT_ORDER_ROW => cols,
        // rocm-07021 adds COL16_4R16/8/4/2 (Tensor Core packed layouts);
        // we don't produce them but the match must be exhaustive.
        _ => rows,
    };
    std::cmp::max(ld, min_ld as i64)
}

fn set_strided_batch_layout(
    desc: hipblasLtMatrixLayout_t,
    batch_count: i32,
    batch_stride: i64,
) -> anyhow::Result<()> {
    unsafe {
        hipblasLtMatrixLayoutSetAttribute(
            desc,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<i32>(),
        )
        .result()?;
        hipblasLtMatrixLayoutSetAttribute(
            desc,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batch_stride as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<i64>(),
        )
        .result()?;
    }
    Ok(())
}

fn rocm_dtype_needs_tensorwide_scale(dtype: hipDataType) -> bool {
    matches!(
        dtype,
        hipDataType::HIP_R_8F_E4M3 | hipDataType::HIP_R_8F_E5M2
    )
}

fn set_scalar_scale_pointer(
    matmul_desc: hipblasLtMatmulDesc_t,
    attr: hipblasLtMatmulDescAttributes_t,
    ptr: u64,
) -> anyhow::Result<()> {
    unsafe {
        hipblasLtMatmulDescSetAttribute(
            matmul_desc,
            attr,
            &ptr as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<u64>(),
        )
        .result()?;
    }
    Ok(())
}

fn run_hipblaslt_matmul(
    stream: &Arc<HipStream>,
    hipblas_lt: &Arc<HipBlasLt>,
    spec: &LtMatmulSpec,
    ptrs: LtMatmulPointers,
) -> anyhow::Result<()> {
    if spec.problem.m == 0 || spec.problem.n == 0 || spec.problem.k == 0 {
        return Err(anyhow::anyhow!(
            "hipblasLt matmul got zero-sized dimensions: m={}, n={}, k={}",
            spec.problem.m,
            spec.problem.n,
            spec.problem.k
        ));
    }

    let mut resources = LtRawDescriptors::default();
    let mut heuristic: hipblasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut algo_count: i32 = 0;

    let workspace = stream.alloc::<u8>(spec.workspace_size)?;
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);

    let a_scale = if rocm_dtype_needs_tensorwide_scale(spec.a.dtype) && ptrs.a_scale.is_none() {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };
    let b_scale = if rocm_dtype_needs_tensorwide_scale(spec.b.dtype) && ptrs.b_scale.is_none() {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };
    let c_scale = if rocm_dtype_needs_tensorwide_scale(spec.c.dtype) {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };
    let d_scale = if rocm_dtype_needs_tensorwide_scale(spec.d.dtype) {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };

    unsafe {
        hipblasLtMatmulDescCreate(
            &mut resources.matmul_desc,
            spec.compute.compute_type,
            spec.compute.scale_dtype,
        )
        .result()?;

        hipblasLtMatmulDescSetAttribute(
            resources.matmul_desc,
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSA,
            &spec.trans_a as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<hipblasOperation_t>(),
        )
        .result()?;
        hipblasLtMatmulDescSetAttribute(
            resources.matmul_desc,
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSB,
            &spec.trans_b as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<hipblasOperation_t>(),
        )
        .result()?;
        hipblasLtMatmulDescSetAttribute(
            resources.matmul_desc,
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_EPILOGUE,
            &spec.compute.epilogue as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<hipblasLtEpilogue_t>(),
        )
        .result()?;
        if let Some(bias_ptr) = ptrs.bias {
            hipblasLtMatmulDescSetAttribute(
                resources.matmul_desc,
                hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_ptr as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<u64>(),
            )
            .result()?;
        }
    }

    let (a_scale_ptr, _a_scale_guard) = if let Some(ptr) = ptrs.a_scale {
        (Some(ptr), None)
    } else if let Some(scale) = &a_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    let (b_scale_ptr, _b_scale_guard) = if let Some(ptr) = ptrs.b_scale {
        (Some(ptr), None)
    } else if let Some(scale) = &b_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    let (c_scale_ptr, _c_scale_guard) = if let Some(scale) = &c_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    let (d_scale_ptr, _d_scale_guard) = if let Some(scale) = &d_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    if let Some(ptr) = a_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            ptr,
        )?;
    }
    if let Some(ptr) = b_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            ptr,
        )?;
    }
    if let Some(ptr) = c_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
            ptr,
        )?;
    }
    if let Some(ptr) = d_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
            ptr,
        )?;
    }

    create_matrix_layout(&mut resources.a_desc, spec.a)?;
    create_matrix_layout(&mut resources.b_desc, spec.b)?;
    create_matrix_layout(&mut resources.c_desc, spec.c)?;
    create_matrix_layout(&mut resources.d_desc, spec.d)?;

    if spec.problem.batch_count > 1 {
        for (desc, matrix) in [
            (resources.a_desc, spec.a),
            (resources.b_desc, spec.b),
            (resources.c_desc, spec.c),
            (resources.d_desc, spec.d),
        ] {
            set_strided_batch_layout(desc, spec.problem.batch_count, matrix.batch_stride)?;
        }
    }

    unsafe {
        hipblasLtMatmulPreferenceCreate(&mut resources.preference).result()?;
        hipblasLtMatmulPreferenceSetAttribute(
            resources.preference,
            hipblasLtMatmulPreferenceAttributes_t::HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &spec.workspace_size as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
        .result()?;

        hipblasLtMatmulAlgoGetHeuristic(
            hipblas_lt.handle(),
            resources.matmul_desc,
            resources.a_desc,
            resources.b_desc,
            resources.c_desc,
            resources.d_desc,
            resources.preference,
            1,
            &mut heuristic,
            &mut algo_count,
        )
        .result()?;

        if algo_count == 0 {
            return Err(anyhow::anyhow!("No suitable hipblasLt algorithm found"));
        }

        let alpha_ptr = spec.compute.alpha.as_ptr();
        let beta_ptr = spec.compute.beta.as_ptr();
        hipblasLtMatmul(
            hipblas_lt.handle(),
            resources.matmul_desc,
            alpha_ptr,
            ptrs.a as *const std::ffi::c_void,
            resources.a_desc,
            ptrs.b as *const std::ffi::c_void,
            resources.b_desc,
            beta_ptr,
            ptrs.c as *const std::ffi::c_void,
            resources.c_desc,
            ptrs.d as *mut std::ffi::c_void,
            resources.d_desc,
            &heuristic.algo,
            workspace_ptr as *mut std::ffi::c_void,
            spec.workspace_size,
            stream.hip_stream() as *mut _,
        )
        .result()?;
    }

    Ok(())
}

fn resolve_hipblaslt_pointers(
    self_node: NodeIndex,
    inputs: &[NodeIndex],
    buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
    beta: f64,
    epilogue: hipblasLtEpilogue_t,
    a_scale_input: bool,
    b_scale_input: bool,
) -> anyhow::Result<LtMatmulPointers> {
    if inputs.len() < 2 {
        return Err(anyhow::anyhow!(
            "hipblasLt matmul expected at least 2 inputs (A, B[, C]), got {}",
            inputs.len()
        ));
    }

    let a = buffers
        .get(&inputs[0])
        .ok_or_else(|| anyhow::anyhow!("missing hipblasLt A input buffer"))?
        .ptr();
    let b = buffers
        .get(&inputs[1])
        .ok_or_else(|| anyhow::anyhow!("missing hipblasLt B input buffer"))?
        .ptr();
    let d = buffers
        .get(&self_node)
        .ok_or_else(|| anyhow::anyhow!("missing hipblasLt output buffer"))?
        .ptr();
    let mut next_input = 2;
    let c = if beta == 0.0 {
        d
    } else {
        let c_input = inputs.get(next_input).ok_or_else(|| {
            anyhow::anyhow!("hipblasLt matmul with beta={beta} requires a third C input")
        })?;
        next_input += 1;
        buffers
            .get(c_input)
            .ok_or_else(|| anyhow::anyhow!("missing hipblasLt C input buffer"))?
            .ptr()
    };

    let bias = if epilogue_uses_bias(epilogue) {
        let bias_input = inputs.get(next_input).ok_or_else(|| {
            anyhow::anyhow!("hipblasLt matmul with {epilogue:?} epilogue requires a bias input")
        })?;
        next_input += 1;
        Some(
            buffers
                .get(bias_input)
                .ok_or_else(|| anyhow::anyhow!("missing hipblasLt bias input buffer"))?
                .ptr(),
        )
    } else {
        None
    };

    let a_scale = if a_scale_input {
        let scale_input = inputs
            .get(next_input)
            .ok_or_else(|| anyhow::anyhow!("hipblasLt matmul requires an A scale input pointer"))?;
        next_input += 1;
        Some(
            buffers
                .get(scale_input)
                .ok_or_else(|| anyhow::anyhow!("missing hipblasLt A scale input buffer"))?
                .ptr(),
        )
    } else {
        None
    };

    let b_scale = if b_scale_input {
        let scale_input = inputs
            .get(next_input)
            .ok_or_else(|| anyhow::anyhow!("hipblasLt matmul requires a B scale input pointer"))?;
        Some(
            buffers
                .get(scale_input)
                .ok_or_else(|| anyhow::anyhow!("missing hipblasLt B scale input buffer"))?
                .ptr(),
        )
    } else {
        None
    };

    Ok(LtMatmulPointers {
        a,
        b,
        c,
        d,
        bias,
        a_scale,
        b_scale,
    })
}

fn epilogue_uses_bias(epilogue: hipblasLtEpilogue_t) -> bool {
    matches!(
        epilogue,
        hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_BIAS
            | hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_RELU_BIAS
            | hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_RELU_AUX_BIAS
            | hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_GELU_BIAS
            | hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
    )
}

impl HipblasLt {
    fn get_hipblaslt(&self, stream: &Arc<HipStream>) -> anyhow::Result<Arc<HipBlasLt>> {
        if let Some(hipblas_lt) = self.hipblas_lt.get() {
            return Ok(hipblas_lt.clone());
        }
        let created = try_create_hipblaslt(stream.clone()).map_err(|message| {
            anyhow::anyhow!("hipblasLt unavailable on this machine: {message}")
        })?;
        let _ = self.hipblas_lt.set(created.clone());
        Ok(created)
    }

    #[cfg(test)]
    pub(crate) fn type_tuple(&self) -> (DType, DType, DType, DType, &'static str, DType) {
        (
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.d_dtype,
            compute_type_name(self.compute_type),
            self.scale_dtype,
        )
    }

    #[cfg(test)]
    pub(crate) fn scale_values(&self) -> (f64, f64) {
        (self.alpha, self.beta)
    }

    #[cfg(test)]
    pub(crate) fn epilogue(&self) -> &'static str {
        epilogue_name(self.epilogue)
    }

    #[cfg(test)]
    pub(crate) fn matrix_orders(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        (
            order_name(self.a_order),
            order_name(self.b_order),
            order_name(self.c_order),
            order_name(self.d_order),
        )
    }

    #[cfg(test)]
    pub(crate) fn transpose_ops(&self) -> (&'static str, &'static str) {
        (
            transpose_op_name(self.a_layout),
            transpose_op_name(self.b_layout),
        )
    }

    #[cfg(test)]
    pub(crate) fn c_d_layouts_match(&self) -> bool {
        let normalize = |expr: Expression| expr.substitute('z', Expression::from(1)).simplify();
        normalize(self.ldc) == normalize(self.ldd)
            && normalize(self.stride_c) == normalize(self.stride_d)
            && self.c_order == self.d_order
    }

    #[cfg(test)]
    pub(crate) fn tensor_scale_inputs(&self) -> (bool, bool) {
        (self.a_scale_input, self.b_scale_input)
    }
}

impl HostOp for HipblasLt {
    fn execute(
        &self,
        stream: &Arc<HipStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // GEMM parameters — resolve z→1 for element stride before exec
        let resolve = |e: &Expression| -> Expression { e.substitute('z', Expression::from(1)) };
        let m = resolve(&self.m).exec(dyn_map).unwrap() as u64;
        let n = resolve(&self.n).exec(dyn_map).unwrap() as u64;
        let k = resolve(&self.k).exec(dyn_map).unwrap() as u64;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = resolve(&self.lda).exec(dyn_map).unwrap() as i64;
        let ldb = resolve(&self.ldb).exec(dyn_map).unwrap() as i64;
        let ldc = resolve(&self.ldc).exec(dyn_map).unwrap() as i64;
        let ldd = resolve(&self.ldd).exec(dyn_map).unwrap() as i64;
        let batch_count = resolve(&self.batch_count).exec(dyn_map).unwrap() as i32;
        let stride_a = resolve(&self.stride_a).exec(dyn_map).unwrap() as i64;
        let stride_b = resolve(&self.stride_b).exec(dyn_map).unwrap() as i64;
        let stride_c = resolve(&self.stride_c).exec(dyn_map).unwrap() as i64;
        let stride_d = resolve(&self.stride_d).exec(dyn_map).unwrap() as i64;

        // Get CUDA types based on the explicit hipblasLt type tuple.
        let a_rocm_dtype = dtype_to_rocm_dtype(self.a_dtype);
        let b_rocm_dtype = dtype_to_rocm_dtype(self.b_dtype);
        let c_rocm_dtype = dtype_to_rocm_dtype(self.c_dtype);
        let d_rocm_dtype = dtype_to_rocm_dtype(self.d_dtype);
        let scale_rocm_dtype = dtype_to_rocm_dtype(self.scale_dtype);
        let element_size = (self.d_dtype.bits() / 8) as u64;
        assert!(
            element_size > 0,
            "cuBLAS LT does not support sub-byte dtype {}",
            self.d_dtype
        );

        let alpha = LtScalar::from_f64(self.scale_dtype, self.alpha)?;
        let beta = LtScalar::from_f64(self.scale_dtype, self.beta)?;

        let ptrs = resolve_hipblaslt_pointers(
            self_node,
            inputs,
            buffers,
            self.beta,
            self.epilogue,
            self.a_scale_input,
            self.b_scale_input,
        )?;

        let (a_rows, a_cols) = if a_layout == hipblasOperation_t::HIPBLAS_OP_N {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if b_layout == hipblasOperation_t::HIPBLAS_OP_N {
            (k, n)
        } else {
            (n, k)
        };
        let lda = clamp_ld_for_order(lda, a_rows, a_cols, self.a_order);
        let ldb = clamp_ld_for_order(ldb, b_rows, b_cols, self.b_order);
        let ldc = clamp_ld_for_order(ldc, m, n, self.c_order);
        let ldd = clamp_ld_for_order(ldd, m, n, self.d_order);

        let _span = span!(
            Level::TRACE,
            "hipblaslt",
            m, n, k, lda, ldb, ldc, ldd, batch_count, ?a_layout, ?b_layout,
            ?self.a_order, ?self.b_order, ?self.c_order, ?self.d_order,
            ?self.a_dtype, ?self.b_dtype, ?self.c_dtype, ?self.d_dtype,
            ?self.compute_type, ?self.scale_dtype, self.alpha, self.beta,
            ?self.epilogue,
        )
        .entered();

        let hipblas_lt = self.get_hipblaslt(stream)?;

        // Allocate workspace (32 MiB)
        const WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
        let c_spec = LtMatrixSpec {
            dtype: c_rocm_dtype,
            rows: m,
            cols: n,
            ld: ldc,
            batch_stride: stride_c,
            order: self.c_order,
        };
        let d_spec = LtMatrixSpec {
            dtype: d_rocm_dtype,
            rows: m,
            cols: n,
            ld: ldd,
            batch_stride: stride_d,
            order: self.d_order,
        };
        let spec = LtMatmulSpec {
            problem: LtMatmulProblem {
                m,
                n,
                k,
                batch_count,
            },
            trans_a: a_layout,
            trans_b: b_layout,
            a: LtMatrixSpec {
                dtype: a_rocm_dtype,
                rows: a_rows,
                cols: a_cols,
                ld: lda,
                batch_stride: stride_a,
                order: self.a_order,
            },
            b: LtMatrixSpec {
                dtype: b_rocm_dtype,
                rows: b_rows,
                cols: b_cols,
                ld: ldb,
                batch_stride: stride_b,
                order: self.b_order,
            },
            c: c_spec,
            d: d_spec,
            compute: LtComputeSpec {
                compute_type: self.compute_type,
                scale_dtype: scale_rocm_dtype,
                alpha,
                beta,
                epilogue: self.epilogue,
            },
            workspace_size: WORKSPACE_SIZE,
        };

        run_hipblaslt_matmul(stream, &hipblas_lt, &spec, ptrs)?;

        // No stream.synchronize() here — CUDA stream ordering guarantees
        // sequential execution. The runtime syncs once at the end of execute().
        Ok(())
    }

    fn output_size(&self) -> Expression {
        let resolve = |e: &Expression| -> Expression { e.substitute('z', Expression::from(1)) };
        resolve(&self.batch_count) * resolve(&self.m) * resolve(&self.n)
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.d_dtype.bits()).ceil_div(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lt_scalar_packs_f32_scale_values() {
        match LtScalar::one(DType::F32).unwrap() {
            LtScalar::F32(value) => assert_eq!(value, 1.0),
            other => panic!("expected f32 scalar, got {other:?}"),
        }

        match LtScalar::zero(DType::F32).unwrap() {
            LtScalar::F32(value) => assert_eq!(value, 0.0),
            other => panic!("expected f32 scalar, got {other:?}"),
        }
    }

    #[test]
    fn lt_scalar_packs_f64_scale_values() {
        match LtScalar::one(DType::F64).unwrap() {
            LtScalar::F64(value) => assert_eq!(value, 1.0),
            other => panic!("expected f64 scalar, got {other:?}"),
        }

        match LtScalar::zero(DType::F64).unwrap() {
            LtScalar::F64(value) => assert_eq!(value, 0.0),
            other => panic!("expected f64 scalar, got {other:?}"),
        }
    }

    #[test]
    fn lt_scalar_packs_low_precision_scale_values() {
        match LtScalar::one(DType::F16).unwrap() {
            LtScalar::F16(value) => assert_eq!(f32::from(value), 1.0),
            other => panic!("expected f16 scalar, got {other:?}"),
        }

        match LtScalar::zero(DType::Bf16).unwrap() {
            LtScalar::Bf16(value) => assert_eq!(f32::from(value), 0.0),
            other => panic!("expected bf16 scalar, got {other:?}"),
        }
    }

    #[test]
    fn lt_scalar_rejects_non_host_scalar_scale_dtypes() {
        assert!(LtScalar::one(DType::TF32).is_err());
        assert!(LtScalar::zero(DType::F8E4M3).is_err());
    }

    #[test]
    fn fp8_rocm_dtypes_request_tensorwide_scales() {
        assert!(rocm_dtype_needs_tensorwide_scale(
            hipDataType::HIP_R_8F_E4M3
        ));
        assert!(rocm_dtype_needs_tensorwide_scale(
            hipDataType::HIP_R_8F_E5M2
        ));
        assert!(!rocm_dtype_needs_tensorwide_scale(hipDataType::HIP_R_32F));
    }

    #[test]
    fn hipblasLt_pointers_alias_output_as_c_for_two_input_beta_zero() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

        let ptrs = resolve_hipblaslt_pointers(
            output,
            &[a, b],
            &buffers,
            0.0,
            hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
    }

    #[test]
    fn hipblasLt_pointers_ignore_extra_inputs_for_beta_zero() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let extra = NodeIndex::new(3);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (extra, 0xEEEE)]);

        let ptrs = resolve_hipblaslt_pointers(
            output,
            &[a, b, extra],
            &buffers,
            0.0,
            hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
    }

    #[test]
    fn hipblasLt_pointers_use_distinct_c_input_when_present() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let c = NodeIndex::new(3);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (c, 0xC000)]);

        let ptrs = resolve_hipblaslt_pointers(
            output,
            &[a, b, c],
            &buffers,
            1.0,
            hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xC000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
    }

    #[test]
    fn hipblasLt_pointers_use_bias_input_for_bias_epilogue() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let bias = NodeIndex::new(3);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (bias, 0xB1A5)]);

        let ptrs = resolve_hipblaslt_pointers(
            output,
            &[a, b, bias],
            &buffers,
            0.0,
            hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_BIAS,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, Some(0xB1A5));
    }

    #[test]
    fn hipblasLt_pointers_use_tensor_scale_inputs_after_base_inputs() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let a_scale = NodeIndex::new(3);
        let b_scale = NodeIndex::new(4);
        let buffers = buffers_for(&[
            (output, 0xD000),
            (a, 0xA000),
            (b, 0xB000),
            (a_scale, 0xA5A5),
            (b_scale, 0xB5B5),
        ]);

        let ptrs = resolve_hipblaslt_pointers(
            output,
            &[a, b, a_scale, b_scale],
            &buffers,
            0.0,
            hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT,
            true,
            true,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
        assert_eq!(ptrs.a_scale, Some(0xA5A5));
        assert_eq!(ptrs.b_scale, Some(0xB5B5));
    }

    #[test]
    fn hipblasLt_pointers_reject_two_input_nonzero_beta() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

        let err = resolve_hipblaslt_pointers(
            output,
            &[a, b],
            &buffers,
            1.0,
            hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("requires a third C input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn hipblasLt_pointers_reject_missing_bias_input() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

        let err = resolve_hipblaslt_pointers(
            output,
            &[a, b],
            &buffers,
            0.0,
            hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_BIAS,
            false,
            false,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("requires a bias input"),
            "unexpected error: {err}"
        );
    }

    fn buffers_for(entries: &[(NodeIndex, u64)]) -> FxHashMap<NodeIndex, DeviceBuffer> {
        entries
            .iter()
            .map(|(node, ptr)| (*node, DeviceBuffer::new(*ptr, 16)))
            .collect()
    }
}
