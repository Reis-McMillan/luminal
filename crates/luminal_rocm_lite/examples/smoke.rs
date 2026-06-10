// crates/luminal_rocm_lite/examples/smoke.rs
use luminal::prelude::*;
use luminal_rocm_lite::runtime::RocmRuntime;
use rocmrc::HipContext;

fn main() {
    let ctx = HipContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let mut cx = Graph::default();
    let a = cx.tensor(64).persist();
    let b = cx.tensor(64).persist();
    let c = (a + b).output();

    let mut rt = RocmRuntime::initialize(stream);
    rt.set_data(a, vec![1.0_f32; 64]);
    rt.set_data(b, vec![2.0_f32; 64]);
    cx.build_search_space::<RocmRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::new(5));
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    assert!(result.iter().all(|x| (x - 3.0).abs() < 1e-5));
    println!("ok: {:?}", &result[..4]);
}
