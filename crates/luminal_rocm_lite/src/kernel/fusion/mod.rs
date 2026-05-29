//! Binary-inclusive elementwise kernel fusion.
//!
//! - `markers` — `FusionStart` / `FusionEnd` ops + the seven egglog rule
//!   families that build and extend FE-bracketed regions.
//! - `elementwise` — generic region-internal HIP elementwise op variants.
//! - `region_codegen` — `kernel_to_host` calls into here to collapse each
//!   FE-rooted region into a single HIP kernel at compile time.
//!
//! The LLIR keeps `FusionStart` / generic elementwise / `FusionEnd` nodes after
//! extraction; `region_codegen` is the only place that walks them.

pub mod elementwise;
pub mod markers;
pub mod region_codegen;

pub use elementwise::{RocmBinaryElementwise, RocmUnaryElementwise};
pub use markers::{FusionEnd, FusionStart};

/// All fusion-related op types that the egglog runtime needs to know about
/// (markers + interior generic elementwise variants). Combined into a flat
/// tuple for the `Ops` registry in `kernel::mod`.
pub type Ops = (markers::Ops, elementwise::Ops);
