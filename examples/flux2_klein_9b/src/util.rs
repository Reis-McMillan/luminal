use luminal::prelude::*;

pub trait Materialize {
    fn materialize(self) -> Self;
}

impl Materialize for GraphTensor {
    fn materialize(self) -> Self {
        self * 1.0
    }
}
