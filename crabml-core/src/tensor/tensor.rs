use crate::error::Result;
use crate::gguf::GGMLType;

use super::strider::TensorStrider;

pub trait Tensor: Sized + Clone {
    type Pool;

    fn new(data: Vec<f32>, shape: &[usize], pool: Self::Pool) -> Result<Self>;

    fn alloc(shape: &[usize], pool: Self::Pool) -> Result<Self>;

    fn with_strider(self, strider: TensorStrider) -> Result<Self>;

    fn reshape(self, shape: &[usize]) -> Result<Self>;

    fn repeat(self, repeats: &[usize]) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn strider(&self) -> &TensorStrider;
}

pub mod ops {
    use super::*;

    pub trait Extend {
        fn extend(&mut self, rhs: &Self) -> Result<()>;
    }

    pub trait CopyFrom {
        fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()>;
    }

    pub trait RopeInplace: Sized {
        fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self>;
    }

    pub trait RmsNormInplace: Sized {
        fn rms_norm_inplace(self, eps: f32) -> Result<Self>;
    }

    pub trait SoftmaxInplace: Sized {
        fn softmax_inplace(self, axis: usize) -> Result<Self>;
    }

    pub trait SiluInplace: Sized {
        fn silu_inplace(self) -> Result<Self>;
    }

    pub trait MulInplace: Sized {
        fn mul_inplace(self, rhs: &Self) -> Result<Self>;
    }

    pub trait AddInplace: Sized {
        fn add_inplace(self, rhs: &Self) -> Result<Self>;
    }

    pub trait DivScalarInplace: Sized {
        fn div_scalar_inplace(self, rhs: f32) -> Result<Self>;
    }

    pub trait Matmul: Sized {
        fn matmul(&self, y: &Self) -> Result<Self>;
    }

    pub trait BatchMatmul: Sized {
        fn batch_matmul(&self, y: &Self) -> Result<Self>;
    }
}
