use crate::error::Result;
use crate::gguf::GGMLType;

pub trait Tensor: Sized + Clone {
    type Pool;

    fn new(data: Vec<f32>, shape: &[usize], pool: Self::Pool) -> Result<Self>;

    fn alloc(shape: &[usize], pool: Self::Pool) -> Result<Self>;

    fn view(self, shape: &[usize]) -> Result<Self>;

    fn repeat(self, repeats: &[usize]) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn extend(&mut self, rhs: &Self) -> Result<()>;

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()>;
}

pub mod ops {
    use super::*;

    pub trait AsRef {
        type Output: Tensor;
        fn as_ref<'a>(&'a self) -> Self::Output;
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

    pub trait MulInplace<RHS: Tensor> {
        type Output: Tensor;
        fn mul_inplace(self, rhs: &RHS) -> Result<Self::Output>;
    }

    pub trait AddInplace<RHS: Tensor> {
        type Output: Tensor;
        fn add_inplace(self, rhs: &RHS) -> Result<Self::Output>;
    }

    pub trait DivScalarInplace: Sized {
        fn div_scalar_inplace(self, rhs: f32) -> Result<Self>;
    }

    pub trait Matmul<RHS: Tensor> {
        type Output: Tensor;
        fn matmul(&self, y: &RHS) -> Result<Self::Output>;
    }

    pub trait BatchMatmul<RHS: Tensor> {
        type Output: Tensor;
        fn batch_matmul(&self, y: &RHS) -> Result<Self::Output>;
    }
}
