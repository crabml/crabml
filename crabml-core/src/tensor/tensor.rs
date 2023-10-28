use crate::error::Result;

pub trait Tensor<'a>: Sized + Clone + TensorArithmetics {
    fn view(self, shape: &[usize]) -> Result<Self>;

    fn repeat(self, repeats: &[usize]) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn as_ref<'b>(&'b self) -> Self where 'b: 'a;

    fn extend(&mut self, rhs: &Self) -> Result<()>;

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()>;
}

pub trait TensorArithmetics: Sized {
    fn mul_inplace(self, y: &Self) -> Result<Self>;

    fn add_inplace(self, y: &Self) -> Result<Self>;

    fn div_scalar_inplace(self, y: f32) -> Result<Self>;

    fn matmul(&self, y: &Self) -> Result<Self>;

    fn batch_matmul(&self, y: &Self) -> Result<Self>;

    fn silu_inplace(self) -> Result<()>;

    fn rms_norm_inplace(self, eps: f32) -> Result<Self>;

    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self>;
}