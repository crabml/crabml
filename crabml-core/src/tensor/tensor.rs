use super::strider::TensorStrider;
use crate::error::Result;

pub trait Tensor: Sized + Clone + TensorArithmetics {
    type Device: Clone;

    fn new(data: Vec<f32>, shape: &[usize], device: Self::Device) -> Result<Self>;

    fn alloc(shape: &[usize], device: Self::Device) -> Result<Self>;

    fn with_strider(self, strider: TensorStrider) -> Result<Self>;

    fn reshape(self, shape: &[usize]) -> Result<Self>;

    fn repeat(self, repeats: &[usize]) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn strider(&self) -> &TensorStrider;

    fn extend(&mut self, rhs: &Self) -> Result<()>;

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()>;

    fn export(&self) -> Result<Box<dyn Iterator<Item = f32> + '_>>;
}

pub trait TensorArithmetics: Sized {
    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self>;

    fn rms_norm_inplace(self, eps: f32) -> Result<Self>;

    fn softmax_inplace(self, axis: usize) -> Result<Self>;

    fn silu_inplace(self) -> Result<Self>;

    fn mul_inplace(self, rhs: &Self) -> Result<Self>;

    fn add_inplace(self, rhs: &Self) -> Result<Self>;

    fn div_scalar_inplace(self, rhs: f32) -> Result<Self>;

    fn matmul(&self, y: &Self) -> Result<Self>;

    fn batch_matmul(&self, y: &Self) -> Result<Self>;
}
