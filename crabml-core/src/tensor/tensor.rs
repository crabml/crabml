use crate::{error::Result, gguf::GGMLType};

pub trait Tensor<'a>: Sized + Clone + TensorArithmetics {
    type Pool;

    fn new(data: Vec<f32>, shape: &[usize], pool: Self::Pool) -> Result<Self>;

    fn alloc(shape: &[usize], pool: Self::Pool) -> Result<Self>;

    fn from_bytes(buf: &'a [u8], typ: GGMLType, shape: &[usize], pool: Self::Pool) -> Result<Self>;

    fn view(self, shape: &[usize]) -> Result<Self>;

    fn repeat(self, repeats: &[usize]) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn as_ref<'b>(&'b self) -> Self
    where 'b: 'a;

    fn extend(&mut self, rhs: &Self) -> Result<()>;

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()>;
}

pub trait TensorArithmetics: Sized {
    fn mul_inplace(self, y: &Self) -> Result<Self>;

    fn add_inplace(self, y: &Self) -> Result<Self>;

    fn div_scalar_inplace(self, y: f32) -> Result<Self>;

    fn matmul(&self, y: &Self) -> Result<Self>;

    fn batch_matmul(&self, y: &Self) -> Result<Self>;

    fn silu_inplace(self) -> Result<Self>;

    fn softmax_inplace(self, axis: usize) -> Result<Self>;

    fn rms_norm_inplace(self, eps: f32) -> Result<Self>;

    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self>;
}
