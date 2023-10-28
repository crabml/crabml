use crate::error::Result;
use crate::gguf::GGMLType;

pub trait Tensor: Sized + Clone + TensorArithmetics {
    type Backend;

    fn alloc(shape: &[usize], backend: Self::Backend) -> Result<Self>;

    fn from_bytes(bytes: &[u8], typ: GGMLType, shape: &[usize], backend: Self::Backend) -> Result<()>;

    fn view(self, shape: &[usize]) -> Result<Self>;

    fn repeat(self, repeats: &[usize]) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn as_ref(&self) -> Self;

    fn extend(&mut self, rhs: &Self) -> Result<()>;

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()>;

    fn export(&self, buf: &mut [f32]) -> Result<()>;
}

pub trait TensorArithmetics: Sized {
    fn mul_inplace(x: &mut Self, y: &Self) -> Result<()>;

    fn add_inplace(x: &mut Self, y: &Self) -> Result<()>;

    fn div_scalar_inplace(x: &mut Self, y: f32) -> Result<()>;

    fn matmul(x: &mut Self, y: &Self) -> Result<Self>;

    fn batch_matmul(x: &mut Self, y: &Self) -> Result<Self>;

    fn silu_inplace(x: &mut Self) -> Result<()>;

    fn rms_norm_inplace(x: &mut Self, eps: f32) -> Result<()>;

    fn rope_inplace(x: &mut Self, pos: usize, rope_dims: usize) -> Result<()>;
}