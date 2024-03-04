use super::strider::TensorStrider;
use crate::error::Result;
use crate::gguf::GGMLType;

pub trait Tensor: Sized + Clone {
    type Device: Clone;

    /// alloc an owned tensor, only used on storing activations and kv caches.
    /// only F32 and F16 (not yet implemented) are supported.
    /// TODO: add dtype parameter
    fn alloc(shape: &[usize], capacity: Option<usize>, device: Self::Device) -> Result<Self>;

    fn dtype(&self) -> GGMLType;

    fn with_strider(self, strider: TensorStrider) -> Result<Self>;

    fn with_name(self, name: String) -> Self;

    fn reshape(self, shape: &[usize]) -> Result<Self>;

    fn repeat_n(self, n: usize) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn strider(&self) -> &TensorStrider;

    fn extend(&mut self, rhs: &Self) -> Result<()>;

    /// copy from another tensor. used on loading weights from vocab table.
    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()>;

    fn export(&self, buf: &mut [f32]) -> Result<()>;

    /// duplicate the tensor and the underlying storage
    fn dup(&self) -> Result<Self>;

    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self>;

    fn rms_norm_inplace(self, eps: f32) -> Result<Self>;

    fn softmax_inplace(self, axis: usize) -> Result<Self>;

    fn silu_inplace(self) -> Result<Self>;

    fn mul_inplace(self, rhs: &Self) -> Result<Self>;

    fn add_inplace(self, rhs: &Self) -> Result<Self>;

    fn div_scalar_inplace(self, rhs: f32) -> Result<Self>;

    fn scale_inplace(self, rhs: f32) -> Result<Self>;

    fn matmul_vec(&self, y: &Self) -> Result<Self>;

    fn batch_matmul_vec(&self, y: &Self) -> Result<Self>;
}
