use super::strider::TensorStrider;
use crate::error::Result;
use crate::gguf::GGMLType;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RopeMode {
    Llama,
    Neox,
}

pub trait Tensor: Sized + Clone {
    type DeviceRef: Clone;

    fn from_cpu(
        buf: &[u8],
        shape: &[usize],
        dtype: GGMLType,
        device: Self::DeviceRef,
    ) -> Result<Self>;

    /// alloc an owned tensor, only used on storing activations and kv caches.
    /// only F32 and F16 are supported.
    fn alloc(shape: &[usize], dtype: GGMLType, device: Self::DeviceRef) -> Result<Self>;

    /// resize the tensor to a smaller size, the underlying storage is not changed,
    /// it's useful on pre-allocated tensors, such as kv caches, which is the only
    /// place where we use this function.
    fn resize(self, axis: usize, n: usize) -> Result<Self>;

    fn dtype(&self) -> GGMLType;

    fn with_strider(self, strider: TensorStrider) -> Result<Self>;

    fn with_name(self, name: String) -> Self;

    fn reshape(self, shape: &[usize]) -> Result<Self>;

    fn transpose(self, shape: &[usize]) -> Result<Self>;

    fn contiguous(self) -> Result<Self>;

    fn shape(&self) -> &[usize];

    fn strider(&self) -> &TensorStrider;

    fn concatenate(&mut self, rhs: &Self, axis: usize) -> Result<()>;

    /// copy from another tensor. used on loading weights from vocab table.
    /// only support copy from 2d tensor to 2d or 1d tensor.
    fn copy_rows_from(&mut self, rhs: &Self, rows: &[usize]) -> Result<()>;

    fn export(&self, buf: &mut [f32]) -> Result<()>;

    /// duplicate the tensor and the underlying storage
    fn dup(&self) -> Result<Self>;

    fn rope_inplace(self, mode: RopeMode, pos: usize, rope_dims: usize) -> Result<Self>;

    fn rms_norm_inplace(self, eps: f32) -> Result<Self>;

    fn softmax_inplace(self, axis: usize) -> Result<Self>;

    fn silu_inplace(self) -> Result<Self>;

    fn gelu_inplace(self) -> Result<Self>;

    fn mul_inplace(self, rhs: &Self) -> Result<Self>;

    /// there're two cases:
    /// 1. both self and rhs have the same shape, it's an element-wise operation.
    /// 2. self are 2d tensor, rhs is 1d tensor, it's a broadcast element-wise operation.
    fn add_inplace(self, rhs: &Self) -> Result<Self>;

    fn scale_inplace(self, rhs: f32) -> Result<Self>;

    fn matmul_vec(&self, y: &Self) -> Result<Self>;

    fn batch_matmul(&self, y: &Self) -> Result<Self>;
}
