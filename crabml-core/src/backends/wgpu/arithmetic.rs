use crate::backends::wgpu::wgpu_tensor::WgpuTensor;
use crate::error::Result;
use crate::tensor::TensorArithmetics;

impl TensorArithmetics for WgpuTensor {
    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self> {
        todo!()
    }

    fn rms_norm_inplace(self, eps: f32) -> Result<Self> {
        todo!()
    }

    fn softmax_inplace(self, axis: usize) -> Result<Self> {
        todo!()
    }

    fn silu_inplace(self) -> Result<Self> {
        todo!()
    }

    fn mul_inplace(self, rhs: &Self) -> Result<Self> {
        todo!()
    }

    fn add_inplace(self, rhs: &Self) -> Result<Self> {
        todo!()
    }

    fn div_scalar_inplace(self, rhs: f32) -> Result<Self> {
        todo!()
    }

    fn matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }

    fn batch_matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }
}
