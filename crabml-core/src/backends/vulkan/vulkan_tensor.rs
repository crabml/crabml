use std::sync::Arc;

use vulkano::descriptor_set::WriteDescriptorSet;

use super::vulkan_device::VulkanTensorDeviceRef;
use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::gguf::GGMLType;
use crate::tensor::Tensor;
use crate::tensor::TensorStrider;

#[derive(Clone)]
pub struct VulkanTensor {
    buf: vulkano::buffer::Subbuffer<[u8]>,
    dtype: GGMLType,
    capacity: usize, // max element count
    strider: TensorStrider,
    device: VulkanTensorDeviceRef,
    name: Option<String>,
}

impl VulkanTensor {
    pub fn new(src: &[f32], shape: &[usize], device: VulkanTensorDeviceRef) -> Result<Self> {
        let buf = device.inner.make_device_buffer_from(src);
        let strider = TensorStrider::new(shape.to_vec());
        if strider.len() != src.len() {
            return Err(Error::new(
                ErrorKind::TensorError,
                "new: buffer size mismatch",
            ));
        };
        Ok(Self {
            buf,
            dtype: GGMLType::F32,
            capacity: src.len(),
            strider,
            device,
            name: None,
        })
    }

    pub fn export(&self, dst: &mut [f32]) -> Result<()> {
        let buf_size = std::mem::size_of_val(dst);
        if buf_size > self.device.opts.staging_buf_bytes {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "buffer size exceeded staging buffer limit: {}, got: {}",
                    self.device.opts.staging_buf_bytes, buf_size,
                ),
            )
                .into());
        }

        let dst_bytes = bytemuck::cast_slice_mut(dst);
        self.device
            .inner
            .copy_device_buffer_to_cpu(self.buf.clone(), dst_bytes);
        Ok(())
    }
}

impl Tensor for VulkanTensor {
    type Device = VulkanTensorDeviceRef;

    fn alloc(shape: &[usize], dtype: GGMLType, device: Self::Device) -> Result<Self> {
        todo!()
    }

    fn resize(self, axis: usize, n: usize) -> Result<Self> {
        todo!()
    }

    fn dtype(&self) -> GGMLType {
        todo!()
    }

    fn with_strider(self, strider: TensorStrider) -> Result<Self> {
        todo!()
    }

    fn with_name(self, name: String) -> Self {
        todo!()
    }

    fn reshape(self, shape: &[usize]) -> Result<Self> {
        todo!()
    }

    fn transpose(self, shape: &[usize]) -> Result<Self> {
        todo!()
    }

    fn contiguous(self) -> Result<Self> {
        todo!()
    }

    fn shape(&self) -> &[usize] {
        todo!()
    }

    fn strider(&self) -> &TensorStrider {
        todo!()
    }

    fn concatenate(&mut self, rhs: &Self, axis: usize) -> Result<()> {
        todo!()
    }

    fn copy_rows_from(&mut self, rhs: &Self, rows: &[usize]) -> Result<()> {
        todo!()
    }

    fn export(&self, buf: &mut [f32]) -> Result<()> {
        todo!()
    }

    fn dup(&self) -> Result<Self> {
        todo!()
    }

    fn rope_inplace(
        self,
        mode: crate::tensor::RopeMode,
        pos: usize,
        rope_dims: usize,
    ) -> Result<Self> {
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

    fn gelu_inplace(self) -> Result<Self> {
        todo!()
    }

    fn mul_inplace(self, rhs: &Self) -> Result<Self> {
        todo!()
    }

    fn add_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.strider.is_contiguous());
        assert!(rhs.strider.is_contiguous());

        // TODO: pass n_elm as meta
        let n_elms = self.strider.len() as u32;
        let bufs = vec![self.buf.clone(), rhs.buf.clone()];
        let dispatches = [n_elms / 32 + 1, 1, 1];
        self.device.inner.dispatch_compute("add", bufs, dispatches);
        Ok(self)
    }

    fn div_scalar_inplace(self, rhs: f32) -> Result<Self> {
        todo!()
    }

    fn scale_inplace(self, rhs: f32) -> Result<Self> {
        todo!()
    }

    fn matmul_vec(&self, y: &Self) -> Result<Self> {
        todo!()
    }

    fn batch_matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::vulkan::vulkan_device::{VulkanTensorDevice, VulkanTensorDeviceOptions};

    #[test]
    fn test_add() {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
    }
}
