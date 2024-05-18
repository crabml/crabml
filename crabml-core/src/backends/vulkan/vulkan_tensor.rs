use std::sync::Arc;

use super::vulkan_device::VulkanTensorDevice;
use super::vulkan_device::VulkanTensorDeviceRef;
use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::gguf::GGMLType;
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
