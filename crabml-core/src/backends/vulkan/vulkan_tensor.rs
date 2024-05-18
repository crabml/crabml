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
    buf: Arc<vulkano::buffer::Subbuffer<[u8]>>,
    dtype: GGMLType,
    capacity: usize, // max element count
    strider: TensorStrider,
    device: VulkanTensorDeviceRef,
    name: Option<String>,
}

impl VulkanTensor {
    pub fn new(src: &[f32], shape: &[usize], device: VulkanTensorDeviceRef) -> Result<Self> {
        let buf = Arc::new(device.inner.make_device_buffer_from(src));
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
}
