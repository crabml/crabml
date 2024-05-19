#![allow(dead_code, unused_variables)]

use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGMLType;
use crabml::tensor::Tensor;
use crabml::tensor::TensorStrider;

use super::vulkan_device::VulkanTensorDeviceRef;
use crate::push_constants::ArithmeticPushConstants;

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
}

impl Tensor for VulkanTensor {
    type DeviceRef = VulkanTensorDeviceRef;

    fn from_cpu(
        buf: &[u8],
        shape: &[usize],
        dtype: GGMLType,
        device: Self::DeviceRef,
    ) -> Result<Self> {
        todo!()
    }

    fn alloc(shape: &[usize], dtype: GGMLType, device: Self::DeviceRef) -> Result<Self> {
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

    fn export(&self, dst: &mut [f32]) -> Result<()> {
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

    fn dup(&self) -> Result<Self> {
        todo!()
    }

    fn rope_inplace(
        self,
        mode: crabml::tensor::RopeMode,
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
        assert!(self.strider.is_contiguous());
        assert!(rhs.strider.is_contiguous());
        assert!(self.strider.len() % rhs.strider.len() == 0);

        let n_elms = self.strider.len() as u32;
        let bufs = vec![self.buf.clone(), rhs.buf.clone()];
        let pcs = ArithmeticPushConstants {
            n_elms,
            op: '*' as u32,
            use_scalar_rhs: 0,
            scalar_rhs: 0.0,
        };
        let dispatches = [n_elms / 32 + 1, 1, 1];
        self.device
            .inner
            .dispatch_compute("arithmetic", bufs, pcs, dispatches);
        Ok(self)
    }

    fn add_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.strider.is_contiguous());
        assert!(rhs.strider.is_contiguous());
        assert!(self.strider.len() % rhs.strider.len() == 0);

        let n_elms = self.strider.len() as u32;
        let bufs = vec![self.buf.clone(), rhs.buf.clone()];
        let pcs = ArithmeticPushConstants {
            n_elms,
            op: '+' as u32,
            use_scalar_rhs: 0,
            scalar_rhs: 0.0,
        };
        let dispatches = [n_elms / 32 + 1, 1, 1];
        self.device
            .inner
            .dispatch_compute("arithmetic", bufs, pcs, dispatches);
        Ok(self)
    }

    fn scale_inplace(self, rhs: f32) -> Result<Self> {
        assert!(self.strider.is_contiguous());

        let n_elms = self.strider.len() as u32;
        let bufs = vec![self.buf.clone(), self.buf.clone()];
        let pcs = ArithmeticPushConstants {
            n_elms,
            op: '*' as u32,
            use_scalar_rhs: 1,
            scalar_rhs: rhs,
        };
        let dispatches = [n_elms / 32 + 1, 1, 1];
        self.device
            .inner
            .dispatch_compute("arithmetic", bufs, pcs, dispatches);
        Ok(self)
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
    use crabml::error::Result;
    use crabml::tensor::Tensor;

    use super::VulkanTensor;
    use crate::vulkan_device::VulkanTensorDevice;
    use crate::vulkan_device::VulkanTensorDeviceOptions;

    #[test]
    fn test_add() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());

        let buf1 = (0..32).map(|v| v as f32).collect::<Vec<_>>();
        let buf2 = vec![2.0; 32];

        let t1 = VulkanTensor::new(&buf1, &[32], d.clone()).unwrap();
        let t2 = VulkanTensor::new(&buf2, &[32], d.clone()).unwrap();

        let t1 = t1.add_inplace(&t2).unwrap();
        let t1 = t1.add_inplace(&t2).unwrap();
        let mut bufo = vec![0.0; 32];
        t1.export(&mut bufo)?;

        assert_eq!(bufo, vec![
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
            33.0, 34.0, 35.0
        ]);
        Ok(())
    }

    #[test]
    fn test_scale_inplace() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());

        let buf1 = (0..34).map(|v| v as f32).collect::<Vec<_>>();

        let t1 = VulkanTensor::new(&buf1, &[34], d.clone()).unwrap();

        let t1 = t1.scale_inplace(2.0).unwrap();
        let mut bufo = vec![0.0; 34];
        t1.export(&mut bufo)?;

        assert_eq!(bufo, vec![
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
            30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0,
            58.0, 60.0, 62.0, 64.0, 66.0
        ]);
        Ok(())
    }
}
