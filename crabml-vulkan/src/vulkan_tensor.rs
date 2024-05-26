#![allow(dead_code, unused_variables)]

use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGMLType;
use crabml::tensor::RopeMode;
use crabml::tensor::Tensor;
use crabml::tensor::TensorStrider;

use super::vulkan_device::VulkanTensorDeviceRef;
use crate::push_constants::ArithmeticPushConstants;
use crate::push_constants::RmsNormPushConstants;
use crate::push_constants::RopePushConstants;
use crate::push_constants::SoftmaxPushConstants;

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
        if axis >= self.shape().len() {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "resize: axis {} is larger than the current shape {:?}",
                    axis,
                    self.shape()
                ),
            )
                .into());
        }

        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = n;

        let new_len: usize = new_shape.iter().product();
        if new_len > self.capacity {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "resize: new shape {:?} is larger than the current shape {:?}",
                    new_shape,
                    self.shape()
                ),
            )
                .into());
        }

        let new_strider = self.strider.resize(&new_shape)?;
        Ok(Self {
            buf: self.buf,
            capacity: self.capacity,
            dtype: self.dtype,
            strider: new_strider,
            device: self.device.clone(),
            name: None,
        })
    }

    fn dtype(&self) -> GGMLType {
        self.dtype
    }

    fn with_strider(mut self, strider: TensorStrider) -> Result<Self> {
        self.strider = strider;
        Ok(self)
    }

    fn with_name(self, name: String) -> Self {
        todo!()
    }

    fn reshape(self, shape: &[usize]) -> Result<Self> {
        let strider = self.strider.reshape(shape.to_vec())?;
        self.with_strider(strider)
    }

    fn transpose(self, dims: &[usize]) -> Result<Self> {
        let strider = self.strider.transpose(dims)?;
        self.with_strider(strider)
    }

    fn contiguous(self) -> Result<Self> {
        todo!()
    }

    fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    fn strider(&self) -> &TensorStrider {
        &self.strider
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
        assert!(self.shape().len() == 3 || self.shape().len() == 2);
        assert!(self.strider.is_contiguous());
        assert!(mode == RopeMode::Llama, "TODO: only support Llama mode yet");

        let (rows, n_head, m) = if self.strider.dims() == 3 {
            (
                self.shape()[0],
                self.shape()[1],
                self.shape()[1] * self.shape()[2],
            )
        } else {
            (1, self.shape()[0], self.shape()[0] * self.shape()[1])
        };
        let pcs = RopePushConstants {
            n_batch: rows as u32,
            n_dims: m as u32,
            pos: pos as u32,
            n_heads: n_head as u32,
            n_rope_dims: rope_dims as u32,
        };
        let dispatches = [rows as u32 / 32 + 1, 1, 1];
        self.device
            .inner
            .dispatch_compute("rope", vec![self.buf.clone()], pcs, dispatches);
        Ok(self)
    }

    fn rms_norm_inplace(self, eps: f32) -> Result<Self> {
        assert!(self.strider.is_contiguous());
        assert!(self.shape().last().unwrap() % 32 == 0);
        assert!([1, 2, 3].contains(&self.shape().len()));

        let (n_rows, n_cols) = match self.shape().len() {
            3 => (self.shape()[0] * self.shape()[1], self.shape()[2]),
            2 => (self.shape()[0], self.shape()[1]),
            1 => (1, self.shape()[0]),
            _ => unreachable!(),
        };

        let bufs = vec![self.buf.clone()];
        let pcs = RmsNormPushConstants {
            n_rows: n_rows as u32,
            n_cols: n_cols as u32,
            eps,
        };
        // each thread block processes a row
        let dispatches = [n_rows as u32, 1, 1];
        self.device
            .inner
            .dispatch_compute("rms_norm", bufs, pcs, dispatches);
        Ok(self)
    }

    fn softmax_inplace(self, axis: usize) -> Result<Self> {
        assert!(axis == self.strider.dims() - 1);
        assert!(self.strider.is_contiguous());
        assert!(self.shape().len() == 3 || self.shape().len() == 2);

        let (n_rows, n_cols) = if self.shape().len() == 3 {
            (self.shape()[0] * self.shape()[1], self.shape()[2])
        } else {
            (self.shape()[0], self.shape()[1])
        };
        let bufs = vec![self.buf.clone()];
        let pcs = SoftmaxPushConstants {
            n_rows: n_rows as u32,
            n_cols: n_cols as u32,
        };
        // each thread block processes a row
        let dispatches = [n_rows as u32 / 32 + 1, 1, 1];

        self.device
            .inner
            .dispatch_compute("softmax", bufs, pcs, dispatches);

        Ok(self)
    }

    fn silu_inplace(self) -> Result<Self> {
        let n_elms = self.strider.len() as u32;
        let bufs = vec![self.buf.clone()];
        let dispatches = [n_elms / 32 + 1, 1, 1];
        self.device
            .inner
            .dispatch_compute("silu", bufs, (), dispatches);
        Ok(self)
    }

    fn gelu_inplace(self) -> Result<Self> {
        let n_elms = self.strider.len() as u32;
        let bufs = vec![self.buf.clone()];
        let dispatches = [n_elms / 32 + 1, 1, 1];
        self.device
            .inner
            .dispatch_compute("gelu", bufs, (), dispatches);
        Ok(self)
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
    use approx::assert_relative_eq;
    use crabml::error::Result;
    use crabml::tensor::RopeMode;
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

    #[test]
    fn test_silu_inplace() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = VulkanTensor::new(&v1, &[6], d.clone()).unwrap();
        let t1 = t1.silu_inplace()?;

        let mut dst1 = vec![0.0; 6];
        t1.export(&mut dst1)?;
        assert_eq!(dst1, vec![
            0.7310586, 1.7615943, 2.8577223, 3.928055, 4.966536, 5.9851646
        ]);

        Ok(())
    }

    #[test]
    fn test_gelu_inplace() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = VulkanTensor::new(&v1, &[6], d.clone()).unwrap();
        let t1 = t1.gelu_inplace()?;

        let mut dst1 = vec![0.0; 6];
        t1.export(&mut dst1)?;
        assert_eq!(dst1, vec![
            0.4750867, 0.99373245, 1.4995073, 1.9999906, 2.5, 3.0
        ]);

        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = VulkanTensor::new(&v1, &[2, 3], d.clone()).unwrap();
        let t1 = t1.softmax_inplace(1)?;

        let mut dst1 = vec![0.0; 6];
        t1.export(&mut dst1)?;
        assert_eq!(dst1, vec![
            0.090030566,
            0.24472848,
            0.665241,
            0.090030566,
            0.24472848,
            0.665241
        ]);

        Ok(())
    }

    #[test]
    fn test_tensor_rms_norm() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());

        pub fn simple_rmsnorm(x: &mut [f32]) {
            let ss = x.iter().fold(0.0, |s, n| s + n * n);
            let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
            let scale = 1.0 / rms;
            // normalize and scale
            for i in x {
                *i *= scale;
            }
        }
        let v1 = (1..129).map(|i| i as f32).collect::<Vec<_>>();

        let t1 = VulkanTensor::new(&v1.clone(), &[128], d.clone())?;
        let t1 = t1.rms_norm_inplace(1e-5)?;
        let mut dst1 = vec![0.0; 128];
        t1.export(&mut dst1)?;

        let mut dst2 = v1.clone();
        simple_rmsnorm(&mut dst2);

        assert_relative_eq!(&dst1[0..10], &dst2[0..10], epsilon = 1e-4);
        Ok(())
    }

    #[test]
    fn test_rope() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let v1 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t1 = VulkanTensor::new(&v1, &[2, 16], d.clone())?;
        let t1 = t1.rope_inplace(RopeMode::Llama, 1, 2)?;

        let mut dst1 = vec![0.0; 32];
        t1.export(&mut dst1)?;

        assert_relative_eq!(
            &dst1[..],
            &[
                -0.841471, 0.54030234, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, -5.6601696, 22.648676, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0
            ][..],
            epsilon = 1e-5
        );

        Ok(())
    }
}
