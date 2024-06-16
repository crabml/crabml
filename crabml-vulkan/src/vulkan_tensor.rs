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
use crate::push_constants::ConcatenatePushConstants;
use crate::push_constants::ContiguousPushConstants;
use crate::push_constants::MatmulPushConstants;
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
        let bytes_size = buf.len();
        let buf = device.inner.make_device_buffer_from(buf);
        let strider = TensorStrider::new(shape.to_vec());
        Ok(Self {
            buf,
            dtype: GGMLType::F32,
            capacity: bytes_size,
            strider,
            device,
            name: None,
        })
    }

    fn alloc(shape: &[usize], dtype: GGMLType, device: Self::DeviceRef) -> Result<Self> {
        assert!(dtype == GGMLType::F32, "wgpu tensor only support F32 yet");
        let strider = TensorStrider::new(shape.to_vec());
        let bytes_size = std::mem::size_of::<f32>() * strider.len();
        let buf = device.inner.make_device_buffer(bytes_size);
        Ok(Self {
            buf,
            dtype,
            capacity: bytes_size,
            strider,
            device,
            name: None,
        })
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
        assert!(self.strider.dims() == 3 || self.strider.dims() == 2);
        if self.strider.is_contiguous() {
            return Ok(self);
        }

        let n_elms = self.strider.len();
        let output = Self::alloc(self.strider.shape(), self.dtype, self.device.clone())?;
        let pcs = ContiguousPushConstants {
            n_dims: self.strider.dims() as u32,
            n_elms: n_elms as u32,
            shape: convert_u32_vec4(self.strider.shape()),
            strides: convert_u32_vec4(self.strider.strides()),
        };
        let bufs = vec![output.buf.clone(), self.buf.clone()];
        let dispatches = [n_elms as u32 / 32 + 1, 1, 1];
        self.device
            .inner
            .dispatch_compute("contiguous", bufs, pcs, dispatches);
        Ok(output)
    }

    fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    fn strider(&self) -> &TensorStrider {
        &self.strider
    }

    fn concatenate(&mut self, rhs: &Self, axis: usize) -> Result<()> {
        if self.shape().len() != 3 {
            return Err((
                ErrorKind::TensorError,
                "only support 3D tensor concatenation yet",
            )
                .into());
        }
        if self.dtype != GGMLType::F32 || rhs.dtype != GGMLType::F32 {
            return Err((ErrorKind::TensorError, "concatenate: only support f32 yet").into());
        }

        let pcs = ConcatenatePushConstants {
            shape1: convert_u32_vec4(self.strider.shape()),
            shape2: convert_u32_vec4(rhs.shape()),
            strides1: convert_u32_vec4(self.strider.strides()),
            strides2: convert_u32_vec4(rhs.strider.strides()),
            axis: axis as u32,
            dims: 3,
            n_elms: rhs.strider.len() as u32,
        };
        let dispatches = [rhs.strider.len() as u32 / 32 + 1, 1, 1];
        let bufs = vec![self.buf.clone(), rhs.buf.clone()];
        self.device
            .inner
            .dispatch_compute("concatenate", bufs, pcs, dispatches);

        let mut new_shape = self.strider.shape().to_vec();
        new_shape[axis] += rhs.strider.shape()[axis];
        self.strider = self.strider.resize(&new_shape)?;
        Ok(())
    }

    fn copy_rows_from(&mut self, src: &Self, src_rows: &[usize]) -> Result<()> {
        assert!(self.strider.is_contiguous());
        assert!(src.strider.is_contiguous());
        assert!(src.strider.dims() == 2);
        assert!(src.dtype == GGMLType::F32); // we need support quantized tensor here, live it as a TODO

        let row_dims = src.shape().last().unwrap();

        for (dst_row, src_row) in src_rows.iter().enumerate() {
            let dst_offset = dst_row * row_dims * std::mem::size_of::<f32>();
            let src_offset = src_row * row_dims * std::mem::size_of::<f32>();
            let row_bytes = row_dims * std::mem::size_of::<f32>();
            self.device.inner.copy_device_buffer(
                src.buf.clone(),
                src_offset,
                self.buf.clone(),
                dst_offset,
                row_bytes,
            );
        }

        Ok(())
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
        assert!(self.dtype == GGMLType::F32, "only support F32 yet");

        let new_tensor = Self::alloc(self.strider.shape(), self.dtype, self.device.clone())?;
        let bytes_size = std::mem::size_of::<f32>() * self.strider.len();
        self.device.inner.copy_device_buffer(
            self.buf.clone(),
            0,
            new_tensor.buf.clone(),
            0,
            bytes_size,
        );
        Ok(new_tensor)
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

    fn matmul_vec(&self, rhs: &Self) -> Result<Self> {
        assert!(self.shape().len() == 2);
        assert!(self.shape().last() == rhs.shape().last());
        assert!(self.strider.is_contiguous());
        assert!(rhs.strider.is_contiguous());

        let output = Self::alloc(
            &[rhs.strider.shape()[0], self.strider.shape()[0]],
            GGMLType::F32,
            self.device.clone(),
        )?;
        let pcs = MatmulPushConstants {
            b: rhs.strider.shape()[0] as u32,
            m: self.strider.shape()[0] as u32,
            k: self.strider.shape()[1] as u32,
        };
        let bufs = vec![self.buf.clone(), rhs.buf.clone(), output.buf.clone()];
        assert!(pcs.m / 32 < 65535); // vulkan limit each dimension to 65535
        let dispatches = [pcs.b, pcs.m / 32, 1];
        self.device
            .inner
            .dispatch_compute("sgemv", bufs, pcs, dispatches);
        Ok(output)
    }

    fn batch_matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }
}

fn convert_u32_vec4(v: &[usize]) -> [u32; 4] {
    let mut vec4 = [0_u32; 4];
    for i in 0..v.len() {
        vec4[i] = v[i] as u32;
    }
    vec4
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use crabml::error::Result;
    use crabml::gguf::GGMLType;
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

        assert_relative_eq!(
            &bufo[..],
            &vec![
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                33.0, 34.0, 35.0
            ][..]
        );
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

        assert_relative_eq!(
            &bufo[..],
            &vec![
                0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0,
                28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0,
                56.0, 58.0, 60.0, 62.0, 64.0, 66.0
            ][..]
        );
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
        assert_relative_eq!(
            &dst1[..],
            &vec![
                0.7310586, 1.7615943, 2.8577223, 3.928055, 4.966536, 5.9851646
            ][..],
            epsilon = 1e-4
        );

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
        assert_relative_eq!(
            &dst1[..],
            &vec![0.4750867, 0.99373245, 1.4995073, 1.9999906, 2.5, 3.0][..]
        );

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
        assert_relative_eq!(
            &dst1[..],
            &vec![
                0.090030566,
                0.24472848,
                0.665241,
                0.090030566,
                0.24472848,
                0.665241
            ][..],
            epsilon = 1e-4
        );

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

    #[test]
    fn test_dup() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let v1 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t1 = VulkanTensor::new(&v1, &[2, 16], d.clone())?;
        let t2 = t1.dup()?;

        let mut dst1 = vec![0.0; 32];
        t2.export(&mut dst1)?;

        assert_relative_eq!(
            &dst1[..],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
                29.0, 30.0, 31.0
            ][..],
            epsilon = 1e-5
        );
        Ok(())
    }

    #[test]
    fn test_copy_rows_from() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let v1 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t1 = VulkanTensor::new(&v1, &[2, 16], d.clone())?;
        let mut t2 = VulkanTensor::alloc(&[16], crabml::gguf::GGMLType::F32, d)?;
        t2.copy_rows_from(&t1, &[1])?;

        let mut dst1 = vec![0.0; 16];
        t2.export(&mut dst1)?;

        assert_relative_eq!(
            &dst1[..],
            &[
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0
            ][..],
            epsilon = 1e-5
        );
        Ok(())
    }

    #[test]
    fn test_contiguous() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        // 4, 5, 6
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = VulkanTensor::new(&v1, &[2, 3], d)?;
        let t1 = t1.transpose(&[1, 0])?;
        let t2 = t1.contiguous()?;
        // 1, 4
        // 2, 5
        // 3, 6

        let mut dst1 = vec![0.0; 6];
        t2.export(&mut dst1)?;

        assert_eq!(t2.strider.shape(), &[3, 2]);
        assert_eq!(t2.strider.dims(), 2);
        assert_relative_eq!(
            &dst1[..],
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0][..],
            epsilon = 1e-5
        );
        Ok(())
    }

    #[test]
    fn test_concatenate() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let mut t1 = VulkanTensor::alloc(&[2, 2, 16], GGMLType::F32, d.clone())?.resize(0, 0)?;

        let v2 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t2 = VulkanTensor::new(&v2, &[1, 2, 16], d.clone())?;

        let v3 = (32..64).map(|i| i as f32).collect::<Vec<_>>();
        let t3 = VulkanTensor::new(&v3, &[1, 2, 16], d.clone())?;

        t1.concatenate(&t2, 0)?;
        t1.concatenate(&t3, 0)?;

        let mut dst1 = vec![0.0; 64];
        t1.export(&mut dst1)?;

        assert_eq!(t1.shape(), &[2, 2, 16]);
        assert_relative_eq!(
            &dst1[..],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
                29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0,
                43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
                57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0
            ][..],
            epsilon = 1e-5
        );
        Ok(())
    }

    #[test]
    fn test_matmul_vec() -> Result<()> {
        let d = VulkanTensorDevice::new(VulkanTensorDeviceOptions::default());
        let v1 = (0..256).map(|i| i as f32).collect::<Vec<_>>();

        let t1 = VulkanTensor::new(&v1, &[32, 8], d.clone())?;
        let t2 = VulkanTensor::new(&[2.0; 8], &[8], d.clone())?;
        let t3 = t1.matmul_vec(&t2)?;
        let mut dst1 = vec![0.0; 32];
        t3.export(&mut dst1)?;
        assert_eq!(dst1[0..32], vec![
            56.0, 184.0, 312.0, 440.0, 568.0, 696.0, 824.0, 952.0, 1080.0, 1208.0, 1336.0, 1464.0,
            1592.0, 1720.0, 1848.0, 1976.0, 2104.0, 2232.0, 2360.0, 2488.0, 2616.0, 2744.0, 2872.0,
            3000.0, 3128.0, 3256.0, 3384.0, 3512.0, 3640.0, 3768.0, 3896.0, 4024.0
        ]);
        Ok(())
    }
}
