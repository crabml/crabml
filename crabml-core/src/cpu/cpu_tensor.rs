use std::borrow::Cow;

use crate::cpu::buf::buf_f16::alloc_f16_buf;
use crate::cpu::buf::CpuTensorBuf;
use crate::cpu::primitives;
use crate::cpu::CpuTensorDeviceRef;
use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::gguf::GGMLType;
use crate::tensor::RopeMode;
use crate::tensor::Tensor;
use crate::tensor::TensorStrider;

#[derive(Debug, Clone)]
pub struct CpuTensor<'a> {
    buf: CpuTensorBuf<'a>,
    strider: TensorStrider,
    device: CpuTensorDeviceRef<'a>,
    pub(crate) name: Option<String>,
}

// A tensor contains a buffer of f32, a shape and a strides. We may refer to
// https://ajcr.net/stride-guide-part-1/ to learn more about how strides works.
// The buffer may be owned in a Vec or an ref to a part of shared memory. Any
// change on the tensor is considered as a move operation, to reduce the need on
// copying the owned buffer. Feel free to clone() the tensor.
impl<'a> CpuTensor<'a> {
    pub fn new(buf: Vec<f32>, shape: &[usize], device: CpuTensorDeviceRef<'a>) -> Result<Self> {
        if buf.len() != shape.iter().product() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: format!("invalid shape {:?} for data of length {}", shape, buf.len()),
                cause: None,
            });
        }

        let strider = TensorStrider::new(shape.to_vec());
        Ok(Self {
            buf: buf.into(),
            strider,
            device: device.clone(),
            name: None,
        })
    }

    pub fn from_bytes(
        buf: &'a [u8],
        typ: GGMLType,
        shape: &[usize],
        device: CpuTensorDeviceRef<'a>,
    ) -> Result<Self> {
        let buf = CpuTensorBuf::from_raw_bytes(buf, typ)?;
        let strider = TensorStrider::new(shape.to_vec());
        Ok(Self {
            buf,
            strider,
            device: device.clone(),
            name: None,
        })
    }

    pub fn dequantize(self, dtype: GGMLType) -> Result<Self> {
        let _t = self.device.metrics.dequantize_walltime.track();
        let strider = self.strider.clone();
        let device = self.device.clone();
        let name = self.name.clone();
        let buf = self.buf.dequantize(dtype)?;
        Ok(Self {
            buf,
            strider,
            device,
            name,
        })
    }

    pub fn typ(&self) -> GGMLType {
        self.buf.dtype()
    }

    pub fn device(&self) -> CpuTensorDeviceRef<'a> {
        self.device.clone()
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strider.is_empty()
    }

    pub fn is_owned(&self) -> bool {
        self.buf.is_owned()
    }

    /// to_vec is only used for test.
    #[allow(dead_code)]
    fn to_vec(&self) -> Vec<f32> {
        assert!(self.dtype() == GGMLType::F32);
        if self.is_contiguous() {
            return self.buf.iter_f32().collect();
        }
        let buf = self.buf().as_f32_ref();
        self.strider.iter().map(|pos| buf[pos]).collect()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strider.is_contiguous()
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    pub fn buf(&self) -> &CpuTensorBuf<'a> {
        &self.buf
    }

    pub(crate) fn buf_mut(&mut self) -> &mut CpuTensorBuf<'a> {
        &mut self.buf
    }
}

impl<'a> Tensor for CpuTensor<'a> {
    type DeviceRef = CpuTensorDeviceRef<'a>;

    fn from_cpu(
        _buf: &[u8],
        _shape: &[usize],
        _dtype: GGMLType,
        _device: Self::DeviceRef,
    ) -> Result<Self> {
        unreachable!("no need to implement from_cpu for CpuTensor")
    }

    fn alloc(shape: &[usize], dtype: GGMLType, device: Self::DeviceRef) -> Result<Self> {
        if dtype != GGMLType::F32 && dtype != GGMLType::F16 {
            return Err((ErrorKind::TensorError, "only f32/f16 is supported").into());
        }

        let buf_size = shape.iter().product();
        let _t = device.metrics.alloc_walltime.track();
        let buf = match dtype {
            GGMLType::F32 => {
                let vec = Cow::Owned(vec![0.0; buf_size]);
                CpuTensorBuf::F32(vec)
            }
            GGMLType::F16 => {
                // it's slow to initialize a vec![f16::ZERO; buf_size], nearly 80~200ms on preparing kv cache
                let vec_f16 = alloc_f16_buf(buf_size);
                let vec = Cow::Owned(vec_f16);
                CpuTensorBuf::F16(vec)
            }
            _ => unreachable!(),
        };

        Ok(Self {
            buf,
            strider: TensorStrider::new(shape.to_vec()),
            device: device.clone(),
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
        if new_len > self.buf.len() {
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
            strider: new_strider,
            device: self.device.clone(),
            name: None,
        })
    }

    fn dtype(&self) -> GGMLType {
        self.buf.dtype()
    }

    fn reshape(self, shape: &[usize]) -> Result<Self> {
        let strider = self.strider.reshape(shape.to_vec())?;
        Ok(Self {
            buf: self.buf,
            strider,
            device: self.device.clone(),
            name: None,
        })
    }

    fn transpose(self, dims: &[usize]) -> Result<Self> {
        let strider = self.strider.transpose(dims)?;
        Ok(Self {
            buf: self.buf,
            strider,
            device: self.device.clone(),
            name: None,
        })
    }

    fn with_strider(self, strider: TensorStrider) -> Result<Self> {
        Ok(Self {
            buf: self.buf,
            strider,
            device: self.device.clone(),
            name: None,
        })
    }

    fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);

        // only used in test
        if self.device.opts.debug_named_tensors {
            self.device.add_debug_tensor(&self);
        }
        self
    }

    fn strider(&self) -> &TensorStrider {
        &self.strider
    }

    fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    fn concatenate(&mut self, rhs: &Self, axis: usize) -> Result<()> {
        let _t = self.device.metrics.concatenate_walltime.track();
        // (2, 1) + (2, 1) at axis 0 -> (4, 1)
        // (2, 1) + (2, 3) at axis 1 -> (2, 4)
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "tensor not owned on concatenate").into());
        }
        if self.dtype() != GGMLType::F32 && self.dtype() != GGMLType::F16 {
            return Err((
                ErrorKind::TensorError,
                "only f32/f16 is supported on concatenate",
            )
                .into());
        }
        if rhs.dtype() != GGMLType::F32 && rhs.dtype() != GGMLType::F16 {
            return Err((
                ErrorKind::TensorError,
                "only f32/f16 is supported on concatenate rhs",
            )
                .into());
        }

        // both tensors must have the same shape (except in the concatenating dimension)
        for i in 0..self.shape().len() {
            if i == axis {
                continue;
            }
            if self.shape()[i] != rhs.shape()[i] {
                return Err((
                    ErrorKind::TensorError,
                    format!(
                        "shape mismatch on concatenate, want {:?} but got {:?}",
                        self.shape(),
                        rhs.shape()
                    ),
                )
                    .into());
            }
        }

        let strider1 = self.strider().clone();
        let strider2 = rhs.strider();
        let new_strider =
            primitives::concatenate_inplace(self.buf_mut(), rhs.buf(), &strider1, strider2, axis)?;
        self.strider = new_strider;
        Ok(())
    }

    fn contiguous(self) -> Result<Self> {
        let _t = self.device.metrics.contiguous_walltime.track();
        if self.is_contiguous() {
            return Ok(self);
        }
        assert!(self.dtype() == GGMLType::F32 || self.dtype() == GGMLType::F16);

        let mut out = CpuTensor::alloc(self.shape(), self.dtype(), self.device())?;
        primitives::contiguous(&self.buf, &self.strider, &mut out.buf);
        Ok(out)
    }

    fn copy_rows_from(&mut self, src: &CpuTensor<'a>, src_rows: &[usize]) -> Result<()> {
        let _t = self.device.metrics.copy_from_walltime.track();
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "dst tensor is not contiguous").into());
        }
        if !src.is_contiguous() {
            return Err((ErrorKind::TensorError, "src tensor is not contiguous").into());
        }
        if src.strider.dims() != 2 && src.strider.dims() != 1 {
            return Err((
                ErrorKind::TensorError,
                "copy_rows_from: src tensor is not 2d or 1d",
            )
                .into());
        }

        let cols = *self.shape().last().unwrap();
        for (dst_row, src_row) in src_rows.iter().enumerate() {
            let src_offset = src_row * cols;
            let dst_offset = dst_row * cols;
            self.buf.copy_from(&src.buf, src_offset, dst_offset, cols)?;
        }
        Ok(())
    }

    fn dup(&self) -> Result<Self> {
        let _t = self.device.metrics.dup_walltime.track();
        let buf = self.buf.iter_f32().collect::<Vec<_>>();
        Self::new(buf, self.shape(), self.device.clone())
    }

    fn export(&self, dst: &mut [f32]) -> Result<()> {
        let _t = self.device.metrics.export_walltime.track();
        assert!(self.is_contiguous());

        dst.iter_mut()
            .zip(self.buf.iter_f32())
            .for_each(|(dst, src)| {
                *dst = src;
            });
        Ok(())
    }

    // (b, m, k) @ (b, k, n) -> (b, m, n)
    fn batch_matmul(&self, b: &CpuTensor<'a>) -> Result<Self> {
        let bufa = self.buf();
        let bufb = b.buf();
        let _t = self.device.metrics.batch_matmul_walltime.track();
        let mut c = CpuTensor::alloc(
            &[self.shape()[0], self.shape()[1], b.shape()[2]],
            GGMLType::F32,
            self.device(),
        )?;
        let bufc = c.buf_mut();
        let strider1 = self.strider();
        let strider2 = b.strider();
        primitives::batch_matmul(&self.device(), bufa, bufb, bufc, strider1, strider2);
        Ok(c)
    }

    // gemv
    // (m, k) @ (k, ) => (m, )
    // (m, k) @ (b, k) => (b, m, )
    fn matmul_vec(&self, x: &CpuTensor<'a>) -> Result<Self> {
        let bufa = self.buf();
        let bufb = x.buf();
        let shape_c = if x.shape().len() == 1 {
            vec![self.shape()[0]]
        } else {
            vec![x.shape()[0], self.shape()[0]]
        };
        let mut c = CpuTensor::alloc(&shape_c, GGMLType::F32, x.device())?;
        let bufc = c.buf_mut();
        let strider1 = self.strider();
        let strider2 = x.strider();
        // let _t = self.device.metrics.matmul_walltime.track();
        primitives::matmul_vec(&self.device, bufa, bufb, bufc, strider1, strider2);
        Ok(c)
    }

    fn mul_inplace(mut self, rhs: &CpuTensor<'a>) -> Result<Self> {
        let strider1 = self.strider().clone();
        let strider2 = rhs.strider();
        let _t = self.device.metrics.mul_walltime.track();
        primitives::mul_inplace(self.buf_mut(), rhs.buf(), &strider1, strider2)?;
        Ok(self)
    }

    fn add_inplace(mut self, b: &Self) -> Result<Self> {
        let strider1 = self.strider().clone();
        let strider2 = b.strider();
        let _t = self.device.metrics.add_walltime.track();
        primitives::add_inplace(self.buf_mut(), b.buf(), &strider1, strider2)?;
        Ok(self)
    }

    fn scale_inplace(mut self, rhs: f32) -> Result<Self> {
        let rhs = CpuTensor::new(vec![rhs], &[1], self.device())?;
        let strider1 = self.strider().clone();
        let strider2 = rhs.strider();
        primitives::mul_inplace(self.buf_mut(), rhs.buf(), &strider1, strider2)?;
        Ok(self)
    }

    fn silu_inplace(mut self) -> Result<Self> {
        let _t = self.device.metrics.activate_walltime.track();
        primitives::silu_inplace(self.device(), self.buf_mut())?;
        Ok(self)
    }

    fn gelu_inplace(mut self) -> Result<Self> {
        let _t = self.device.metrics.activate_walltime.track();
        primitives::gelu_inplace(self.device(), self.buf_mut())?;
        Ok(self)
    }

    fn softmax_inplace(mut self, axis: usize) -> Result<Self> {
        let _t = self.device.metrics.softmax_walltime.track();
        let strider1 = self.strider().clone();
        primitives::softmax_inplace(self.device(), self.buf_mut(), strider1, axis)?;
        Ok(self)
    }

    fn rope_inplace(mut self, mode: RopeMode, pos: usize, rope_dims: usize) -> Result<Self> {
        let _t = self.device.metrics.rope_walltime.track();
        let strider1 = self.strider().clone();
        let buf1 = self.buf_mut();
        primitives::rope_inplace(buf1, &strider1, mode, pos, rope_dims)?;
        Ok(self)
    }

    fn rms_norm_inplace(mut self, eps: f32) -> Result<Self> {
        let _t = self.device.metrics.rms_norm_walltime.track();
        let strider1 = self.strider().clone();
        let buf1 = self.buf_mut();
        primitives::rms_norm_inplace(buf1, &strider1, eps)?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::cpu::CpuTensorDevice;

    #[test]
    fn test_tensor_view() -> Result<()> {
        let device = CpuTensorDevice::new();
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device.clone())?;
        let t = t.reshape(&[3, 2])?;

        let tr = t.reshape(&[2, 3])?;
        assert_eq!(tr.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_copy_from() -> Result<()> {
        // 1 2
        // 3 4
        let device = CpuTensorDevice::new();
        let t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone())?;
        let mut t2 = CpuTensor::new(vec![0.0; 2], &[2], device.clone())?;

        t2.copy_rows_from(&t1, &[1])?;
        assert_eq!(t2.to_vec(), vec![3.0, 4.0]);

        t2.copy_rows_from(&t1, &[0])?;
        assert_eq!(t2.to_vec(), vec![1.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_rms_norm() -> Result<()> {
        pub fn simple_rmsnorm(x: &mut [f32]) {
            let ss = x.iter().fold(0.0, |s, n| s + n * n);
            let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
            // normalize and scale
            for i in x {
                *i /= rms;
            }
        }

        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        simple_rmsnorm(&mut v);
        assert_eq!(v, vec![
            0.2567762, 0.5135524, 0.77032864, 1.0271049, 1.2838811, 1.5406573
        ]);
        let mut v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        simple_rmsnorm(&mut v);
        assert_eq!(v, vec![
            0.999995, 0.999995, 0.999995, 0.999995, 0.999995, 0.999995
        ]);

        Ok(())
    }

    #[test]
    fn test_rope() -> Result<()> {
        let device = CpuTensorDevice::new();
        let v1 = (0..32).map(|v| v as f32).collect::<Vec<_>>();
        let t1 = CpuTensor::new(v1, &[2, 16], device.clone())?;

        let r1 = t1.rope_inplace(RopeMode::Llama, 1, 2)?;
        let out = r1.to_vec();
        assert_relative_eq!(
            &out[..],
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
    fn test_matmul() -> Result<()> {
        let device = CpuTensorDevice::new();
        let w = CpuTensor::new(vec![4.0; 32], &[16, 2], device.clone())?;
        let b = CpuTensor::new(vec![1.0, 2.0], &[2], device.clone())?;
        let out = w.matmul_vec(&b)?;
        assert_eq!(out.to_vec(), &[
            12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
            12.0, 12.0
        ]);

        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<()> {
        let device = CpuTensorDevice::new();
        let t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device.clone())?;
        let t1 = t1.softmax_inplace(1)?;

        assert_relative_eq!(
            &t1.to_vec()[..],
            &[
                0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094
            ][..],
            epsilon = 1e-3
        );
        Ok(())
    }

    #[test]
    fn test_silu() -> Result<()> {
        let device = CpuTensorDevice::new();
        let t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], device.clone())?;
        let t1 = t1.silu_inplace()?;

        assert_relative_eq!(
            &t1.to_vec()[..],
            &[
                0.7310586, 1.761594, 2.8577225, 3.928055, 4.9665356, 5.9851646
            ][..],
            epsilon = 1e-1
        );
        Ok(())
    }

    #[test]
    fn test_contigous() -> Result<()> {
        let device = CpuTensorDevice::new();
        let t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device.clone())?;
        let t1 = t1.transpose(&[1, 0])?; // 3 x 2
        let t2 = t1.contiguous()?;

        // 1 4
        // 2 5
        // 3 6
        assert_eq!(t2.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(t2.shape(), &[3, 2]);

        let t1 = CpuTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            device.clone(),
        )?;
        let t1 = t1.transpose(&[2, 1, 0])?; // 3 x 2 x 1
        let v1 = t1.to_vec();
        let t2 = t1.contiguous()?;

        // 1 4
        // 2 5
        // 3 6
        assert_eq!(t2.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(t2.shape(), &[3, 2, 1]);
        assert_eq!(v1, t2.to_vec());
        Ok(())
    }

    #[test]
    fn test_resize() -> Result<()> {
        // todo:
        Ok(())
    }
}
