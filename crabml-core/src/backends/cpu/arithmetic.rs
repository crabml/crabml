use std::borrow::Cow;
use std::simd::f32x32;
use std::simd::f32x8;
use std::simd::prelude::SimdFloat;

use half::f16;
use rayon::prelude::*;

use super::buf::buf_q8_0::vec_dot_q8_0_f16;
use super::buf::QuantBufQ8_0;
use super::primitives;
use crate::backends::cpu::buf::BufVecDot;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::validate::require_tensor_contiguous;
use crate::backends::cpu::validate::require_tensor_dims;
use crate::backends::cpu::validate::require_tensor_matmul_2d_shapes;
use crate::backends::cpu::CpuTensor;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::TensorArithmetics;

/// ! arithmetic.rs contains the tensor arithmetics operations like matmul, accum, etc.

impl<'a, 'b> TensorArithmetics for CpuTensor<'a> {
    fn batch_matmul(&self, y: &CpuTensor<'a>) -> Result<Self> {
        do_batch_matmul(self, y)
    }

    // gemv
    // (m, k) @ (k, ) => (m, )
    fn matmul_vec(&self, x: &CpuTensor<'a>) -> Result<Self> {
        let bufa = self.buf();
        let bufb = x.buf();
        let mut c = CpuTensor::alloc(&[self.shape()[0]], None, x.device())?;
        let bufc = c.buf_mut();
        let strider1 = self.strider();
        let strider2 = x.strider();
        primitives::matmul_vec(bufa, bufb, bufc, strider1, strider2)?;
        Ok(c)
    }

    fn mul_inplace(mut self, rhs: &CpuTensor<'a>) -> Result<Self> {
        let strider1 = self.strider().clone();
        let strider2 = rhs.strider();
        primitives::mul_inplace(self.buf_mut(), rhs.buf(), &strider1, strider2)?;
        Ok(self)
    }

    fn add_inplace(mut self, b: &Self) -> Result<Self> {
        let strider1 = self.strider().clone();
        let strider2 = b.strider();
        primitives::add_inplace(self.buf_mut(), b.buf(), &strider1, strider2)?;
        Ok(self)
    }

    fn div_scalar_inplace(mut self, b: f32) -> Result<Self> {
        let rhs = CpuTensor::new(vec![b], &[1], self.device())?;
        let strider1 = self.strider().clone();
        let strider2 = rhs.strider();
        primitives::div_inplace(self.buf_mut(), rhs.buf(), &strider1, strider2)?;
        Ok(self)
    }

    fn silu_inplace(mut self) -> Result<Self> {
        primitives::silu_inplace(self.buf_mut())?;
        Ok(self)
    }

    fn softmax_inplace(mut self, axis: usize) -> Result<Self> {
        let strider1 = self.strider().clone();
        primitives::softmax_inplace(self.buf_mut(), strider1, axis)?;
        Ok(self)
    }

    fn rope_inplace(mut self, pos: usize, rope_dims: usize) -> Result<Self> {
        let strider1 = self.strider().clone();
        let buf1 = self.buf_mut();
        primitives::rope_inplace(buf1, &strider1, pos, rope_dims)?;
        Ok(self)
    }

    fn rms_norm_inplace(mut self, eps: f32) -> Result<Self> {
        let strider1 = self.strider().clone();
        let buf1 = self.buf_mut();
        primitives::rms_norm_inplace(buf1, &strider1, eps)?;
        Ok(self)
    }
}

pub fn do_batch_matmul<'a, 'b>(w: &CpuTensor<'a>, x: &CpuTensor<'b>) -> Result<CpuTensor<'b>>
where 'b: 'a {
    require_tensor_dims(w, &[3])?;
    require_tensor_dims(x, &[2])?;

    if w.shape()[0] != x.shape()[0] || w.shape()[2] != x.shape()[1] {
        return Err((
            ErrorKind::TensorError,
            format!(
                "mismatched tensor shapes on batch matmul: {:?} @ {:?}",
                w.shape(),
                x.shape()
            ),
        )
            .into());
    }

    // (batch_size, w_rows, w_cols) @ (batch_size, w_cols, ) -> (batch_size, w_rows, )
    let batch_size = w.shape()[0];
    let w_rows = w.shape()[1];
    let mut out = CpuTensor::alloc(&[batch_size, w_rows], None, x.device())?;
    for b in 0..batch_size {
        let o_iter = out.iter_axis_mut(vec![b, 0], 1)?; // w_cols
        o_iter.enumerate().for_each(|(w_row, o)| {
            let w_iter = w.iter_axis(&[b, w_row, 0], 2).unwrap(); // w_rows
            let x_iter = x.iter_axis(&[b, 0], 1).unwrap(); // w_rows
            *o = w_iter.zip(x_iter).map(|(w, x)| w * x).sum::<f32>();
        })
    }
    return Ok(out);
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::backends::cpu::cpu_tensor::CpuTensorDevice;
    use crate::tensor::TensorArithmetics;

    #[test]
    fn test_rms_norm() -> Result<()> {
        pub fn simple_rmsnorm(x: &mut [f32]) {
            let ss = x.iter().fold(0.0, |s, n| s + n * n);
            let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
            // normalize and scale
            for i in 0..x.len() {
                x[i] = x[i] / rms;
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

        let r1 = t1.rope_inplace(1, 2)?;
        let out = r1.iter().collect::<Vec<_>>();
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
        // 1, 2, 3
        // 4, 5, 6
        let device = CpuTensorDevice::new();
        let w = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device.clone())?;
        // 1
        // 2
        // 3
        let b = CpuTensor::new(vec![1.0, 2.0, 3.0], &[3], device.clone())?;
        // 0
        // 0
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18
        let out = w.matmul_vec(&b)?;
        assert_eq!(out.iter().collect::<Vec<_>>(), &[14.0, 32.0]);

        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<()> {
        let device = CpuTensorDevice::new();
        let t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device.clone())?;
        let t1 = t1.softmax_inplace(1)?;

        assert_eq!(t1.iter().collect::<Vec<_>>(), &[
            0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094
        ]);
        Ok(())
    }

    #[test]
    fn test_silu() -> Result<()> {
        let device = CpuTensorDevice::new();
        let t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], device.clone())?;
        let t1 = t1.silu_inplace()?;

        assert_eq!(t1.iter().collect::<Vec<_>>(), &[
            0.7310586, 1.761594, 2.8577225, 3.928055, 4.9665356, 5.9851646
        ]);
        Ok(())
    }
}
