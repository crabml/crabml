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
use crate::backends::cpu::validate::require_tensor_shape;
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

    // W (w_rows,w_cols) @ x (w_cols,x_cols) -> xout (w_rows,x_cols)
    // W (w_rows,w_cols) @ x (w_cols,) -> xout (w_rows,)
    fn matmul(&self, x: &CpuTensor<'a>) -> Result<Self> {
        let w = self;
        require_tensor_dims(w, &[2])?;
        require_tensor_dims(x, &[1])?;
        require_tensor_matmul_2d_shapes(w, x)?;
        require_tensor_contiguous(w)?;
        require_tensor_contiguous(x)?;

        match maybe_matmul_vec_2d_1d(w, x) {
            Some(r) => return r,
            _ => (),
        }

        let mut out = CpuTensor::alloc(&[w.shape()[0]], None, x.device())?;
        let o_row_iter = out.iter_axis_mut(vec![0], 0)?; // (x_cols, )
        o_row_iter.enumerate().for_each(|(w_row, o)| {
            let w_row_iter = w.iter_axis(&[w_row, 0], 1).unwrap(); // (w_cols, )
            let x_col_iter = x.iter_axis(&[0], 0).unwrap(); // (w_cols, )
            *o = w_row_iter.zip(x_col_iter).map(|(w, x)| w * x).sum::<f32>();
        });
        return Ok(out);
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

    fn silu_inplace(self) -> Result<Self> {
        let mut x = self;
        x.iter_mut()?.for_each(|n| *n = *n / (1.0 + (-*n).exp()));
        Ok(x)
    }

    fn softmax_inplace(self, axis: usize) -> Result<Self> {
        let mut t = self;
        require_tensor_dims(&t, &[2])?;

        if axis != 1 {
            return Err((ErrorKind::TensorError, "only axis=1 is supported").into());
        }

        for row in 0..t.shape()[0] {
            let max = t.iter_axis(&[row, 0], 1)?.fold(f32::NAN, |a, b| a.max(b));
            let sum = t.iter_axis_mut(vec![row, 0], 1)?.fold(0.0, |mut acc, val| {
                *val = (*val - max).exp();
                acc += *val;
                acc
            });
            t.iter_axis_mut(vec![row, 0], 1)?.for_each(|val| {
                *val /= sum;
            });
        }

        Ok(t)
    }

    fn rope_inplace(mut self, pos: usize, rope_dims: usize) -> Result<Self> {
        let strider1 = self.strider().clone();
        let buf1 = self.buf_mut();
        primitives::rope_inplace(buf1, &strider1, pos, rope_dims)?;
        Ok(self)
    }

    fn rms_norm_inplace(mut self, eps: f32) -> Result<Self> {
        require_tensor_contiguous(&self)?;
        require_tensor_dims(&self, &[1])?;

        match self.buf_mut() {
            CpuTensorBuf::F32(Cow::Owned(xb)) => {
                rms_norm_inplace_vec_f32(xb, eps);
                return Ok(self);
            }
            _ => (),
        }

        let len = self.shape()[0];
        let sum = self.iter_axis(&[0], 0)?.fold(0.0, |s, n| s + n * n);
        let rms = ((sum / len as f32) + eps).sqrt();
        self.iter_axis_mut(vec![0], 0)?.for_each(|n| *n = *n / rms);
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

fn rms_norm_inplace_vec_f32(x: &mut [f32], eps: f32) {
    let len = x.len();
    assert!(len % 32 == 0);
    let mut sum = 0.0;
    for chunk in x.as_chunks::<32>().0 {
        let mut v = f32x32::from_slice(chunk);
        v *= v;
        sum += v.reduce_sum();
    }
    let rms = ((sum / len as f32) + eps).sqrt();
    for chunk in x.as_chunks_mut::<32>().0 {
        let mut v = f32x32::from_slice(chunk);
        v /= f32x32::splat(rms);
        v.copy_to_slice(chunk);
    }
}

pub fn add_inplace_vec_f32(a: &mut [f32], b: &[f32]) {
    let acs = a.as_chunks_mut::<32>().0;
    let bcs = b.as_chunks::<32>().0;
    acs.iter_mut().zip(bcs.iter()).for_each(|(ac, bc)| {
        let mut va = f32x32::from_slice(ac);
        let vb = f32x32::from_slice(bc);
        va += vb;
        va.copy_to_slice(ac);
    });
}

pub fn silu_inplace_vec_f32(buf: &mut [f32]) {
    let chunks = buf.as_chunks_mut::<8>().0;
    chunks.iter_mut().for_each(|chunk| {
        let v0 = f32x8::from_slice(chunk);
        let v1 = f32x8::from_array([
            (-chunk[0]).exp(),
            (-chunk[1]).exp(),
            (-chunk[2]).exp(),
            (-chunk[3]).exp(),
            (-chunk[4]).exp(),
            (-chunk[5]).exp(),
            (-chunk[6]).exp(),
            (-chunk[7]).exp(),
        ]);
        let v2 = v1 + f32x8::splat(1.0);
        let v3 = v0 / v2;
        v3.copy_to_slice(chunk);
    })
}

pub fn maybe_matmul_vec_2d_1d<'a, 'b: 'a>(
    w: &CpuTensor<'a>,
    x: &CpuTensor<'b>,
) -> Option<Result<CpuTensor<'b>>> {
    if !(w.is_contiguous() && x.is_contiguous()) {
        return None;
    }
    let mut out: Vec<f32> = vec![0.0; w.shape()[0]];

    match (w.buf(), x.buf()) {
        (CpuTensorBuf::Q8_0(wb), CpuTensorBuf::F32(xb)) => {
            matmul_vec_q8_0_f32_2d_1d(wb, xb, &mut out)
        }
        (CpuTensorBuf::F32(wb), CpuTensorBuf::F32(xb)) => {
            if w.len() % 32 != 0 {
                return None;
            }
            matmul_vec_generic_xxx_f32_2d_1d(wb, xb, &mut out)
        }
        _ => return None,
    };

    Some(CpuTensor::new(out, &[w.shape()[0]], x.device()))
}

pub fn matmul_vec_generic_xxx_f32_2d_1d<'a, T: BufVecDot + Sync>(
    wb: &T,
    xb: &[f32],
    out: &mut [f32],
) {
    // wb: [w_rows, w_cols]
    // xb: [w_cols]
    // out: [w_rows]
    let w_cols = xb.len();
    out.par_iter_mut().enumerate().for_each(|(w_row, o)| {
        let offset = w_row * w_cols;
        *o = wb.vec_dot_f32(offset, xb);
    });
}

pub fn matmul_vec_q8_0_f32_2d_1d<'a>(wb: &QuantBufQ8_0<'a>, xb: &[f32], out: &mut [f32]) {
    // wb: [w_rows, w_cols]
    // xb: [w_cols]
    // out: [w_rows]
    let w_cols = xb.len();
    let xb16 = xb.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>();
    let xb_chunk_size: usize = 32;
    let xb_chunks: &[[f16; 32]] = xb16.as_chunks().0; // to keep it in L1 cache
    assert!(
        xb16.len() % xb_chunk_size == 0,
        "xb16.len() need to be a multiple of {}, but got {}",
        xb_chunk_size,
        xb16.len()
    );
    let out_chunk_size = out.len() / 32;
    out.par_chunks_mut(out_chunk_size)
        .enumerate()
        .for_each(move |(o_chunk_idx, o_chunk)| {
            for (oi, o) in o_chunk.iter_mut().enumerate() {
                let w_row = o_chunk_idx * out_chunk_size + oi;
                for (xb_chunk_idx, xb_chunk) in xb_chunks.iter().enumerate() {
                    let w_offset = w_row * w_cols + xb_chunk_size * xb_chunk_idx;
                    let wbq = wb.blocks_range(w_offset, w_offset + xb_chunk_size);
                    *o += vec_dot_q8_0_f16(wbq, xb_chunk);
                }
            }
        });
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
        let out = w.matmul(&b)?;
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
