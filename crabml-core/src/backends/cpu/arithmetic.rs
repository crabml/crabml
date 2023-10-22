use std::borrow::Cow;
use std::simd::f32x32;
use std::simd::f32x8;
use std::simd::SimdFloat;

use half::f16;
use rayon::prelude::*;

use super::buf::buf_q8_0::vec_dot_q8_0_f16;
use super::buf::QuantBufQ8_0;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::backends::cpu::buf::BufVecDot;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::validate::require_tensor_contiguous;
use crate::backends::cpu::validate::require_tensor_dims;
use crate::backends::cpu::validate::require_tensor_matmul_2d_shapes;
use crate::backends::cpu::validate::require_tensor_shape;
use crate::backends::cpu::CpuTensor;

/// ! arithmetic.rs contains the tensor arithmetics operations like matmul, accum, etc.

pub fn rms_norm_inplace(mut x: CpuTensor<'_>, eps: f32) -> Result<CpuTensor<'_>> {
    require_tensor_contiguous(&x)?;
    require_tensor_dims(&x, &[1])?;

    match x.buf_mut() {
        CpuTensorBuf::F32(Cow::Owned(xb)) => {
            rms_norm_inplace_vec_f32(xb, eps);
            return Ok(x);
        }
        _ => (),
    }

    let len = x.shape()[0];
    let sum = x.iter_axis(&[0], 0)?.fold(0.0, |s, n| s + n * n);
    let rms = ((sum / len as f32) + eps).sqrt();
    x.iter_axis_mut(vec![0], 0)?.for_each(|n| *n = *n / rms);
    Ok(x)
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

pub fn mul_inplace<'a>(mut a: CpuTensor<'a>, b: &CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    require_tensor_shape(&a, b.shape())?;

    if a.is_contiguous() && b.is_contiguous() {
        match (a.buf_mut(), b.buf()) {
            (CpuTensorBuf::F32(Cow::Owned(ab)), CpuTensorBuf::F32(bb)) => {
                mul_inplace_vec_f32(ab, bb);
                return Ok(a);
            }
            _ => (),
        }
    }

    for (ia, ib) in a.iter_mut()?.zip(b.iter()) {
        *ia *= ib;
    }
    Ok(a)
}

fn mul_inplace_vec_f32(a: &mut [f32], b: &[f32]) {
    let ac = a.as_chunks_mut::<32>().0;
    let bc = b.as_chunks::<32>().0;
    ac.iter_mut().zip(bc).for_each(|(a, b)| {
        let mut va = f32x32::from_slice(a);
        let vb = f32x32::from_slice(b);
        va *= vb;
        va.copy_to_slice(a);
    });
}

pub fn div_scalar_inplace<'a>(mut a: CpuTensor<'a>, b: f32) -> Result<CpuTensor<'a>> {
    a.iter_mut()?.for_each(|ia| {
        *ia /= b;
    });
    Ok(a)
}

pub fn add_inplace<'a>(mut a: CpuTensor<'a>, b: &CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    require_tensor_shape(&a, b.shape())?;
    require_tensor_contiguous(&a)?;
    require_tensor_contiguous(b)?;

    a.iter_mut()?.zip(b.iter()).for_each(|(ia, ib)| {
        *ia += ib;
    });
    Ok(a)
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

pub fn silu_inplace<'a>(mut x: CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    // for i in 0..buf.len() {
    //    buf[i] = buf[i] * (1.0 / (1.0 + (-buf[i]).exp()));
    // }
    if x.is_contiguous() {
        if let CpuTensorBuf::F32(Cow::Owned(xb)) = x.buf_mut() {
            silu_inplace_vec_f32(xb);
            return Ok(x);
        }
    }
    x.iter_mut()?.for_each(|n| *n = *n / (1.0 + (-*n).exp()));
    Ok(x)
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

// W (w_rows,w_cols) @ x (w_cols,x_cols) -> xout (w_rows,x_cols)
// W (w_rows,w_cols) @ x (w_cols,) -> xout (w_rows,)
pub fn matmul_2d_1d<'a>(w: &CpuTensor<'a>, x: &CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    require_tensor_dims(w, &[2])?;
    require_tensor_dims(x, &[1])?;
    require_tensor_matmul_2d_shapes(w, x)?;
    require_tensor_contiguous(w)?;
    require_tensor_contiguous(x)?;

    match maybe_matmul_vec_2d_1d(w, x) {
        Some(r) => return r,
        _ => (),
    }

    let mut out = CpuTensor::zeros(vec![w.shape()[0]])?;
    let o_row_iter = out.iter_axis_mut(vec![0], 0)?; // (x_cols, )
    o_row_iter.enumerate().for_each(|(w_row, o)| {
        let w_row_iter = w.iter_axis(&[w_row, 0], 1).unwrap(); // (w_cols, )
        let x_col_iter = x.iter_axis(&[0], 0).unwrap(); // (w_cols, )
        *o = w_row_iter.zip(x_col_iter).map(|(w, x)| w * x).sum::<f32>();
    });
    return Ok(out);
}

pub fn maybe_matmul_vec_2d_1d<'a>(
    w: &CpuTensor<'a>,
    x: &CpuTensor<'a>,
) -> Option<Result<CpuTensor<'a>>> {
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

    Some(CpuTensor::new(out, vec![w.shape()[0]]))
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

pub fn batch_matmul<'a, 'b>(w: &CpuTensor<'a>, x: &CpuTensor<'a>) -> Result<CpuTensor<'b>>
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
    let mut out = CpuTensor::zeros(vec![batch_size, w_rows])?;
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

// t: (rows, cols)
pub fn softmax_inplace<'a>(mut t: CpuTensor<'a>, axis: usize) -> Result<CpuTensor<'a>> {
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

pub fn rope_inplace<'a>(
    mut q: CpuTensor<'a>,
    pos: usize,
    rope_dims: usize,
) -> Result<CpuTensor<'a>> {
    require_tensor_contiguous(&q)?;
    require_tensor_dims(&q, &[2])?;

    let n_heads = q.shape()[0];
    let head_size = q.shape()[1];
    let qb = q.f32_buf_mut()?;

    // apply RoPE rotation for each head
    for h in 0..n_heads {
        for i in 0..rope_dims / 2 {
            let theta_scale = 10000_f32.powf(-2.0 * i as f32 / head_size as f32);
            let theta = pos as f32 * theta_scale;

            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let qp = &mut qb[h * head_size + i * 2..];
            let qp0 = qp[0];
            let qp1 = qp[1];
            qp[0] = qp0 * cos_theta - qp1 * sin_theta;
            qp[1] = qp0 * sin_theta + qp1 * cos_theta;
        }
    }

    Ok(q)
}

// q: (n_heads, head_size)
pub fn rope_inplace_old<'a>(
    mut q: CpuTensor<'a>,
    mut k: CpuTensor<'a>,
    pos: usize,
    freq_base: f32,
    freq_scale: f32,
    _n_rot: usize,
) -> Result<(CpuTensor<'a>, CpuTensor<'a>)> {
    require_tensor_contiguous(&q)?;
    require_tensor_contiguous(&k)?;
    require_tensor_dims(&q, &[2])?;
    require_tensor_dims(&k, &[2])?;

    let kv_dim: usize = k.shape().iter().product();
    let head_size = q.shape()[1];

    for i in (0..kv_dim).step_by(2) {
        let head_dim = i % head_size;
        let freq = freq_base / freq_scale.powf(head_dim as f32 / head_size as f32);
        let val = pos as f32 * freq;
        let fcr = val.cos();
        let fci = val.sin();
        let rotn = if i < kv_dim { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
        for v in 0..rotn {
            let vec = if v == 0 {
                q.f32_buf_mut()?
            } else {
                k.f32_buf_mut()?
            };
            let v0 = vec[i];
            let v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
    Ok((q, k))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_matmul() -> Result<()> {
        // 1, 2, 3
        // 4, 5, 6
        let w = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        // 1
        // 2
        // 3
        let b = CpuTensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        // 0
        // 0
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18
        let out = matmul_2d_1d(&w, &b)?;
        assert_eq!(out.iter().collect::<Vec<_>>(), &[14.0, 32.0]);

        Ok(())
    }
}