use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::CpuTensor;
use rayon::prelude::*;

///! arithmetic.rs contains the tensor arithmetics operations like matmul, accum, etc.

pub fn rms_norm_inplace(mut x: CpuTensor<'_>, eps: f32) -> Result<CpuTensor<'_>> {
    require_tensor_contiguous(&x)?;
    require_tensor_dims(&x, &[1])?;

    let len = x.shape()[0];
    let sum = x.iter_axis(&[0], 0)?.fold(0.0, |s, n| s + n * n);
    let rms = ((sum / len as f32) + eps).sqrt();
    x.par_iter_axis_mut(vec![0], 0)?.for_each(|n| *n = *n / rms);
    Ok(x)
}

pub fn mul_inplace<'a>(mut a: CpuTensor<'a>, b: &CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    require_tensor_shape(&a, b.shape())?;

    for (ia, ib) in a.iter_mut()?.zip(b.iter()) {
        *ia *= ib;
    }
    Ok(a)
}

pub fn div_scalar_inplace<'a>(mut a: CpuTensor<'a>, b: f32) -> Result<CpuTensor<'a>> {
    a.par_iter_mut()?.for_each(|ia| {
        *ia /= b;
    });
    Ok(a)
}

pub fn add_inplace<'a>(mut a: CpuTensor<'a>, b: &CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    require_tensor_shape(&a, b.shape())?;
    require_tensor_contiguous(&a)?;
    require_tensor_contiguous(b)?;

    a.par_iter_mut()?.zip(b.par_iter()?).for_each(|(ia, ib)| {
        *ia += *ib;
    });
    Ok(a)
}

pub fn silu_inplace<'a>(mut x: CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    // for i in 0..buf.len() {
    //    buf[i] = buf[i] * (1.0 / (1.0 + (-buf[i]).exp()));
    // }
    x.par_iter_mut()?
        .for_each(|n| *n = *n / (1.0 + (-*n).exp()));
    Ok(x)
}

// W (w_rows,w_cols) @ x (w_cols,x_cols) -> xout (w_rows,x_cols)
// W (w_rows,w_cols) @ x (w_cols,) -> xout (w_rows,)
pub fn matmul<'a>(w: &CpuTensor<'a>, x: &CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    require_tensor_dims(w, &[2])?;
    require_tensor_dims(x, &[1, 2])?;
    require_tensor_matmul_2d_shapes(w, x)?;
    require_tensor_contiguous(w)?;

    if x.is_contiguous() && x.shape().len() == 1 {
        return matmul_specialized_2d_1d(w, x);
    }

    if x.shape().len() == 1 {
        let mut out = CpuTensor::zeros(vec![w.shape()[0]])?;
        let o_row_iter = out.par_iter_axis_mut(vec![0], 0)?; // (x_cols, )
        o_row_iter.enumerate().for_each(|(w_row, o)| {
            let w_row_iter = w.iter_axis(&[w_row, 0], 1).unwrap(); // (w_cols, )
            let x_col_iter = x.iter_axis(&[0], 0).unwrap(); // (w_cols, )
            *o = w_row_iter.zip(x_col_iter).map(|(w, x)| w * x).sum::<f32>();
        });
        return Ok(out);
    }

    let mut out = CpuTensor::zeros(vec![w.shape()[0], x.shape()[1]])?;
    let w_rows = w.shape()[0];
    for w_row in 0..w_rows {
        let o_row_iter = out.par_iter_axis_mut(vec![w_row, 0], 1)?; // (x_cols, )
        o_row_iter.enumerate().for_each(|(x_col, o)| {
            let w_row_iter = w.iter_axis(&[w_row, 0], 1).unwrap(); // (w_cols, )
            let x_col_iter = x.iter_axis(&[0, x_col], 0).unwrap(); // (w_cols, )
            *o = w_row_iter.zip(x_col_iter).map(|(w, x)| w * x).sum::<f32>();
        });
    }
    Ok(out)
}

pub fn matmul_specialized_2d_1d<'a>(w: &CpuTensor<'a>, x: &CpuTensor<'a>) -> Result<CpuTensor<'a>> {
    let mut xout = CpuTensor::zeros(vec![w.shape()[0]])?;
    let wb = w.buf();
    let xb = x.buf();
    let x_dim = x.len();

    xout.par_iter_mut()?.enumerate().for_each(|(w_row, xo)| {
        let wi = wb[w_row * x_dim..(w_row + 1) * x_dim].iter();
        let xi = xb.iter();
        *xo = wi.zip(xi).map(|(w, x)| w * x).sum::<f32>();
    });
    Ok(xout)
}

pub fn batch_matmul<'a, 'b>(w: &CpuTensor<'a>, x: &CpuTensor<'a>) -> Result<CpuTensor<'b>>
where
    'b: 'a,
{
    require_tensor_dims(w, &[3])?;
    require_tensor_dims(x, &[2, 3])?;

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

    if x.shape().len() == 2 {
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

    let batch_size = w.shape()[0];
    let w_rows = w.shape()[1];
    let x_cols = x.shape()[2];
    let mut out = CpuTensor::zeros(vec![batch_size, w_rows, x_cols])?;
    for b in 0..batch_size {
        for w_row in 0..w_rows {
            let o_row_iter = out.iter_axis_mut(vec![b, w_row, 0], 2)?; // (x_cols, )
            o_row_iter.enumerate().for_each(|(x_col, o)| {
                let w_row_iter = w.iter_axis(&[b, w_row, 0], 2).unwrap(); // (w_rows, )
                let x_col_iter = x.iter_axis(&[b, 0, x_col], 1).unwrap(); // (w_rows, )
                *o = w_row_iter.zip(x_col_iter).map(|(w, x)| w * x).sum::<f32>();
            });
        }
    }

    Ok(out)
}

// t: (rows, cols)
pub fn softmax_inplace<'a>(mut t: CpuTensor<'a>, axis: usize) -> Result<CpuTensor<'a>> {
    require_tensor_dims(&t, &[2])?;

    if axis != 1 {
        return Err((ErrorKind::TensorError, "only axis=1 is supported").into());
    }

    for row in 0..t.shape()[0] {
        let max = t.iter_axis(&[row, 0], 1)?.fold(f32::NAN, |a, b| a.max(*b));
        let sum = t.iter_axis_mut(vec![row, 0], 1)?.fold(0.0, |mut acc, val| {
            *val = (*val - max).exp();
            acc += *val;
            acc
        });
        t.par_iter_axis_mut(vec![row, 0], 1)?.for_each(|val| {
            *val /= sum;
        });
    }

    Ok(t)
}

// q: (n_heads, head_size)
// k_cache: (n_seq, n_kv_heads, head_size)
// v_cache: (n_seq, n_kv_heads, head_size)
// attn: (n_seq, )
// out: (n_heads, head_size)
pub fn multi_query_attention_specialized<'a>(
    q: &CpuTensor<'a>,
    k_cache: &CpuTensor<'a>,
    v_cache: &CpuTensor<'a>,
) -> Result<CpuTensor<'a>> {
    require_tensor_contiguous(q)?;
    require_tensor_contiguous(k_cache)?;
    require_tensor_dims(k_cache, &[3])?;

    let n_heads = q.shape()[0];
    let n_kv_heads = k_cache.shape()[1];
    let head_size = q.shape()[1];
    let n_seq = k_cache.shape()[0];

    let mut out = CpuTensor::zeros(vec![n_heads, head_size])?;
    let mut attn = CpuTensor::zeros(vec![n_seq])?;

    // get attention scores
    for h in 0..n_heads {
        let kvh = h / (n_heads / n_kv_heads);
        attn.par_iter_mut()?.enumerate().for_each(|(tok, attn)| {
            let q_head = q.iter_axis(&[h, 0], 1).unwrap(); // (head_size, )
            let k_head = k_cache.iter_axis(&[tok, kvh, 0], 2).unwrap(); // (head_size, )
            let score = q_head.zip(k_head).map(|(q, k)| q * k).sum::<f32>();
            *attn = score / (head_size as f32).sqrt();
        });

        attn = attn.view(&[1, n_seq])?; // (1, n_seq
        attn = softmax_inplace(attn, 1)?;

        let kvh = h / (n_heads / n_kv_heads);
        for (tok, attn) in attn.iter().enumerate() {
            let v_head = v_cache.iter_axis(&[tok, kvh, 0], 2)?; // (head_size, )
            let out_buf = out.iter_axis_mut(vec![h, 0], 1)?; // (head_size, )
            for (i, (o, v)) in out_buf.zip(v_head).enumerate() {
                *o += v * attn
            }
        }
    }

    Ok(out)
}

// q: (n_heads, head_size)
pub fn rope_inplace<'a>(
    mut q: CpuTensor<'a>,
    mut k: CpuTensor<'a>,
    pos: usize,
    freq_base: f32,
    freq_scale: f32,
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
            let vec = if v == 0 { q.buf_mut()? } else { k.buf_mut()? };
            let v0 = vec[i];
            let v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
    Ok((q, k))
}

fn require_tensor_shape(t: &CpuTensor, shape: &[usize]) -> Result<()> {
    if !t.shape().eq(shape) {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("tensor shape is not {:?}, but {:?}", shape, t.shape(),),
            cause: None,
        });
    }
    Ok(())
}

fn require_tensor_owned(t: &CpuTensor) -> Result<()> {
    if !t.is_owned() {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: "not owned".into(),
            cause: None,
        });
    }
    Ok(())
}

fn require_tensor_dims(t: &CpuTensor, dims: &[usize]) -> Result<()> {
    if !dims.contains(&t.shape().len()) {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "tensor is required for {} dimensions, but got {}",
                dims.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(" or "),
                t.shape().len(),
            ),
            cause: None,
        });
    }

    Ok(())
}

fn require_tensor_matmul_2d_shapes(t1: &CpuTensor, t2: &CpuTensor) -> Result<()> {
    if t1.shape()[1] != t2.shape()[0] {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "mismatched tensor shapes on matmul: {:?} @ {:?}",
                t1.shape(),
                t2.shape()
            ),
            cause: None,
        });
    }
    Ok(())
}

fn require_tensor_contiguous(t: &CpuTensor) -> Result<()> {
    if !t.is_contiguous() {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("tensor need to be contiguous",),
            cause: None,
        });
    }

    Ok(())
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
        assert_eq!(
            v,
            vec![0.2567762, 0.5135524, 0.77032864, 1.0271049, 1.2838811, 1.5406573]
        );
        let mut v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        simple_rmsnorm(&mut v);
        assert_eq!(
            v,
            vec![0.999995, 0.999995, 0.999995, 0.999995, 0.999995, 0.999995]
        );

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
        let out = matmul(&w, &b)?;
        assert_eq!(out.iter().cloned().collect::<Vec<_>>(), &[14.0, 32.0]);

        // 1, 2, 3
        // 4, 5, 6
        let w = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        // 1, 2, 3
        // 4, 5, 6
        // 7, 8, 9
        // 10, 11, 12
        let b = CpuTensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4],
        )?;
        let out = matmul(&w, &b)?;
        assert_eq!(
            out.iter().cloned().collect::<Vec<_>>(),
            &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
        );
        assert_eq!(out.shape(), vec![2, 4]);

        Ok(())
    }

    #[test]
    fn test_batch_matmul() -> Result<()> {
        let w = CpuTensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            vec![2, 2, 3],
        )?;
        let b = CpuTensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3, 1])?;

        let o = batch_matmul(&w, &b)?;
        assert_eq!(o.shape(), vec![2, 2, 1]);
        assert_eq!(
            o.iter().cloned().collect::<Vec<_>>(),
            vec![3.0, 12.0, 21.0, 30.0]
        );
        Ok(())
    }
}
