use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::Tensor;

///! arithmetic.rs contains the tensor arithmetics operations like matmul, accum, etc.

pub fn tensor_copy<'a>(dst: &mut Tensor<'a>, src: &Tensor<'a>) -> Result<()> {
    Ok(())
}

// x: (prompt_len, embed_len)
pub fn tensor_2d_rms_norm<'a>(out: &mut Tensor<'a>, xs: &Tensor<'a>, eps: f32) -> Result<()> {
    require_tensor_shape(out, xs.shape())?;
    require_tensor_contiguous(xs)?;
    require_tensor_contiguous(out)?;
    require_tensor_dims(xs, &[1, 2])?;

    let xs = if xs.shape().len() == 1 {
        xs.clone().view(&[1, xs.shape()[0]])?
    } else {
        xs.clone()
    };

    let out_buf = out.flat_mut()?;
    for (i, x) in xs.subtensors()?.iter().enumerate() {
        let x_buf = x.flat();
        let sum = x_buf.iter().fold(0.0, |s, n| s + n * n);
        let rms = ((sum / x.len() as f32) + eps).sqrt();
        for j in 0..x_buf.len() {
            out_buf[i * x_buf.len() + j] = x_buf[j] / rms;
        }
    }

    Ok(())
}

pub fn tensor_mul<'a>(out: &mut Tensor<'a>, a: &Tensor<'a>, b: &Tensor<'a>) -> Result<()> {
    require_tensor_shape(a, b.shape())?;
    require_tensor_shape(out, b.shape())?;

    let out_buf = out.flat_mut()?;
    let a_buf = a.flat();
    let b_buf = b.flat();

    for (i, (a, b)) in a_buf.iter().zip(b_buf.iter()).enumerate() {
        out_buf[i] = a * b;
    }
    Ok(())
}

// W (w_rows,w_cols) @ x (w_cols,x_cols) -> xout (w_rows,x_cols)
// W (w_rows,w_cols) @ x (w_cols,) -> xout (w_rows,)
pub fn tensor_2d_matmul<'a>(out: &mut Tensor<'a>, w: &Tensor<'a>, x: &Tensor<'a>) -> Result<()> {
    require_tensor_dims(w, &[2])?;
    require_tensor_dims(x, &[1, 2])?;
    require_tensor_matmul_2d_shapes(w, x)?;
    require_tensor_contiguous(w)?;
    require_tensor_contiguous(x)?;
    require_tensor_contiguous(out)?;

    let w_rows = w.shape()[0];
    let w_cols = w.shape()[1];
    let x_cols = if x.shape().len() == 2 {
        x.shape()[1]
    } else {
        1
    };

    let obuf = out.flat_mut()?;
    let wbuf = w.flat();
    let xbuf = x.flat();

    for w_row in 0..w_rows {
        for x_col in 0..x_cols {
            for w_col in 0..w_cols {
                obuf[w_row * x_cols + x_col] +=
                    wbuf[w_row * w_cols + w_col] * xbuf[w_col * x_cols + x_col];
            }
        }
    }

    Ok(())
}

// t: (rows, cols)
pub fn tensor_2d_softmax_inplace<'a>(t: &mut Tensor<'a>) -> Result<()> {
    require_tensor_dims(t, &[2])?;

    let rows = t.shape()[0];
    let cols = t.shape()[1];
    let buf = t.flat_mut()?;

    for row in 0..rows {
        let row_buf = &mut buf[row * cols..(row + 1) * cols];

        let max = row_buf.iter().fold(f32::NAN, |a, b| a.max(*b));
        let mut sum = 0.0;
        for val in row_buf.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }
        for val in row_buf.iter_mut() {
            *val /= sum;
        }
    }
    Ok(())
}

// q: (n_heads, head_size)
// k: (n_seq, n_kv_heads, head_size)
// v: (n_seq, n_hv_heads, head_size)
// attn: (n_heads, n_seq)
// out: (n_heads, head_size)
pub fn tensor_mha<'a>(out: &mut Tensor<'a>, attn: &mut Tensor<'a>, q: &Tensor<'a>, k_cache: &Tensor<'a>, v_cache: &Tensor<'a>, pos: usize) -> Result<()> {
    require_tensor_contiguous(q)?;
    require_tensor_contiguous(k_cache)?;
    require_tensor_contiguous(attn)?;
    require_tensor_dims(k_cache, &[3])?;

    let n_heads = q.shape()[0];
    let n_kv_heads = k_cache.shape()[1];
    let head_size = q.shape()[1];
    let attn_buf = attn.flat_mut()?;
    let n_seq = k_cache.shape()[0];

    // get attention scores
    for tok in 0..pos+1 {
        let k_tok = k_cache.subtensor(tok)?; // (n_kv_heads, head_size)
        for h in 0..n_heads {
            let kvh = h / (n_heads / n_kv_heads);
            let q_head = q.subtensor(h)?; // (head_size, )
            let k_head = k_tok.subtensor(kvh)?; // (head_size, )
            let q_buf = q_head.flat();
            let k_buf = k_head.flat();
            let score = (0..head_size).map(|i| q_buf[i] * k_buf[i]).sum::<f32>();
            attn_buf[h * n_seq + tok] = score / (head_size as f32).sqrt();
        }
    }
    tensor_2d_softmax_inplace(attn)?;

    todo!();

    Ok(())
}

// t: (n_heads, head_size)
pub fn tensor_rope_inplace<'a>(q: &mut Tensor<'a>, k: &mut Tensor<'a>, pos: usize, freq_base: f32, freq_scale: f32) -> Result<()> {
    require_tensor_contiguous(q)?;
    require_tensor_contiguous(k)?;
    require_tensor_dims(q, &[2])?;
    require_tensor_dims(k, &[2])?;

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
                q.flat_mut()?
            } else {
                k.flat_mut()?
            };
            let v0 = vec[i];
            let v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
    Ok(())
}

pub fn tensor_copy_chunk<'a>(out: &mut Tensor<'_>, n: usize, row: &Tensor<'a>) -> Result<()> {
    require_tensor_owned(out)?;
    require_tensor_contiguous(row)?;

    if n >= out.shape()[0] {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "tensor ~{} row {} is out of bounds",
                out.name().unwrap_or_default(),
                n,
            ),
            cause: None,
        });
    }

    let row_size = row.len();
    let target_buf = &mut out.flat_mut()?[row_size * n.. row_size * (n+1)];
    target_buf.copy_from_slice(row.flat());
    Ok(())
}

fn require_tensor_shape(t: &Tensor, shape: &[usize]) -> Result<()> {
    if !t.shape().eq(shape) {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "tensor ~{} shape is not {:?}, but {:?}",
                t.name().unwrap_or_default(),
                shape,
                t.shape(),
            ),
            cause: None,
        });
    }
    Ok(())
}

fn require_tensor_owned(t: &Tensor) -> Result<()> {
    if !t.is_owned() {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "tensor {} is not owned",
                t.name().unwrap_or_default(),
            ),
            cause: None,
        })
    }
    Ok(())
}

fn require_tensor_dims(t: &Tensor, dims: &[usize]) -> Result<()> {
    if !dims.contains(&t.shape().len()) {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "tensor ~{} is required for {} dimensions, but got {}",
                t.name().unwrap_or_default(),
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

fn require_tensor_matmul_2d_shapes(t1: &Tensor, t2: &Tensor) -> Result<()> {
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

fn require_tensor_contiguous(t: &Tensor) -> Result<()> {
    if !t.is_contiguous() {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "tensor ~{} need to be contiguous",
                t.name().unwrap_or_default()
            ),
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

        let mut out = Tensor::new(vec![0.0; 6], vec![6])?;
        let xs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6])?;
        tensor_2d_rms_norm(&mut out, &xs, 1e-5)?;
        assert_eq!(
            out.flat(),
            &[0.2567762, 0.5135524, 0.77032864, 1.0271049, 1.2838811, 1.5406573]
        );

        let mut out = Tensor::new(vec![0.0; 12], vec![2, 6])?;
        let xs = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2, 6],
        )?;
        tensor_2d_rms_norm(&mut out, &xs, 1e-5)?;
        assert_eq!(
            out.flat(),
            &[
                0.2567762, 0.5135524, 0.77032864, 1.0271049, 1.2838811, 1.5406573, 0.999995,
                0.999995, 0.999995, 0.999995, 0.999995, 0.999995
            ]
        );

        Ok(())
    }

    #[test]
    fn test_matmul_2d() -> Result<()> {
        // 1, 2, 3
        // 4, 5, 6
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        // 1
        // 2
        // 3
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        // 0
        // 0
        let mut out = Tensor::new(vec![0.0; 2], vec![2])?;
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18
        tensor_2d_matmul(&mut out, &w, &b)?;
        assert_eq!(out.flat(), &[14.0, 32.0]);

        // 1, 2, 3
        // 4, 5, 6
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        // 1, 2, 3
        // 4, 5, 6
        // 7, 8, 9
        // 10, 11, 12
        let b = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4],
        )?;
        let mut out = Tensor::new(vec![0.0; 8], vec![2, 4])?;
        tensor_2d_matmul(&mut out, &w, &b)?;
        assert_eq!(
            out.flat(),
            &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
        );
        assert_eq!(out.shape(), vec![2, 4]);
        Ok(())
    }
}
