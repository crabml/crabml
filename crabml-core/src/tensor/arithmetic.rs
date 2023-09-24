use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
///! arithmetic.rs contains the tensor arithmetics operations like matmul, accum, etc.
use crate::tensor::Tensor;

pub fn tensor_copy<'a>(dst: &mut Tensor<'a>, src: &Tensor<'a>) -> Result<()> {
    Ok(())
}

// W (w_rows,w_cols) @ x (w_cols,x_cols) -> xout (w_rows,x_cols)
// W (w_rows,w_cols) @ x (w_cols,) -> xout (w_rows,)
pub fn tensor_matmul_2d<'a>(out: &mut Tensor<'a>, w: &Tensor<'a>, x: &Tensor<'a>) -> Result<()> {
    require_tensor_dims(w, "w", &[2])?;
    require_tensor_dims(x, "x", &[1, 2])?;
    require_tensor_matmul_2d_shapes(w, x)?;
    require_tensor_contiguous(w, "w")?;
    require_tensor_contiguous(x, "x")?;
    require_tensor_contiguous(out, "out")?;

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

fn require_tensor_dims(t: &Tensor, tensor_name: &str, dims: &[usize]) -> Result<()> {
    if !dims.contains(&t.shape().len()) {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!(
                "tensor {} is available for {} dimension, but got {}",
                tensor_name,
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

fn require_tensor_contiguous(t: &Tensor, tensor_name: &str) -> Result<()> {
    if !t.is_contiguous() {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("{} need to be contiguous", tensor_name),
            cause: None,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        tensor_matmul_2d(&mut out, &w, &b)?;
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
        tensor_matmul_2d(&mut out, &w, &b)?;
        assert_eq!(
            out.flat(),
            &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
        );
        assert_eq!(out.shape(), vec![2, 4]);
        Ok(())
    }
}
