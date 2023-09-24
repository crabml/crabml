///! arithmetic.rs contains the tensor arithmetics operations like matmul, accum, etc.

use crate::tensor::Tensor;
use crate::error::Result;
use crate::error::Error;
use crate::error::ErrorKind;

// W (w_rows,w_cols) @ x (w_cols,x_cols) -> xout (w_rows,x_cols)
// W (w_rows,w_cols) @ x (w_cols,) -> xout (w_rows,)
pub fn matmul_2d<'a>(out: &mut Tensor<'a>, w: &Tensor<'a>, x: &Tensor<'a>) -> Result<()> {
    if w.shape().len() != 2 {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("arg w requires 2 dimension"),
            cause: None,
        })
    }
    if x.shape().len() != 1 && x.shape().len() != 2 {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("arg x requires 1 or 2 dimensions"),
            cause: None,
        })
    }

    let w_rows = w.shape()[0];
    let w_cols = w.shape()[1];
    let x_rows = x.shape()[0];
    let x_cols = if x.shape().len() == 2 {
        x.shape()[1] 
    } else {
        1
    };

    if w_cols != x_rows {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("matmul mismatched tensor shapes: {:?} @ {:?}", w.shape(), x.shape()),
            cause: None,
        })
    }
    if !(w.is_contiguous() && x.is_contiguous() && out.is_contiguous()) {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("all the parameters need to be contiguous"),
            cause: None,
        })
    }

    let obuf = out.flat_mut()?;
    let wbuf = w.flat();
    let xbuf = x.flat();

    for w_row in 0..w_rows {
        for x_col in 0..x_cols {
            for w_col in 0..w_cols {
                obuf[w_row * x_cols + x_col] += wbuf[w_row * w_cols + w_col] * xbuf[w_col * x_cols + x_col];
            }
        }
    }

   Ok(())
}