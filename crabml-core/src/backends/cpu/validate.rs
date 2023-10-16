use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::backends::cpu::CpuTensor;

pub fn require_tensor_shape(t: &CpuTensor, shape: &[usize]) -> Result<()> {
    if !t.shape().eq(shape) {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("tensor shape is not {:?}, but {:?}", shape, t.shape(),),
            cause: None,
        });
    }
    Ok(())
}

pub fn require_tensor_owned(t: &CpuTensor) -> Result<()> {
    if !t.is_owned() {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: "not owned".into(),
            cause: None,
        });
    }
    Ok(())
}

pub fn require_tensor_dims(t: &CpuTensor, dims: &[usize]) -> Result<()> {
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

pub fn require_tensor_matmul_2d_shapes(t1: &CpuTensor, t2: &CpuTensor) -> Result<()> {
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

pub fn require_tensor_contiguous(t: &CpuTensor) -> Result<()> {
    if !t.is_contiguous() {
        return Err(Error {
            kind: ErrorKind::TensorError,
            message: format!("tensor need to be contiguous",),
            cause: None,
        });
    }

    Ok(())
}
