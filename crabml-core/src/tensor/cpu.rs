use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use std::borrow::Cow;
use std::slice;

use super::strider::TensorStrider;

#[derive(Debug, Clone)]
pub struct CpuTensor<'a> {
    buf: Cow<'a, [f32]>,
    strider: TensorStrider,
}

// A tensor contains a buffer of f32, a shape and a strides. We may refer to
// https://ajcr.net/stride-guide-part-1/ to learn more about how strides works.
// The buffer may be owned in a Vec or an ref to a part of shared memory. Any
// change on the tensor is considered as a move operation, to reduce the need on
// copying the owned buffer. Feel free to clone() the tensor.
impl<'a> CpuTensor<'a> {
    pub fn new(buf: impl Into<Cow<'a, [f32]>>, shape: Vec<usize>) -> Result<Self> {
        let buf = buf.into();
        if buf.len() != shape.iter().product() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: format!("invalid shape {:?} for data of length {}", shape, buf.len()),
                cause: None,
            });
        }

        let strider = TensorStrider::new(shape);

        Ok(Self { buf, strider })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        let buf = vec![0.0; shape.iter().product()];
        Self::new(buf, shape)
    }

    pub fn from_raw_bytes(buf: &'a [u8], shape: Vec<usize>) -> Result<Self> {
        let len = buf.len();
        assert_eq!(
            len % std::mem::size_of::<f32>(),
            0,
            "Length of slice must be multiple of f32 size"
        );
        let new_len = len / std::mem::size_of::<f32>();
        let ptr = buf.as_ptr() as *const f32;
        let f32_buf = unsafe { slice::from_raw_parts(ptr, new_len) };
        Self::new(f32_buf, shape)
    }

    pub fn at(&self, idx: &[usize]) -> Result<f32> {
        self.strider.at(idx).map(|offset| self.buf[offset])
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn view(self, shape: &[usize]) -> Result<Self> {
        let strider = self.strider.view(shape.to_vec())?;
        Ok(Self {
            buf: self.buf,
            strider,
        })
    }

    pub fn at_unchecked(&self, idx: &[usize]) -> f32 {
        let offset = self.strider.at_unchecked(idx);
        self.buf[offset]
    }

    pub fn is_owned(&self) -> bool {
        match self.buf {
            Cow::Owned(_) => true,
            _ => false,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.strider.is_contiguous()
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}