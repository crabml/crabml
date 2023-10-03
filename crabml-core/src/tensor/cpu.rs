use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use std::borrow::Cow;
use std::ops::Index;
use std::ops::IndexMut;
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

    pub fn iter_axis(&'a self, pos: Vec<usize>, axis: usize) -> Result<impl Iterator<Item = &'a f32>> {
        Ok(self.strider.iter_axis(pos, axis)?.map(|i| &self.buf[i] ))
    }

    pub fn iter_axis_mut(&'a mut self, pos: Vec<usize>, axis: usize) -> Result<CpuTensorAxisIterMut<'a>> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        let buf = match self.buf {
            Cow::Owned(ref mut buf) => buf,
            _ => unreachable!(),
        };

        let strider = self.strider.clone();
        let pos_iter = Box::new(strider.into_iter_axis(pos, axis)?);

        Ok(CpuTensorAxisIterMut {
            tail: Some(buf),
            pos: None,
            pos_iter: pos_iter,
        })
    }

    pub fn is_contiguous(&self) -> bool {
        self.strider.is_contiguous()
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }
}

pub struct CpuTensorAxisIterMut<'a> {
    tail: Option<&'a mut [f32]>,
    pos_iter: Box<dyn Iterator<Item=usize>>,
    pos: Option<usize>,
}

impl<'a> Iterator for CpuTensorAxisIterMut<'a> {
    type Item = &'a mut f32;

    fn next(&mut self) -> Option<Self::Item> {
        // on the first iter, seek to the first position
        if self.pos.is_none() {
            let pos = self.pos_iter.next()?;
            let tail = self.tail.take().unwrap();
            let (_, tail) = tail.split_at_mut(pos);
            self.tail = Some(tail);
            self.pos = Some(pos);
        }

        // pop the current position
        let tail = self.tail.take()?;
        let (n, tail) = tail.split_first_mut()?;

        // prepare the next position
        let pos = self.pos.unwrap();
        if let Some(next_pos) = self.pos_iter.next() {
            let (_, tail) = tail.split_at_mut(next_pos - pos - 1);
            self.tail = Some(tail);
            self.pos = Some(next_pos);
        } else {
            self.tail = None;
        }

        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_iter_axis() -> Result<()> {
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let r = t.iter_axis(vec![0, 0], 1)?.cloned().collect::<Vec<_>>();
        assert_eq!(r, vec![]);

        Ok(())
    }
}