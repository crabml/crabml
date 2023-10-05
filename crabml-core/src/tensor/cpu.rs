use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use rayon::prelude::*;
use std::borrow::Cow;
use std::ops::Index;
use std::ops::IndexMut;
use std::slice;
use std::slice::SliceIndex;

use super::strider::TensorStrider;
use super::tensor::Tensor;

#[derive(Debug, Clone, Default)]
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

    pub fn copy_from(&mut self, pos: &[usize], t: &CpuTensor<'a>) -> Result<()> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        let idx = self.strider.at(pos)?;
        let buf = &mut self.buf.to_mut()[idx..];

        buf.iter_mut().zip(t.iter()).for_each(|(a, b)| *a = *b);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn view(self, shape: &[usize]) -> Result<CpuTensor<'a>> {
        let strider = self.strider.view(shape.to_vec())?;
        Ok(Self {
            buf: self.buf,
            strider,
        })
    }

    pub fn view_ref<'b>(&'b self, shape: &[usize]) -> Result<CpuTensor<'a>>
    where
        'b: 'a,
    {
        let strider = self.strider.view(shape.to_vec())?;
        let buf = self.buf.as_ref();
        Ok(Self {
            buf: Cow::Borrowed(buf),
            strider,
        })
    }

    pub fn as_ref<'b>(&'b self) -> CpuTensor<'a>
    where
        'b: 'a,
    {
        Self {
            buf: Cow::Borrowed(self.buf.as_ref()),
            strider: self.strider.clone(),
        }
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

    pub fn iter_axis(&'a self, pos: &[usize], axis: usize) -> Result<CpuTensorAxisIter<'a, f32>> {
        if self.strider.is_contiguous_on_axis(axis) {
            if axis == self.shape().len() - 1 && pos[axis] == 0 {
                let start = self.strider.at(pos)?;
                let buf = &self.buf[start..start + self.strider.shape()[axis]];
                return Ok(CpuTensorAxisIter::Slice(buf.iter()));
            }

            let stride = self.strider.strides()[axis];
            let start = self.strider.at(pos)?;
            let buf = &self.buf[start..];
            return Ok(CpuTensorAxisIter::StepBy(buf.iter().step_by(stride)));
        }

        // slow path
        Ok(CpuTensorAxisIter::Boxed(Box::new(
            self.strider.iter_axis(pos, axis)?.map(|i| &self.buf[i]),
        )))
    }

    pub fn iter_axis_mut(
        &mut self,
        pos: Vec<usize>,
        axis: usize,
    ) -> Result<impl Iterator<Item = &mut f32>> {
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

        // on a contiguous tensor, if we move one position according to the axis, the step length must equals the stride
        let buf = &mut buf[self.strider.at(&pos)?..];
        let stride = self.strider.strides()[axis];
        let count = self.strider.shape()[axis] - pos[axis];

        // whenever you wanna make a IterMut, the builtin functions like split_at_mut / chunks_mut are your friend
        let iter = buf.chunks_mut(stride).take(count).map(|chunk| {
            let (n, _) = chunk.split_first_mut().unwrap();
            n
        });
        Ok(iter)
    }

    pub fn par_iter_axis_mut(
        &mut self,
        pos: Vec<usize>,
        axis: usize,
    ) -> Result<impl rayon::iter::IndexedParallelIterator<Item = &mut f32>> {
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

        // on a contiguous tensor, if we move one position according to the axis, the step length must equals the stride
        let buf = &mut buf[self.strider.at(&pos)?..];
        let stride = self.strider.strides()[axis];
        let count = self.strider.shape()[axis] - pos[axis];

        // whenever you wanna make a IterMut, the builtin functions like split_at_mut / chunks_mut are your friend
        let iter = buf.par_chunks_mut(stride).take(count).map(|chunk| {
            let (n, _) = chunk.split_first_mut().unwrap();
            n
        });
        Ok(iter)
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.strider.iter().map(|i| &self.buf[i])
    }

    pub fn par_iter(&self) -> Result<impl IndexedParallelIterator<Item = &f32>> {
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        Ok(self.buf.par_iter())
    }

    pub fn iter_mut(&mut self) -> Result<impl Iterator<Item = &mut f32>> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }
        Ok(self.buf.to_mut().iter_mut())
    }

    pub fn par_iter_mut(&mut self) -> Result<impl IndexedParallelIterator<Item = &mut f32>> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }
        Ok(self.buf.to_mut().par_iter_mut())
    }

    pub fn is_contiguous(&self) -> bool {
        self.strider.is_contiguous()
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    // only used on specialized performance critical cases
    pub fn buf(&self) -> &[f32] {
        &self.buf
    }

    // only used on specialized performance critical cases
    pub fn buf_mut(&mut self) -> Result<&mut [f32]> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        Ok(self.buf.to_mut())
    }
}

pub enum CpuTensorAxisIter<'a, T> {
    Slice(slice::Iter<'a, T>),
    StepBy(std::iter::StepBy<std::slice::Iter<'a, T>>),
    Boxed(Box<dyn Iterator<Item = &'a T> + 'a>),
}

impl<'a, T> Iterator for CpuTensorAxisIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CpuTensorAxisIter::Slice(iter) => iter.next(),
            CpuTensorAxisIter::StepBy(iter) => iter.next(),
            CpuTensorAxisIter::Boxed(iter) => iter.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view() -> Result<()> {
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let t = t.view(&[3, 2])?;

        let tr = t.view_ref(&[2, 3])?;
        assert_eq!(
            tr.iter().cloned().collect::<Vec<f32>>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn test_tensor_iter_axis() -> Result<()> {
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let r = t.iter_axis(&[0, 0], 1)?.cloned().collect::<Vec<_>>();
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
        let r = t.iter_axis(&[0, 0], 0)?.cloned().collect::<Vec<_>>();
        assert_eq!(r, vec![1.0, 4.0]);
        // 1, 2, 3
        // 4, 5, 6
        let r = t.iter_axis(&[0, 1], 0)?.cloned().collect::<Vec<_>>();
        assert_eq!(r, vec![2.0, 5.0]);

        let mut t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let r = t
            .iter_axis_mut(vec![0, 0], 1)?
            .map(|f| *f)
            .collect::<Vec<_>>();
        assert_eq!(r, vec![1.0, 2.0, 3.0]);

        let mut t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let r = t
            .iter_axis_mut(vec![0, 0], 0)?
            .map(|f| *f)
            .collect::<Vec<_>>();
        assert_eq!(r, vec![1.0, 4.0]);

        Ok(())
    }
}
