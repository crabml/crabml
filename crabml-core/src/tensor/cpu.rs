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

    pub fn extend(&mut self, t: &CpuTensor<'a>) -> Result<()> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }
        if !t.shape().eq(&self.shape()[1..]) {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "shape mismatch on extend, want {:?} but got {:?}",
                    &self.shape()[1..],
                    &t.shape()
                ),
            )
                .into());
        }

        self.buf.to_mut().extend(t.iter());
        let new_shape = {
            let mut shape = self.shape().to_vec();
            shape[0] += 1;
            shape
        };
        self.strider = TensorStrider::new(new_shape);
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

    // view_ref is called on an owned Tensor, call view() if the tensor is mmapped.
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

    /// called on an owned Tensor, may used on MGQ where we have multiple query head on each key/value head
    pub fn repeat_ref<'b>(&'b self, repeats: &[usize]) -> Result<CpuTensor<'a>>
    where
        'b: 'a,
    {
        let strider = self.strider.repeat(repeats.to_vec())?;
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
        // speculize the fast path on iterating a contiguous memory buf
        if self.strider.is_contiguous_on_axis(axis) {
            if axis == self.shape().len() - 1 && pos[axis] == 0 {
                let start = self.strider.at(pos)?;
                let buf = &self.buf[start..start + self.strider.shape()[axis]];
                return Ok(CpuTensorAxisIter::Slice(buf.iter()));
            }
        }

        let stride = self.strider.strides()[axis];
        let start = self.strider.at(pos)?;

        // iterate the original buf, and repeat each element `repeats[axis]` times.
        // if this axis is repeated, the original buf of this axis is `repeats[axis]` times smaller than
        // the shape. e.g. shape = [2, 6], repeats = [1, 2], then the actual buf is [2, 3]
        if let Some(repeats) = self.strider.repeats() {
            let remains = (self.strider.shape()[axis] - pos[axis]) / repeats[axis] - 1;
            let buf = &self.buf[start..start + remains * stride + 1];
            if repeats[axis] == 1 {
                return Ok(CpuTensorAxisIter::StepBy(buf.iter().step_by(stride)));
            }
            let iter = buf
                .iter()
                .step_by(stride)
                .flat_map(move |n| std::iter::repeat(n).take(repeats[axis]));
            return Ok(CpuTensorAxisIter::Boxed(Box::new(iter)));
        }

        // normal case: to iterate arbitary axis, just step by the stride
        let remains = self.strider.shape()[axis] - pos[axis] - 1;
        let buf = &self.buf[start..start + remains * stride + 1];
        return Ok(CpuTensorAxisIter::StepBy(buf.iter().step_by(stride)));
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

// a enum dispatcher seems 3 times faster than a trait object on the benchmarks
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
        struct Test<'a> {
            tensor: &'a CpuTensor<'a>,
            input: (Vec<usize>, usize),
            want: Vec<f32>,
        };

        // 1, 2, 3
        // 4, 5, 6
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;

        let tests = vec![
            Test {
                tensor: &t,
                input: (vec![0, 0], 1),
                want: vec![1.0, 2.0, 3.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 0], 0),
                want: vec![1.0, 4.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 1], 0),
                want: vec![2.0, 5.0],
            },
        ];
        for tt in tests {
            let r = tt.tensor.iter_axis(&tt.input.0, tt.input.1)?.cloned().collect::<Vec<_>>();
            assert_eq!(r, tt.want);
        }

        // iter_axis with repeat
        // 1, 1, 2, 2, 3, 3
        // 4, 4, 5, 5, 6, 6
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let t = t.repeat_ref(&[1, 2])?;

        let tests = vec![
            Test {
                tensor: &t,
                input: (vec![0, 0], 1),
                want: vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 0], 0),
                want: vec![1.0, 4.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 1], 0),
                want: vec![1.0, 4.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 2], 0),
                want: vec![2.0, 5.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 3], 0),
                want: vec![2.0, 5.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 4], 0),
                want: vec![3.0, 6.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 5], 0),
                want: vec![3.0, 6.0],
            },
            Test {
                tensor: &t,
                input: (vec![1, 0], 1),
                want: vec![4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 0], 1),
                want: vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            }
        ];
        for tt in tests {
            let r = tt.tensor.iter_axis(&tt.input.0, tt.input.1)?.cloned().collect::<Vec<_>>();
            assert_eq!(r, tt.want);
        }

        Ok(())
    }

    #[test]
    fn test_tensor_iter_axis_mut() -> Result<()> {
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

    #[test]
    fn test_extend() -> Result<()> {
        let mut t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3])?;
        let t2 = CpuTensor::new(vec![1.0; 6], vec![2, 3])?;
        t1.extend(&t2)?;

        assert_eq!(t1.shape(), &[2, 2, 3]);
        assert_eq!(
            t1.buf(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
        Ok(())
    }
}
