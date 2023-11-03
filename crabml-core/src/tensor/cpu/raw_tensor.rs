use std::rc::Rc;

use super::buf::CpuTensorBufIter;
use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::gguf::GGMLType;
use crate::tensor::cpu::buf::CpuTensorBuf;
use crate::tensor::strider::TensorStrider;
use crate::tensor::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct CpuTensor<'a> {
    buf: CpuTensorBuf<'a>,
    strider: TensorStrider,
    pool: CpuTensorPoolRef<'a>,
}

// A tensor contains a buffer of f32, a shape and a strides. We may refer to
// https://ajcr.net/stride-guide-part-1/ to learn more about how strides works.
// The buffer may be owned in a Vec or an ref to a part of shared memory. Any
// change on the tensor is considered as a move operation, to reduce the need on
// copying the owned buffer. Feel free to clone() the tensor.
impl<'a> CpuTensor<'a> {
    pub fn typ(&self) -> GGMLType {
        self.buf.typ()
    }

    pub fn from_bytes(
        buf: &'a [u8],
        typ: GGMLType,
        shape: &[usize],
        pool: CpuTensorPoolRef<'a>,
    ) -> Result<Self> {
        let buf = CpuTensorBuf::from_raw_bytes(buf, typ)?;
        let strider = TensorStrider::new(shape.to_vec());
        Ok(Self {
            buf,
            strider,
            pool: pool.clone(),
        })
    }

    pub fn pool(&self) -> CpuTensorPoolRef<'a> {
        self.pool.clone()
    }

    pub(crate) fn as_ref_inner<'b>(&'b self) -> CpuTensor<'a>
    where 'b: 'a {
        Self {
            buf: self.buf.as_ref(),
            strider: self.strider.clone(),
            pool: self.pool.clone(),
        }
    }

    pub fn at(&self, idx: &[usize]) -> Result<f32> {
        self.strider
            .at(idx)
            .map(|offset| self.buf.at_unchecked(offset))
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn at_unchecked(&self, idx: &[usize]) -> f32 {
        let offset = self.strider.at_unchecked(idx);
        self.buf.at_unchecked(offset)
    }

    pub fn is_owned(&self) -> bool {
        self.buf.is_owned()
    }

    pub fn iter_axis(&'a self, pos: &[usize], axis: usize) -> Result<CpuTensorBufIter> {
        // speculize the fast path on iterating a contiguous memory buf
        if self.strider.is_contiguous_on_axis(axis) {
            if axis == self.shape().len() - 1 && pos[axis] == 0 {
                let start = self.strider.at(pos)?;
                let end = start + self.strider.shape()[axis];
                return Ok(self.buf.iter_range(start, end, 1));
            }
        }

        let stride = self.strider.strides()[axis];
        let start = self.strider.at(pos)?;

        // iterate the original buf, and repeat each element `repeats[axis]` times.
        // if this axis is repeated, the original buf of this axis is `repeats[axis]` times smaller than
        // the shape. e.g. shape = [2, 6], repeats = [1, 2], then the actual buf is [2, 3]
        let axis_repeats = self
            .strider
            .repeats()
            .map(|repeats| repeats[axis])
            .unwrap_or(1);
        let remains = (self.strider.shape()[axis] - pos[axis]) / axis_repeats - 1;
        let end = start + remains * stride + 1;
        if axis_repeats == 1 {
            let iter = self.buf.iter_range(start, end, stride);
            return Ok(iter);
        }
        let iter = self.buf.iter_range(start, end, stride);
        let iter = iter.flat_map(move |n| std::iter::repeat(n).take(axis_repeats));
        return Ok(CpuTensorBufIter::Boxed(
            Box::new(iter),
            2 + remains * axis_repeats,
        ));
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

        // on a contiguous tensor, if we move one position according to the axis, the step length must equals the stride
        let start = self.strider.at(&pos)?;
        let remains = self.strider.shape()[axis] - pos[axis] - 1;
        let stride = self.strider.strides()[axis];
        let end = start + remains * stride + 1;

        let iter = self.buf.iter_range_mut(start, end, stride);
        Ok(iter)
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        if self.is_contiguous() {
            return self.buf.iter();
        }
        let iter = self.strider.iter().map(|i| self.buf.at_unchecked(i));
        CpuTensorBufIter::Boxed(Box::new(iter), self.len())
    }

    pub fn iter_from(&self, pos: &[usize]) -> Result<impl Iterator<Item = f32> + '_> {
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        let start = self.strider.at(pos).unwrap();
        let iter = (start..self.strider.len()).map(|i| self.buf.at_unchecked(i));
        Ok(CpuTensorBufIter::Boxed(Box::new(iter), self.len()))
    }

    pub fn iter_mut(&mut self) -> Result<impl Iterator<Item = &mut f32>> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }
        Ok(self.buf.iter_mut())
    }

    pub fn is_contiguous(&self) -> bool {
        self.strider.is_contiguous()
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    // only used on specialized performance critical cases
    pub(crate) fn buf(&self) -> &CpuTensorBuf<'a> {
        &self.buf
    }

    pub(crate) fn buf_mut(&mut self) -> &mut CpuTensorBuf<'a> {
        &mut self.buf
    }

    // TODO: only used in rope, remoe it later
    pub(crate) fn f32_buf_mut(&mut self) -> Result<&mut [f32]> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        Ok(self.buf.buf_mut())
    }
}

#[derive(Debug)]
pub struct CpuTensorPool<'a> {
    _bufs: Vec<CpuTensorBuf<'a>>,
}

pub type CpuTensorPoolRef<'a> = Rc<CpuTensorPool<'a>>;

impl<'a> CpuTensorPool<'a> {
    pub fn new() -> CpuTensorPoolRef<'a> {
        let pool = Self { _bufs: vec![] };
        Rc::new(pool)
    }

    pub fn export_tensor(self: Rc<Self>, tensor: &CpuTensor<'a>, dst: &mut [f32]) -> Result<()> {
        tensor.iter().zip(dst.iter_mut()).for_each(|(src, dst)| {
            *dst = src;
        });
        Ok(())
    }
}

impl<'a> Tensor for CpuTensor<'a> {
    type Pool = CpuTensorPoolRef<'a>;

    fn new(buf: Vec<f32>, shape: &[usize], pool: Self::Pool) -> Result<Self> {
        if buf.len() != shape.iter().product() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: format!("invalid shape {:?} for data of length {}", shape, buf.len()),
                cause: None,
            });
        }

        let strider = TensorStrider::new(shape.to_vec());
        Ok(Self {
            buf: buf.into(),
            strider,
            pool: pool.clone(),
        })
    }

    fn alloc(shape: &[usize], pool: Self::Pool) -> Result<Self> {
        let buf = vec![0.0; shape.iter().product()];
        Self::new(buf, shape, pool)
    }

    fn view(self, shape: &[usize]) -> Result<Self> {
        let strider = self.strider.view(shape.to_vec())?;
        Ok(Self {
            buf: self.buf,
            strider,
            pool: self.pool.clone(),
        })
    }

    fn repeat(self, repeats: &[usize]) -> Result<Self> {
        let strider = self.strider.repeat(repeats.to_vec())?;
        Ok(Self {
            buf: self.buf,
            strider,
            pool: self.pool.clone(),
        })
    }

    fn transpose(self, dims: &[usize]) -> Result<Self> {
        let strider = self.strider.transpose(dims)?;
        Ok(Self {
            buf: self.buf,
            strider,
            pool: self.pool.clone(),
        })
    }

    fn extend(&mut self, t: &Self) -> Result<()> {
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

        self.buf.extend(t.iter());
        let new_shape = {
            let mut shape = self.shape().to_vec();
            shape[0] += 1;
            shape
        };
        self.strider = TensorStrider::new(new_shape);
        Ok(())
    }

    fn copy_from(&mut self, t: &Self, pos: &[usize], len: usize) -> Result<()> {
        if !self.is_owned() {
            return Err((ErrorKind::TensorError, "not owned").into());
        }
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        self.iter_mut()?
            .zip(t.iter_from(pos)?.take(len))
            .for_each(|(dst, src)| {
                *dst = src;
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view() -> Result<()> {
        let pool = CpuTensorPool::new();
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], pool.clone())?;
        let t = t.view(&[3, 2])?;

        let tr = t.view(&[2, 3])?;
        assert_eq!(tr.iter().collect::<Vec<f32>>(), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        ]);
        Ok(())
    }

    #[test]
    fn test_tensor_iter_axis() -> Result<()> {
        struct Test<'a> {
            tensor: &'a CpuTensor<'a>,
            input: (Vec<usize>, usize),
            want: Vec<f32>,
        }

        // 1, 2, 3
        // 4, 5, 6
        let pool = CpuTensorPool::new();
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], pool.clone())?;

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
            let r = tt
                .tensor
                .iter_axis(&tt.input.0, tt.input.1)?
                .collect::<Vec<_>>();
            assert_eq!(r, tt.want);
        }

        // iter_axis with repeat
        // 1, 1, 2, 2, 3, 3
        // 4, 4, 5, 5, 6, 6
        let t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], pool.clone())?;
        let t = t.repeat(&[1, 2])?;

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
            },
        ];
        for tt in tests {
            let r = tt
                .tensor
                .iter_axis(&tt.input.0, tt.input.1)?
                .collect::<Vec<_>>();
            assert_eq!(r, tt.want);
        }

        Ok(())
    }

    #[test]
    fn test_tensor_iter_axis_on_repeat_and_transpose() -> Result<()> {
        struct Test<'a> {
            tensor: &'a CpuTensor<'a>,
            input: (Vec<usize>, usize),
            want: Vec<f32>,
        }

        // 0, 1, 2
        // 3, 4, 5
        let pool = CpuTensorPool::new();
        let t = CpuTensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], pool)?;

        // 0, 3,
        // 1, 4
        // 2, 5
        let t = t.transpose(&[1, 0])?;

        // 0, 0, 3, 3
        // 1, 1, 4, 4
        // 2, 2, 5, 5
        let t = t.repeat(&[1, 2])?;

        let tests = vec![
            Test {
                tensor: &t,
                input: (vec![0, 0], 1),
                want: vec![0.0, 0.0, 3.0, 3.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 0], 0),
                want: vec![0.0, 1.0, 2.0],
            },
            Test {
                tensor: &t,
                input: (vec![0, 1], 0),
                want: vec![0.0, 1.0, 2.0],
            },
        ];
        for tt in tests {
            let r = tt
                .tensor
                .iter_axis(&tt.input.0, tt.input.1)?
                .collect::<Vec<_>>();
            assert_eq!(r, tt.want);
        }

        Ok(())
    }

    #[test]
    fn test_tensor_iter_axis_mut() -> Result<()> {
        let pool = CpuTensorPool::new();
        let mut t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], pool.clone())?;
        let r = t
            .iter_axis_mut(vec![0, 0], 1)?
            .map(|f| *f)
            .collect::<Vec<_>>();
        assert_eq!(r, vec![1.0, 2.0, 3.0]);

        let mut t = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], pool)?;
        let r = t
            .iter_axis_mut(vec![0, 0], 0)?
            .map(|f| *f)
            .collect::<Vec<_>>();
        assert_eq!(r, vec![1.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_copy_from() -> Result<()> {
        // 1 2
        // 3 4
        let pool = CpuTensorPool::new();
        let t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], pool.clone())?;
        let mut t2 = CpuTensor::new(vec![0.0; 2], &[2], pool.clone())?;

        t2.copy_from(&t1, &[1, 0], 2)?;
        assert_eq!(t2.iter().collect::<Vec<_>>(), vec![3.0, 4.0]);

        t2.copy_from(&t1, &[0, 0], 2)?;
        assert_eq!(t2.iter().collect::<Vec<_>>(), vec![1.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_extend() -> Result<()> {
        let pool = CpuTensorPool::new();
        let mut t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], pool.clone())?;
        let t2 = CpuTensor::new(vec![1.0; 6], &[2, 3], pool)?;
        t1.extend(&t2)?;

        assert_eq!(t1.shape(), &[2, 2, 3]);
        assert_eq!(t1.iter().collect::<Vec<_>>(), &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]);
        Ok(())
    }
}
