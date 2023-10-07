use rayon::prelude::*;
use std::slice;

use crate::error::Result;
use crate::gguf::GGMLType;

use super::quant::BlockBufQ8_0;

/// All the quantized tensor are read-only.
/// to implement a quantized tensor, we need to implement the following:
/// - iter_range
#[derive(Debug)]
pub enum CpuTensorBuf<'a> {
    Owned(Vec<f32>),
    Flat(&'a [f32]),
    Q8_0(BlockBufQ8_0<'a>),
}

impl<'a> CpuTensorBuf<'a> {
    pub fn from_raw_bytes(buf: &'a [u8], typ: GGMLType) -> Result<Self> {
        match typ {
            GGMLType::F32 => Ok(Self::from_raw_bytes_f32(buf)),
            GGMLType::Q8_0 => Ok(CpuTensorBuf::Q8_0(BlockBufQ8_0::from_bytes(buf))),
            _ => unimplemented!(),
        }
    }

    pub fn at_unchecked(&self, pos: usize) -> f32 {
        match self {
            CpuTensorBuf::Owned(buf) => buf[pos],
            CpuTensorBuf::Flat(buf) => buf[pos],
            CpuTensorBuf::Q8_0(buf) => buf.iter_range(pos, pos + 1, 1).next().unwrap(),
        }
    }

    pub fn is_owned(&self) -> bool {
        match self {
            CpuTensorBuf::Owned(_) => true,
            _ => false,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuTensorBuf::Owned(buf) => buf.len(),
            CpuTensorBuf::Flat(buf) => buf.len(),
            CpuTensorBuf::Q8_0(buf) => buf.len(),
        }
    }

    pub fn as_ref(&self) -> CpuTensorBuf<'_> {
        match self {
            CpuTensorBuf::Owned(buf) => CpuTensorBuf::Flat(buf),
            CpuTensorBuf::Flat(buf) => CpuTensorBuf::Flat(buf),
            CpuTensorBuf::Q8_0(buf) => CpuTensorBuf::Q8_0(buf.clone()),
        }
    }

    pub fn extend(&mut self, iter: impl Iterator<Item = f32>) {
        match self {
            CpuTensorBuf::Owned(buf) => buf.extend(iter),
            _ => unreachable!("only owned buffers can be extended"),
        }
    }

    pub fn iter_range(&self, start: usize, end: usize, step: usize) -> CpuTensorBufIter {
        match self {
            CpuTensorBuf::Owned(buf) => {
                CpuTensorBufIter::StepBy(buf[start..end].iter().step_by(step))
            }
            CpuTensorBuf::Flat(buf) => {
                CpuTensorBufIter::StepBy(buf[start..end].iter().step_by(step))
            }
            CpuTensorBuf::Q8_0(buf) => CpuTensorBufIter::Boxed(
                Box::new(buf.iter_range(start, end, step)),
                self.len() / step,
            ),
        }
    }

    pub fn iter_range_mut(
        &mut self,
        start: usize,
        end: usize,
        step: usize,
    ) -> impl ExactSizeIterator<Item = &mut f32> {
        match self {
            CpuTensorBuf::Owned(buf) => buf[start..end].iter_mut().step_by(step),
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }

    pub fn par_iter_range_mut(
        &mut self,
        start: usize,
        end: usize,
        step: usize,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = &mut f32> {
        let buf = match self {
            CpuTensorBuf::Owned(buf) => buf,
            _ => unreachable!("only owned buffers can be mutable"),
        };

        buf[start..end].par_iter_mut().step_by(step)
    }

    pub fn iter(&self) -> CpuTensorBufIter {
        match self {
            CpuTensorBuf::Owned(buf) => CpuTensorBufIter::Slice(buf.iter()),
            CpuTensorBuf::Flat(buf) => CpuTensorBufIter::Slice(buf.iter()),
            CpuTensorBuf::Q8_0(buf) => {
                CpuTensorBufIter::Boxed(Box::new(buf.iter_range(0, buf.len(), 1)), self.len())
            }
        }
    }

    pub fn par_iter(&self) -> impl rayon::iter::IndexedParallelIterator<Item = &f32> {
        match self {
            CpuTensorBuf::Owned(buf) => buf.par_iter(),
            CpuTensorBuf::Flat(buf) => buf.par_iter(),
            CpuTensorBuf::Q8_0(buf) => unimplemented!(),
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        match self {
            CpuTensorBuf::Owned(buf) => buf.iter_mut(),
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }

    pub fn par_iter_mut(&mut self) -> impl rayon::iter::IndexedParallelIterator<Item = &mut f32> {
        match self {
            CpuTensorBuf::Owned(buf) => buf.par_iter_mut(),
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }

    pub fn from_raw_bytes_f32(buf: &'a [u8]) -> Self {
        let len = buf.len();
        assert_eq!(
            len % std::mem::size_of::<f32>(),
            0,
            "Length of slice must be multiple of f32 size"
        );
        let new_len = len / std::mem::size_of::<f32>();
        let ptr = buf.as_ptr() as *const f32;
        let f32_buf = unsafe { slice::from_raw_parts(ptr, new_len) };
        f32_buf.into()
    }

    pub fn buf(&self) -> &[f32] {
        match self {
            CpuTensorBuf::Owned(buf) => buf,
            CpuTensorBuf::Flat(buf) => buf,
            _ => unreachable!("only f32 buffers can access the raw buffer"),
        }
    }

    pub fn buf_mut(&mut self) -> &mut [f32] {
        match self {
            CpuTensorBuf::Owned(buf) => buf,
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }
}

impl Clone for CpuTensorBuf<'_> {
    fn clone(&self) -> Self {
        match self {
            CpuTensorBuf::Owned(buf) => Self::Owned(buf.clone()),
            CpuTensorBuf::Flat(buf) => Self::Flat(buf),
            CpuTensorBuf::Q8_0(buf) => Self::Q8_0(buf.clone()),
        }
    }
}

impl Default for CpuTensorBuf<'_> {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}

impl From<Vec<f32>> for CpuTensorBuf<'_> {
    fn from(buf: Vec<f32>) -> Self {
        Self::Owned(buf)
    }
}

impl<'a> From<&'a [f32]> for CpuTensorBuf<'a> {
    fn from(buf: &'a [f32]) -> Self {
        Self::Flat(buf)
    }
}

// a enum dispatcher seems 3 times faster than a trait object on the benchmarks
pub enum CpuTensorBufIter<'a> {
    Slice(slice::Iter<'a, f32>),
    StepBy(std::iter::StepBy<std::slice::Iter<'a, f32>>),
    Boxed(Box<dyn Iterator<Item = f32> + 'a>, usize),
}

impl<'a> Iterator for CpuTensorBufIter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CpuTensorBufIter::Slice(iter) => iter.next().cloned(),
            CpuTensorBufIter::StepBy(iter) => iter.next().cloned(),
            CpuTensorBufIter::Boxed(iter, _) => iter.next(),
        }
    }
}

impl<'a> ExactSizeIterator for CpuTensorBufIter<'a> {
    fn len(&self) -> usize {
        match self {
            CpuTensorBufIter::Slice(iter) => iter.len(),
            CpuTensorBufIter::StepBy(iter) => iter.len(),
            CpuTensorBufIter::Boxed(_, hint) => *hint,
        }
    }
}
