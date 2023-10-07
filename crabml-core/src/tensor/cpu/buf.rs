use std::slice;
use rayon::{prelude::*, vec};

#[derive(Debug)]
pub enum CpuTensorBuf<'a, T: Copy+Send> {
    Owned(Vec<T>),
    Flat(&'a [T]),
    // Quantized8,
}

impl<'a, T: Copy+Send> CpuTensorBuf<'a, T> {
    pub fn at_unchecked(&self, pos: usize) -> T {
        match self {
            CpuTensorBuf::Owned(buf) => buf[pos],
            CpuTensorBuf::Flat(buf) => buf[pos],
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
        }
    }

    pub fn as_ref(&self) -> CpuTensorBuf<'_, T> {
        match self {
            CpuTensorBuf::Owned(buf) => CpuTensorBuf::Flat(buf),
            CpuTensorBuf::Flat(buf) => CpuTensorBuf::Flat(buf),
        }
    }

    pub fn extend(&mut self, iter: impl Iterator<Item = T>) {
        match self {
            CpuTensorBuf::Owned(buf) => buf.extend(iter),
            _ => unreachable!("only owned buffers can be extended"),
        }
    }

    pub fn iter_between(
        &self,
        start: usize,
        end: usize,
        step: usize,
    ) -> impl ExactSizeIterator<Item = &T> {
        match self {
            CpuTensorBuf::Owned(buf) => buf[start..end].iter().step_by(step),
            CpuTensorBuf::Flat(buf) => buf[start..end].iter().step_by(step),
        }
    }

    pub fn iter_mut_between(
        &mut self,
        start: usize,
        end: usize,
        step: usize,
    ) -> impl ExactSizeIterator<Item = &mut T> {
        match self {
            CpuTensorBuf::Owned(buf) => buf[start..end].iter_mut().step_by(step),
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }

    pub fn par_iter_mut_between(
        &mut self,
        start: usize,
        end: usize,
        step: usize,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = &mut T> {
        let buf = match self {
            CpuTensorBuf::Owned(buf) => buf,
            _ => unreachable!("only owned buffers can be mutable"),
        };

        buf[start..end].par_iter_mut().step_by(step)
    }


    pub fn iter(&self) -> impl Iterator<Item = &T> {
        match self {
            CpuTensorBuf::Owned(buf) => buf.iter(),
            CpuTensorBuf::Flat(buf) => buf.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        match self {
            CpuTensorBuf::Owned(buf) => buf.iter_mut(),
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }

    pub fn par_iter_mut(&mut self) -> impl rayon::iter::IndexedParallelIterator<Item = &mut T> {
        match self {
            CpuTensorBuf::Owned(buf) => buf.par_iter_mut(),
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }
}

impl<'a> CpuTensorBuf<'a, f32> {
    pub fn from_raw_bytes(buf: &'a [u8]) -> Self {
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

    pub fn par_iter(&self) -> impl rayon::iter::IndexedParallelIterator<Item = &f32> {
        match self {
            CpuTensorBuf::Owned(buf) => buf.par_iter(),
            CpuTensorBuf::Flat(buf) => buf.par_iter(),
        }
    }

    pub fn buf(&self) -> &[f32] {
        match self {
            CpuTensorBuf::Owned(buf) => buf,
            CpuTensorBuf::Flat(buf) => buf,
        }
    }

    pub fn buf_mut(&mut self) -> &mut [f32] {
        match self {
            CpuTensorBuf::Owned(buf) => buf,
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }
}

impl<T: Copy+Send> Clone for CpuTensorBuf<'_, T> {
    fn clone(&self) -> Self {
        match self {
            CpuTensorBuf::Owned(buf) => Self::Owned(buf.clone()),
            CpuTensorBuf::Flat(buf) => Self::Flat(buf),
        }
    }
}

impl<T: Copy+Send> Default for CpuTensorBuf<'_, T> {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}

impl<T: Copy+Send> From<Vec<T>> for CpuTensorBuf<'_, T> {
    fn from(buf: Vec<T>) -> Self {
        Self::Owned(buf)
    }
}

impl<'a, T: Copy+Send> From<&'a [T]> for CpuTensorBuf<'a, T> {
    fn from(buf: &'a [T]) -> Self {
        Self::Flat(buf)
    }
}
