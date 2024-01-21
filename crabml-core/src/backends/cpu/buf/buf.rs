use std::borrow::Cow;
use std::slice;

use crate::backends::cpu::buf::QuantBufQ8_0;
use crate::error::Result;
use crate::gguf::GGMLType;

/// All the quantized tensor are read-only.
#[derive(Debug)]
pub enum CpuTensorBuf<'a> {
    F32(Cow<'a, [f32]>),
    Q8_0(QuantBufQ8_0<'a>),
}

impl<'a> CpuTensorBuf<'a> {
    pub fn from_raw_bytes(buf: &'a [u8], typ: GGMLType) -> Result<Self> {
        match typ {
            GGMLType::F32 => Ok(Self::from_raw_bytes_f32(buf)),
            GGMLType::Q8_0 => Ok(CpuTensorBuf::Q8_0(QuantBufQ8_0::from_bytes(buf))),
            _ => unimplemented!(),
        }
    }

    pub fn at_unchecked(&self, pos: usize) -> f32 {
        match self {
            CpuTensorBuf::F32(buf) => buf[pos],
            CpuTensorBuf::Q8_0(buf) => buf.iter_range(pos, pos + 1, 1).next().unwrap(),
        }
    }

    pub fn is_owned(&self) -> bool {
        match self {
            CpuTensorBuf::F32(Cow::Owned(_)) => true,
            _ => false,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuTensorBuf::F32(buf) => buf.len(),
            CpuTensorBuf::Q8_0(buf) => buf.len(),
        }
    }

    pub fn typ(&self) -> GGMLType {
        match self {
            CpuTensorBuf::F32(_) => GGMLType::F32,
            CpuTensorBuf::Q8_0(_) => GGMLType::Q8_0,
        }
    }

    pub fn as_ref(&'a self) -> CpuTensorBuf<'a> {
        match self {
            CpuTensorBuf::F32(buf) => CpuTensorBuf::F32(Cow::Borrowed(buf.as_ref())),
            CpuTensorBuf::Q8_0(buf) => CpuTensorBuf::Q8_0(buf.clone()),
        }
    }

    pub fn extend(&mut self, iter: impl Iterator<Item = f32>) {
        match self {
            CpuTensorBuf::F32(Cow::Owned(buf)) => buf.extend(iter),
            _ => unreachable!("only owned buffers can be extended"),
        }
    }

    pub fn iter(&self) -> CpuTensorBufIter {
        match self {
            CpuTensorBuf::F32(buf) => CpuTensorBufIter::Slice(buf.iter()),
            CpuTensorBuf::Q8_0(buf) => {
                CpuTensorBufIter::Boxed(Box::new(buf.iter_range(0, buf.len(), 1)), self.len())
            }
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        match self {
            CpuTensorBuf::F32(Cow::Owned(buf)) => buf.iter_mut(),
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }

    fn from_raw_bytes_f32(buf: &'a [u8]) -> Self {
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

    pub fn buf_mut(&mut self) -> &mut [f32] {
        match self {
            CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
            _ => unreachable!("only owned buffers can be mutable"),
        }
    }
}

impl Clone for CpuTensorBuf<'_> {
    fn clone(&self) -> Self {
        match self {
            CpuTensorBuf::F32(buf) => Self::F32(buf.clone()),
            CpuTensorBuf::Q8_0(buf) => Self::Q8_0(buf.clone()),
        }
    }
}

impl Default for CpuTensorBuf<'_> {
    fn default() -> Self {
        Self::F32(Vec::new().into())
    }
}

impl From<Vec<f32>> for CpuTensorBuf<'_> {
    fn from(buf: Vec<f32>) -> Self {
        Self::F32(buf.into())
    }
}

impl<'a> From<&'a [f32]> for CpuTensorBuf<'a> {
    fn from(buf: &'a [f32]) -> Self {
        Self::F32(buf.into())
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

pub trait VecDotF32 {
    fn vec_dot_f32(&self, row: usize, x: &[f32]) -> f32;
}
