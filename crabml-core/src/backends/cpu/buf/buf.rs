use std::borrow::Cow;
use std::slice;

use crate::backends::cpu::buf::QuantBufQ8_0;
use crate::error::ErrorKind;
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

    pub fn is_owned(&self) -> bool {
        match self {
            CpuTensorBuf::F32(Cow::Owned(_)) => true,
            _ => false,
        }
    }

    pub fn is_quantized(&self) -> bool {
        match self {
            CpuTensorBuf::F32(_) => true,
            _ => false,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuTensorBuf::F32(buf) => buf.len(),
            CpuTensorBuf::Q8_0(buf) => buf.len(),
        }
    }

    pub fn dtype(&self) -> GGMLType {
        match self {
            CpuTensorBuf::F32(_) => GGMLType::F32,
            CpuTensorBuf::Q8_0(_) => GGMLType::Q8_0,
        }
    }

    /// dequantize the quantized tensors to f32 or f16.
    /// f32 to f16 is not considered as dequantization, but it still will be supported to
    /// simplify the conversion on half-precision activation is enabled.
    pub fn dequantize(self, dtype: GGMLType) -> Result<Self> {
        if dtype != GGMLType::F32 && dtype != GGMLType::F16 {
            return Err((
                ErrorKind::TensorError,
                format!("dequantize to {:?} is not supported", dtype),
            )
                .into());
        }

        match self {
            CpuTensorBuf::F32(buf) => Ok(CpuTensorBuf::F32(buf)),
            CpuTensorBuf::Q8_0(buf) => match dtype {
                GGMLType::F32 => Ok(CpuTensorBuf::F32(buf.dequantize(0).collect())),
                _ => unimplemented!(),
            },
        }
    }

    pub fn extend(&mut self, iter: impl Iterator<Item = f32>) {
        match self {
            CpuTensorBuf::F32(Cow::Owned(buf)) => buf.extend(iter),
            _ => unreachable!("only owned buffers can be extended"),
        }
    }

    pub fn copy_from(&mut self, src: &Self, offset: usize, len: usize) -> Result<()> {
        assert!(self.is_owned(), "only owned buffers can be copied to");
        assert!(
            self.dtype() == GGMLType::F32 || self.dtype() == GGMLType::F16,
            "only f32/f16 can be copied to"
        );
        assert!(self.dtype() == src.dtype(), "only same dtype can be copied");

        match src {
            CpuTensorBuf::F32(buf) => {
                let src_iter = buf.iter().skip(offset).take(len);
                self.iter_f32_mut().zip(src_iter).for_each(|(dst, src)| {
                    *dst = *src;
                });
            }
            // TODO: add f16 support
            _ => unreachable!("only f32/f16 buffers can be copied"),
        };

        Ok(())
    }

    /// the quantized tensor can not be iterated directly. to iterate the quantized tensor,
    /// use `dequantize` to convert it to f32/f16 tensor first.
    pub fn iter_f32(&self) -> impl Iterator<Item = f32> + '_ {
        match self {
            CpuTensorBuf::F32(buf) => buf.iter().cloned(),
            _ => unreachable!("only f32/f16 buffers can be iterated"),
        }
    }

    pub fn iter_f32_mut(&mut self) -> impl Iterator<Item = &mut f32> {
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
}

impl Clone for CpuTensorBuf<'_> {
    fn clone(&self) -> Self {
        match self {
            CpuTensorBuf::F32(buf) => Self::F32(buf.clone()),
            CpuTensorBuf::Q8_0(buf) => Self::Q8_0(buf.clone()),
        }
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

pub trait CpuTensorBufVecDot {
    fn vec_dot_f32(&self, row: usize, x: &[f32]) -> f32;
}
