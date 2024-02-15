use std::borrow::Cow;

use super::buf_f32::f32_buf_from_bytes;
use super::buf_f32::f32_buf_vec_dot_f32;
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
            GGMLType::F32 => Ok(CpuTensorBuf::F32(f32_buf_from_bytes(buf))),
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

    pub fn quantize(&self, dtype: GGMLType) -> Result<Self> {
        match dtype {
            GGMLType::F32 => Ok(CpuTensorBuf::F32(self.as_f32_ref().to_vec().into())),
            GGMLType::Q8_0 => Ok(CpuTensorBuf::Q8_0(QuantBufQ8_0::quantize(
                self.as_f32_ref(),
            ))),
            _ => Err((
                ErrorKind::TensorError,
                format!("quantize to {:?} is not supported", dtype),
            )
                .into()),
        }
    }

    pub fn vec_dot(&self, row: usize, b: &Self) -> f32 {
        assert!(
            self.dtype() == b.dtype(),
            "only same dtype can be dotted, but got {:?} and {:?}",
            self.dtype(),
            b.dtype()
        );

        use CpuTensorBuf::*;
        match (self, b) {
            (F32(a), F32(b)) => f32_buf_vec_dot_f32(a, row, b),
            (Q8_0(a), Q8_0(b)) => a.vec_dot(row, b),
            _ => unreachable!(),
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

    pub fn as_f32_ref(&self) -> &[f32] {
        match self {
            CpuTensorBuf::F32(buf) => buf,
            _ => panic!("not f32, but got {:?}", self.dtype()),
        }
    }

    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        match self {
            CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
            _ => panic!("not f32, but got {:?}", self.dtype()),
        }
    }

    /// the quantized tensor can not be iterated directly. to iterate the quantized tensor,
    /// use `dequantize` to convert it to f32/f16 tensor first.
    pub fn iter_f32(&self) -> impl Iterator<Item = f32> + '_ {
        // TODO: convert f16 to f32 here, to make debug easier.
        self.as_f32_ref().iter().copied()
    }

    pub fn iter_f32_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.as_f32_mut().iter_mut()
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
