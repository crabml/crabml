use std::borrow::Cow;

use half::f16;

use super::buf_f16::dequantize_f16_buf;
use super::buf_f16::f16_buf_from_bytes;
use super::buf_f16::quantize_f32_f16;
use super::buf_f32::f32_buf_from_bytes;
use super::buf_f32::vec_dot_f32_f32;
use crate::cpu::buf::buf_f16::vec_dot_f16_f16;
use crate::cpu::buf::QuantBufQ2K;
use crate::cpu::buf::QuantBufQ3K;
use crate::cpu::buf::QuantBufQ4K;
use crate::cpu::buf::QuantBufQ4_0;
use crate::cpu::buf::QuantBufQ4_1;
use crate::cpu::buf::QuantBufQ5K;
use crate::cpu::buf::QuantBufQ5_0;
use crate::cpu::buf::QuantBufQ5_1;
use crate::cpu::buf::QuantBufQ6K;
use crate::cpu::buf::QuantBufQ8K;
use crate::cpu::buf::QuantBufQ8_0;
use crate::cpu::buf::QuantBufQ8_1;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::gguf::GGMLType;

/// All the quantized tensor are read-only.
#[derive(Debug)]
#[non_exhaustive]
pub enum CpuTensorBuf<'a> {
    F32(Cow<'a, [f32]>),
    F16(Cow<'a, [f16]>),
    Q2K(QuantBufQ2K<'a>),
    Q3K(QuantBufQ3K<'a>),
    Q8_0(QuantBufQ8_0<'a>),
    Q8_1(QuantBufQ8_1<'a>),
    Q8K(QuantBufQ8K<'a>),
    Q4_0(QuantBufQ4_0<'a>),
    Q4_1(QuantBufQ4_1<'a>),
    Q4K(QuantBufQ4K<'a>),
    Q5_0(QuantBufQ5_0<'a>),
    Q5_1(QuantBufQ5_1<'a>),
    Q5K(QuantBufQ5K<'a>),
    Q6K(QuantBufQ6K<'a>),
}

impl<'a> CpuTensorBuf<'a> {
    pub fn from_raw_bytes(buf: &'a [u8], typ: GGMLType) -> Result<Self> {
        match typ {
            GGMLType::F32 => Ok(CpuTensorBuf::F32(f32_buf_from_bytes(buf))),
            GGMLType::F16 => Ok(CpuTensorBuf::F16(f16_buf_from_bytes(buf))),
            GGMLType::Q2K => Ok(CpuTensorBuf::Q2K(QuantBufQ2K::from_bytes(buf))),
            GGMLType::Q3K => Ok(CpuTensorBuf::Q3K(QuantBufQ3K::from_bytes(buf))),
            GGMLType::Q8_0 => Ok(CpuTensorBuf::Q8_0(QuantBufQ8_0::from_bytes(buf))),
            GGMLType::Q8_1 => Ok(CpuTensorBuf::Q8_1(QuantBufQ8_1::from_bytes(buf))),
            GGMLType::Q8K => Ok(CpuTensorBuf::Q8K(QuantBufQ8K::from_bytes(buf))),
            GGMLType::Q4_0 => Ok(CpuTensorBuf::Q4_0(QuantBufQ4_0::from_bytes(buf))),
            GGMLType::Q4_1 => Ok(CpuTensorBuf::Q4_1(QuantBufQ4_1::from_bytes(buf))),
            GGMLType::Q4K => Ok(CpuTensorBuf::Q4K(QuantBufQ4K::from_bytes(buf))),
            GGMLType::Q5_0 => Ok(CpuTensorBuf::Q5_0(QuantBufQ5_0::from_bytes(buf))),
            GGMLType::Q5_1 => Ok(CpuTensorBuf::Q5_1(QuantBufQ5_1::from_bytes(buf))),
            GGMLType::Q5K => Ok(CpuTensorBuf::Q5K(QuantBufQ5K::from_bytes(buf))),
            GGMLType::Q6K => Ok(CpuTensorBuf::Q6K(QuantBufQ6K::from_bytes(buf))),
            _ => unimplemented!(),
        }
    }

    pub fn as_bytes(&'a self) -> &'a [u8] {
        match self {
            Self::F32(buf) => bytemuck::cast_slice(buf),
            Self::F16(buf) => bytemuck::cast_slice(buf),
            Self::Q2K(buf) => buf.as_bytes(),
            Self::Q3K(buf) => buf.as_bytes(),
            Self::Q4_0(buf) => buf.as_bytes(),
            Self::Q4_1(buf) => buf.as_bytes(),
            Self::Q4K(buf) => buf.as_bytes(),
            Self::Q5_0(buf) => buf.as_bytes(),
            Self::Q5_1(buf) => buf.as_bytes(),
            Self::Q5K(buf) => buf.as_bytes(),
            Self::Q6K(buf) => buf.as_bytes(),
            Self::Q8_0(buf) => buf.as_bytes(),
            Self::Q8_1(buf) => buf.as_bytes(),
            Self::Q8K(buf) => buf.as_bytes(),
        }
    }

    pub fn is_owned(&self) -> bool {
        matches!(
            self,
            CpuTensorBuf::F32(Cow::Owned(_)) | CpuTensorBuf::F16(Cow::Owned(_))
        )
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, CpuTensorBuf::F32(_))
    }

    pub fn len(&self) -> usize {
        match self {
            CpuTensorBuf::F32(buf) => buf.len(),
            CpuTensorBuf::F16(buf) => buf.len(),
            CpuTensorBuf::Q2K(buf) => buf.len(),
            CpuTensorBuf::Q3K(buf) => buf.len(),
            CpuTensorBuf::Q8_0(buf) => buf.len(),
            CpuTensorBuf::Q8_1(buf) => buf.len(),
            CpuTensorBuf::Q8K(buf) => buf.len(),
            CpuTensorBuf::Q5_0(buf) => buf.len(),
            CpuTensorBuf::Q5_1(buf) => buf.len(),
            CpuTensorBuf::Q4_0(buf) => buf.len(),
            CpuTensorBuf::Q4_1(buf) => buf.len(),
            CpuTensorBuf::Q4K(buf) => buf.len(),
            CpuTensorBuf::Q5K(buf) => buf.len(),
            CpuTensorBuf::Q6K(buf) => buf.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> GGMLType {
        match self {
            CpuTensorBuf::F32(_) => GGMLType::F32,
            CpuTensorBuf::F16(_) => GGMLType::F16,
            CpuTensorBuf::Q2K(_) => GGMLType::Q2K,
            CpuTensorBuf::Q3K(_) => GGMLType::Q3K,
            CpuTensorBuf::Q8_0(_) => GGMLType::Q8_0,
            CpuTensorBuf::Q8_1(_) => GGMLType::Q8_1,
            CpuTensorBuf::Q8K(_) => GGMLType::Q8K,
            CpuTensorBuf::Q4_0(_) => GGMLType::Q4_0,
            CpuTensorBuf::Q4_1(_) => GGMLType::Q4_1,
            CpuTensorBuf::Q4K(_) => GGMLType::Q4K,
            CpuTensorBuf::Q5_0(_) => GGMLType::Q5_0,
            CpuTensorBuf::Q5_1(_) => GGMLType::Q5_1,
            CpuTensorBuf::Q5K(_) => GGMLType::Q5K,
            CpuTensorBuf::Q6K(_) => GGMLType::Q6K,
        }
    }

    pub fn vec_dot_rhs_dtype(&self) -> GGMLType {
        match self {
            CpuTensorBuf::F32(_) => GGMLType::F32,
            CpuTensorBuf::F16(_) => GGMLType::F16,
            CpuTensorBuf::Q2K(_) => GGMLType::Q8K,
            CpuTensorBuf::Q3K(_) => GGMLType::Q8K,
            CpuTensorBuf::Q8_0(_) => GGMLType::Q8_0,
            CpuTensorBuf::Q8_1(_) => GGMLType::Q8_1,
            CpuTensorBuf::Q8K(_) => GGMLType::Q8K,
            CpuTensorBuf::Q5_0(_) => GGMLType::Q8_0,
            CpuTensorBuf::Q5_1(_) => GGMLType::Q8_1,
            CpuTensorBuf::Q4_0(_) => GGMLType::Q8_0,
            CpuTensorBuf::Q4_1(_) => GGMLType::Q8_1,
            CpuTensorBuf::Q4K(_) => GGMLType::Q8K,
            CpuTensorBuf::Q5K(_) => GGMLType::Q8K,
            CpuTensorBuf::Q6K(_) => GGMLType::Q8K,
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

        match dtype {
            GGMLType::F32 => Ok(CpuTensorBuf::F32(match self {
                CpuTensorBuf::F32(buf) => buf,
                CpuTensorBuf::F16(buf) => dequantize_f16_buf(&buf, 0).collect(),
                CpuTensorBuf::Q2K(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q3K(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q8_0(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q8_1(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q8K(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q4_0(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q4_1(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q4K(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q5_0(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q5_1(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q5K(buf) => buf.dequantize(0).collect(),
                CpuTensorBuf::Q6K(buf) => buf.dequantize(0).collect(),
            })),
            GGMLType::F16 => unimplemented!(),
            _ => unreachable!(),
        }
    }

    pub fn quantize(&self, dtype: GGMLType) -> Result<Self> {
        match dtype {
            GGMLType::F32 => Ok(CpuTensorBuf::F32(self.as_f32_ref().to_vec().into())),
            GGMLType::F16 => Ok(CpuTensorBuf::F16(quantize_f32_f16(self.as_f32_ref()))),
            GGMLType::Q2K => Ok(CpuTensorBuf::Q2K(QuantBufQ2K::quantize(self.as_f32_ref()))),
            GGMLType::Q3K => Ok(CpuTensorBuf::Q3K(QuantBufQ3K::quantize(self.as_f32_ref()))),
            GGMLType::Q8_0 => Ok(CpuTensorBuf::Q8_0(QuantBufQ8_0::quantize(
                self.as_f32_ref(),
            ))),
            GGMLType::Q8_1 => Ok(CpuTensorBuf::Q8_1(QuantBufQ8_1::quantize(
                self.as_f32_ref(),
            ))),
            GGMLType::Q8K => Ok(CpuTensorBuf::Q8K(QuantBufQ8K::quantize(self.as_f32_ref()))),
            GGMLType::Q4_0 => Ok(CpuTensorBuf::Q4_0(QuantBufQ4_0::quantize(
                self.as_f32_ref(),
            ))),
            GGMLType::Q4_1 => Ok(CpuTensorBuf::Q4_1(QuantBufQ4_1::quantize(
                self.as_f32_ref(),
            ))),
            GGMLType::Q4K => Ok(CpuTensorBuf::Q4K(QuantBufQ4K::quantize(self.as_f32_ref()))),
            GGMLType::Q5_1 => Ok(CpuTensorBuf::Q5_1(QuantBufQ5_1::quantize(
                self.as_f32_ref(),
            ))),
            GGMLType::Q5_0 => Ok(CpuTensorBuf::Q5_0(QuantBufQ5_0::quantize(
                self.as_f32_ref(),
            ))),
            GGMLType::Q5K => Ok(CpuTensorBuf::Q5K(QuantBufQ5K::quantize(self.as_f32_ref()))),
            GGMLType::Q6K => Ok(CpuTensorBuf::Q6K(QuantBufQ6K::quantize(self.as_f32_ref()))),
            _ => Err((
                ErrorKind::TensorError,
                format!("quantize to {:?} is not supported", dtype),
            )
                .into()),
        }
    }

    pub fn vec_dot(&self, a_offset: usize, b: &Self, b_offset: usize, len: usize) -> f32 {
        use CpuTensorBuf::*;
        match (self, b) {
            (F32(a), F32(b)) => vec_dot_f32_f32(a, a_offset, b, b_offset, len),
            (F16(a), F16(b)) => vec_dot_f16_f16(a, a_offset, b, b_offset, len),
            (Q2K(a), Q8K(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q3K(a), Q8K(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q8_0(a), Q8_0(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q8_1(a), Q8_1(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q8K(a), Q8K(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q4_0(a), Q8_0(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q4_1(a), Q8_1(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q4K(a), Q8K(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q5_0(a), Q8_0(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q5_1(a), Q8_1(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q5K(a), Q8K(b)) => a.vec_dot(a_offset, b, b_offset, len),
            (Q6K(a), Q8K(b)) => a.vec_dot(a_offset, b, b_offset, len),
            _ => unreachable!(),
        }
    }

    pub fn extend(&mut self, iter: impl Iterator<Item = f32>) {
        match self {
            CpuTensorBuf::F32(Cow::Owned(buf)) => buf.extend(iter),
            CpuTensorBuf::F16(Cow::Owned(buf)) => {
                let iter = iter.map(f16::from_f32);
                buf.extend(iter);
            }
            _ => unreachable!("only owned buffers can be extended"),
        }
    }

    pub fn copy_from(
        &mut self,
        src: &Self,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        assert!(self.is_owned(), "only owned buffers can be copied to");
        assert!(
            self.dtype() == GGMLType::F32 || self.dtype() == GGMLType::F16,
            "only f32/f16 can be copied to"
        );

        match src {
            CpuTensorBuf::F32(buf) => {
                self.copy_from_iter(buf.iter().skip(src_offset).cloned(), dst_offset, len)
            }
            CpuTensorBuf::F16(buf) => {
                self.copy_from_iter(dequantize_f16_buf(buf, src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q2K(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q3K(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q8_0(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q8_1(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q8K(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q4_0(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q4_1(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q4K(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q5_0(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q5_1(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q5K(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
            CpuTensorBuf::Q6K(buf) => {
                self.copy_from_iter(buf.dequantize(src_offset), dst_offset, len)
            }
        };

        Ok(())
    }

    pub fn copy_from_iter(
        &mut self,
        iter: impl Iterator<Item = f32>,
        dst_offset: usize,
        len: usize,
    ) {
        assert!(self.is_owned(), "only owned buffers can be copied to");
        assert!(
            self.dtype() == GGMLType::F32 || self.dtype() == GGMLType::F16,
            "only f32/f16 can be copied to"
        );

        match self {
            CpuTensorBuf::F32(Cow::Owned(buf)) => buf[dst_offset..dst_offset + len]
                .iter_mut()
                .zip(iter)
                .for_each(|(dst, src)| {
                    *dst = src;
                }),
            CpuTensorBuf::F16(Cow::Owned(buf)) => buf[dst_offset..dst_offset + len]
                .iter_mut()
                .zip(iter)
                .for_each(|(dst, src)| {
                    *dst = f16::from_f32(src);
                }),
            _ => unreachable!("only f32/f16 buffers can be copied"),
        }
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
            _ => panic!(
                "not owned f32, but got {:?}, owned: {}",
                self.dtype(),
                self.is_owned()
            ),
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
            CpuTensorBuf::F16(buf) => Self::F16(buf.clone()),
            CpuTensorBuf::Q2K(buf) => Self::Q2K(buf.clone()),
            CpuTensorBuf::Q3K(buf) => Self::Q3K(buf.clone()),
            CpuTensorBuf::Q8_0(buf) => Self::Q8_0(buf.clone()),
            CpuTensorBuf::Q8_1(buf) => Self::Q8_1(buf.clone()),
            CpuTensorBuf::Q8K(buf) => Self::Q8K(buf.clone()),
            CpuTensorBuf::Q5_0(buf) => Self::Q5_0(buf.clone()),
            CpuTensorBuf::Q5_1(buf) => Self::Q5_1(buf.clone()),
            CpuTensorBuf::Q4_0(buf) => Self::Q4_0(buf.clone()),
            CpuTensorBuf::Q4_1(buf) => Self::Q4_1(buf.clone()),
            CpuTensorBuf::Q4K(buf) => Self::Q4K(buf.clone()),
            CpuTensorBuf::Q5K(buf) => Self::Q5K(buf.clone()),
            CpuTensorBuf::Q6K(buf) => Self::Q6K(buf.clone()),
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
