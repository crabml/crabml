use super::scalar::binary_inplace;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

pub fn add_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    binary_inplace::<_>(buf1, buf2, strider1, strider2, |ia, ib| *ia += ib)
}

#[allow(dead_code)]
pub fn sub_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    binary_inplace::<_>(buf1, buf2, strider1, strider2, |ia, ib| *ia -= ib)
}

pub fn mul_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    binary_inplace::<_>(buf1, buf2, strider1, strider2, |ia, ib| *ia *= ib)
}

pub fn div_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    binary_inplace::<_>(buf1, buf2, strider1, strider2, |ia, ib| *ia /= ib)
}
