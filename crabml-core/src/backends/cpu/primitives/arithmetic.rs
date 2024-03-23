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

#[inline]
pub fn binary_inplace<'a, F>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
    f: F,
) -> Result<()>
where
    F: Fn(&mut f32, f32),
{
    assert!(buf1.len() % buf2.len() == 0);
    assert!(strider1.shape().last() == strider2.shape().last());
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    if buf2.len() == 1 {
        let ib = buf2.iter_f32().next().unwrap();
        buf1.iter_f32_mut().for_each(|ia| {
            f(ia, ib);
        });
        return Ok(());
    }

    buf1.iter_f32_mut()
        .zip(buf2.as_f32_ref().iter().cycle())
        .for_each(|(ia, ib)| {
            f(ia, *ib);
        });

    Ok(())
}
