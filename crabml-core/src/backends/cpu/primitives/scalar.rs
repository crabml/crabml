use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

#[inline]
pub fn unary<'a, F>(buf1: &mut CpuTensorBuf<'a>, f: F) -> Result<()>
where F: Fn(f32) -> f32 {
    buf1.iter_f32_mut().for_each(|ia| *ia = f(*ia));
    Ok(())
}

#[allow(dead_code)]
#[inline]
pub fn unary_inplace<'a, F>(buf1: &mut CpuTensorBuf<'a>, f: F) -> Result<()>
where F: Fn(&mut f32) {
    buf1.iter_f32_mut().for_each(|ia| f(ia));
    Ok(())
}

#[allow(dead_code)]
#[inline]
pub fn binary<'a, F>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
    f: F,
) -> Result<()>
where
    F: Fn(f32, f32) -> f32,
{
    assert!(buf1.len() == buf2.len() || buf2.len() == 1);
    assert!(strider1.shape() == strider2.shape() || strider2.len() == 1);
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    if buf2.len() == 1 {
        let ib = buf2.iter_f32().next().unwrap();
        buf1.iter_f32_mut().for_each(|ia| {
            *ia = f(*ia, ib);
        });
        return Ok(());
    }

    buf1.iter_f32_mut()
        .zip(buf2.iter_f32())
        .for_each(|(ia, ib)| {
            *ia = f(*ia, ib);
        });

    Ok(())
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
    assert!(buf1.len() == buf2.len() || buf2.len() == 1);
    assert!(strider1.shape() == strider2.shape() || strider2.len() == 1);
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
        .zip(buf2.iter_f32())
        .for_each(|(ia, ib)| {
            f(ia, ib);
        });

    Ok(())
}
