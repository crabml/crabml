use crate::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

pub fn add_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(buf1.len() % buf2.len() == 0);
    assert!(strider1.shape().last() == strider2.shape().last() || buf2.len() == 1);
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    if buf2.len() == 1 {
        let ib = buf2.iter_f32().next().unwrap();
        buf1.iter_f32_mut().for_each(|ia| {
            *ia += ib;
        });
        return Ok(());
    }

    let buf1 = buf1.as_f32_mut();
    let buf2 = buf2.as_f32_ref();
    buf1.chunks_exact_mut(4)
        .zip(buf2.chunks_exact(4).cycle())
        .for_each(|(ia, ib)| {
            let va = std::simd::f32x4::from_slice(ia);
            let vb = std::simd::f32x4::from_slice(ib);
            let va = va + vb;
            va.copy_to_slice(ia);
        });
    Ok(())
}

pub fn mul_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(buf1.len() % buf2.len() == 0);
    assert!(strider1.shape().last() == strider2.shape().last() || buf2.len() == 1);
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    if buf2.len() == 1 {
        let ib = buf2.iter_f32().next().unwrap();
        buf1.iter_f32_mut().for_each(|ia| {
            *ia *= ib;
        });
        return Ok(());
    }

    let buf1 = buf1.as_f32_mut();
    let buf2 = buf2.as_f32_ref();
    buf1.chunks_exact_mut(4)
        .zip(buf2.chunks_exact(4).cycle())
        .for_each(|(ia, ib)| {
            let va = std::simd::f32x4::from_slice(ia);
            let vb = std::simd::f32x4::from_slice(ib);
            let va = va * vb;
            va.copy_to_slice(ia);
        });

    Ok(())
}

#[allow(dead_code)]
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
    assert!(strider1.shape().last() == strider2.shape().last() || buf2.len() == 1);
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    if buf2.len() == 1 {
        let ib = buf2.iter_f32().next().unwrap();
        buf1.iter_f32_mut().for_each(|ia| {
            f(ia, ib);
        });
        return Ok(());
    }

    // it seems that using cycle is slower
    buf1.iter_f32_mut()
        .zip(buf2.as_f32_ref().iter().cycle())
        .for_each(|(ia, ib)| {
            f(ia, *ib);
        });

    Ok(())
}
