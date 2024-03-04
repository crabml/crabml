use std::simd::f32x32;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// both buf1 and buf2 have to be owned, the dtype should be the same
pub fn mul_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(buf1.is_owned());
    assert!(buf1.len() == buf2.len() || buf2.len() == 1);
    assert!(strider1.shape() == strider2.shape() || strider2.len() == 1);
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());
    assert!(buf1.dtype() == buf2.dtype());

    if buf2.len() == 1 {
        let ib = buf2.iter_f32().next().unwrap();
        buf1.iter_f32_mut().for_each(|ia| {
            *ia *= ib;
        });
        return Ok(());
    }

    for (ia, ib) in buf1.iter_f32_mut().zip(buf2.iter_f32()) {
        *ia *= ib;
    }

    Ok(())
}

#[allow(unused)]
fn mul_inplace_vec_f32(a: &mut [f32], b: &[f32]) {
    let ac = a.as_chunks_mut::<32>().0;
    let bc = b.as_chunks::<32>().0;
    ac.iter_mut().zip(bc).for_each(|(a, b)| {
        let mut va = f32x32::from_slice(a);
        let vb = f32x32::from_slice(b);
        va *= vb;
        va.copy_to_slice(a);
    });
}
