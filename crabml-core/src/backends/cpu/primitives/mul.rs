use std::simd::f32x32;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// buf1 have to be owned, buf2 can be quantized
pub fn mul_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(buf1.is_owned());
    assert!(strider1.shape() == strider2.shape());
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    for (ia, ib) in buf1.iter_mut().zip(buf2.iter()) {
        *ia *= ib;
    }

    Ok(())
}

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
