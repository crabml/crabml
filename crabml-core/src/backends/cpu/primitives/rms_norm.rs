use std::borrow::Cow;
use std::simd::f32x32;
use std::simd::SimdFloat;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

pub fn rms_norm_inplace(
    buf: &mut CpuTensorBuf<'_>,
    strider: &TensorStrider,
    eps: f32,
) -> Result<()> {
    assert!(strider.is_contiguous());
    assert!(strider.shape().len() == 1);

        if let CpuTensorBuf::F32(Cow::Owned(xb)) = buf {
              rms_norm_inplace_vec_f32(xb, eps);
                return Ok(());
          }


    let len = strider.shape()[0];
    let sum = buf.iter_f32().fold(0.0, |s, n| s + n * n);
    let rms = ((sum / len as f32) + eps).sqrt();
    buf.iter_f32_mut().for_each(|n| *n /= rms);
    Ok(())
}

fn rms_norm_inplace_vec_f32(x: &mut [f32], eps: f32) {
    let len = x.len();
    assert!(len % 32 == 0);
    let mut sum = 0.0;
    for chunk in x.as_chunks::<32>().0 {
        let mut v = f32x32::from_slice(chunk);
        v *= v;
        sum += v.reduce_sum();
    }
    let rms = ((sum / len as f32) + eps).sqrt();
    for chunk in x.as_chunks_mut::<32>().0 {
        let mut v = f32x32::from_slice(chunk);
        v /= f32x32::splat(rms);
        v.copy_to_slice(chunk);
    }
}
