use std::simd::f32x32;
use std::simd::num::SimdFloat;

use crate::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::gguf::GGMLType;
use crate::tensor::TensorStrider;

pub fn rms_norm_inplace(
    buf: &mut CpuTensorBuf<'_>,
    strider: &TensorStrider,
    eps: f32,
) -> Result<()> {
    assert!(strider.is_contiguous());
    assert!(strider.shape().len() == 1 || strider.shape().len() == 2);
    assert!(buf.dtype() == GGMLType::F32);

    let (rows, cols) = if strider.shape().len() == 1 {
        (1, strider.shape()[0])
    } else {
        (strider.shape()[0], strider.shape()[1])
    };

    let buf = buf.as_f32_mut();
    for row in 0..rows {
        rms_norm_inplace_vec_f32(&mut buf[row * cols..(row + 1) * cols], eps)
    }

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
