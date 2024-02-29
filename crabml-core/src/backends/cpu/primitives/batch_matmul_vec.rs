use rayon::prelude::*;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// (b, m, k) @ (b, k, ) -> (b, m, )
// a is allowed to be not contiguous, but not quantized
pub fn batch_matmul_vec<'a>(
    a: &CpuTensorBuf<'a>,
    b: &CpuTensorBuf<'a>,
    c: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(strider1.shape().len() == 3);
    assert!(strider2.shape().len() == 2);
    assert!(strider1.shape()[0] == strider2.shape()[0]);
    assert!(strider1.shape()[2] == strider2.shape()[1]);
    assert!(strider2.is_contiguous());

    let bufa = a.as_f32_ref();
    let bufb = b.as_f32_ref();
    let bufc = c.as_f32_mut();

    let m = strider1.shape()[1];
    let k = strider1.shape()[2];
    let bi_stride = strider1.strides()[0];
    let mi_stride = strider1.strides()[1];
    let ki_stride = strider1.strides()[2];

    bufc.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
        let mi = i % m;
        let bi = (i - mi) / m;
        *bufcp = dot_product_f32_simd(
            bufa,
            bi * bi_stride + mi * mi_stride,
            ki_stride,
            k,
            &bufb[bi * k..(bi + 1) * k],
        );
    });

    Ok(())
}

#[allow(unused)]
pub fn dot_product_f32(a: &[f32], a_base: usize, a_stride: usize, k: usize, b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let k_rounded = k - k % 4;
    for ki in (0..k_rounded).step_by(4) {
        sum += a[a_base + ki * a_stride] * b[ki];
        sum += a[a_base + (ki + 1) * a_stride] * b[ki + 1];
        sum += a[a_base + (ki + 2) * a_stride] * b[ki + 2];
        sum += a[a_base + (ki + 3) * a_stride] * b[ki + 3];
    }
    for ki in (k_rounded..k).step_by(1) {
        sum += a[a_base + ki * a_stride] * b[ki];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
pub fn dot_product_f32_simd(a: &[f32], a_base: usize, a_stride: usize, k: usize, b: &[f32]) -> f32 {
    use std::arch::aarch64;

    unsafe {
        let a_ptr = a.as_ptr().add(a_base);

        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        let k_rounded = k - k % 8;
        for ki in (0..k_rounded).step_by(8) {
            let av_tmp = [
                *a_ptr.add(ki * a_stride),
                *a_ptr.add((ki + 1) * a_stride),
                *a_ptr.add((ki + 2) * a_stride),
                *a_ptr.add((ki + 3) * a_stride),
                *a_ptr.add((ki + 4) * a_stride),
                *a_ptr.add((ki + 5) * a_stride),
                *a_ptr.add((ki + 6) * a_stride),
                *a_ptr.add((ki + 7) * a_stride),
            ];
            let av0 = aarch64::vld1q_f32(av_tmp.as_ptr());
            let bv0 = aarch64::vld1q_f32(b.as_ptr().add(ki));
            let av1 = aarch64::vld1q_f32(av_tmp.as_ptr().add(4));
            let bv1 = aarch64::vld1q_f32(b.as_ptr().add(ki + 4));
            sumv0 = aarch64::vfmaq_f32(sumv0, av0, bv0);
            sumv1 = aarch64::vfmaq_f32(sumv1, av1, bv1);
        }

        let mut sum = aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1);
        for ki in k_rounded..k {
            sum += a[a_base + ki * a_stride] * b[ki];
        }
        sum
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn dot_product_f32_simd(
    _a: &[f32],
    _a_base: usize,
    _a_stride: usize,
    _k: usize,
    _b: &[f32],
) -> f32 {
    unimplemented!("non aarch64 is not supported yet")
}
