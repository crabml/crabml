use std::time::Duration;
use std::time::Instant;

use rayon::prelude::*;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// matmul_vec is an implementation of GEMV: A (m,k) @ B (k,) -> xout (m,).
// A is allowed to be not contiguous and quantized
pub fn matmul_vec<'a>(
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(strider1.shape().len() == 2);
    assert!(strider2.shape().len() == 1);
    assert!(strider1.shape()[1] == strider2.shape()[0]);
    assert!(strider2.is_contiguous());

    // if the input is contiguous, we can use SIMD to accelerate the computation
    if strider1.is_contiguous() && bufa.len() % 32 == 0 {
        gemv_simd_f32(bufa, bufb, bufc);
        return Ok(());
    }

    // fall back to the naive implementation if stride1 is not contiguous
    gemv_naive_f32(bufa, bufb, bufc, strider1);
    return Ok(());
}

fn gemv_naive_f32<'a>(
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
) {
    let c = bufc.as_f32_mut();
    let a = bufa.as_f32_ref();
    let b = bufb.as_f32_ref();
    let m_stride = strider1.strides()[0];
    let k_stride = strider1.strides()[1];
    let m = strider1.shape()[0];
    let k = strider1.shape()[1];

    for mi in 0..m {
        let mut sum = 0.0;
        for ki in 0..k {
            sum += a[mi * m_stride + ki * k_stride] * b[ki];
        }
        c[mi] = sum;
    }
}

fn gemv_simd_f32<'a>(
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
) {
    let start_time = Instant::now();
    assert!(bufa.len() % 32 == 0);

    let bufc = bufc.as_f32_mut();
    let bufb = bufb.as_f32_ref();

    let k = bufb.len();
    bufc.par_chunks_mut(8).enumerate().for_each(|(cn, cp)| {
        let mi = cn * 8;
        cp[0] = bufa.vec_dot_f32(mi * k, bufb);
        cp[1] = bufa.vec_dot_f32((mi + 1) * k, bufb);
        cp[2] = bufa.vec_dot_f32((mi + 2) * k, bufb);
        cp[3] = bufa.vec_dot_f32((mi + 3) * k, bufb);
        cp[4] = bufa.vec_dot_f32((mi + 4) * k, bufb);
        cp[5] = bufa.vec_dot_f32((mi + 5) * k, bufb);
        cp[6] = bufa.vec_dot_f32((mi + 6) * k, bufb);
        cp[7] = bufa.vec_dot_f32((mi + 7) * k, bufb);
    });

    println!(
        "gemv_simd_f32: {:?} ms",
        start_time.elapsed().as_secs_f64() * 1000.0
    );
}
