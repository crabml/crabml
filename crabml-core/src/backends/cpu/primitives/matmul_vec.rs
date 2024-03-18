use rayon::prelude::*;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::Result;
use crate::tensor::TensorStrider;

// matmul_vec is an implementation of GEMV: A (m,k) @ B (k,) -> xout (m,).
// A is allowed to be not contiguous and quantized
pub fn matmul_vec<'a>(
    device: CpuTensorDeviceRef<'a>,
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
    assert!(strider1.is_contiguous());

    // if the input is contiguous, we can use SIMD to accelerate the computation
    if bufa.len() % 32 == 0 {
        gemv_simd(device, bufa, bufb, bufc);
        return Ok(());
    }

    // fall back to the naive implementation if stride1 is not contiguous
    gemv_naive_f32(bufa, bufb, bufc, strider1);
    Ok(())
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

fn gemv_simd<'a>(
    device: CpuTensorDeviceRef<'a>,
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
) {
    assert!(bufa.len() % 32 == 0);
    let metrics = device.metrics().clone();

    let bufc = bufc.as_f32_mut();
    let bufb = {
        let _t = metrics.matmul_quantize_walltime.track();
        &bufb.quantize(bufa.vec_dot_rhs_dtype()).unwrap()
    };

    let _t = metrics.matmul_vec_dot_walltime.track();

    let m = bufc.len();
    let k = bufb.len();
    let threads = 1;
    let chunk_size = m / threads;
    assert!(m % threads == 0);

    bufc.chunks_exact_mut(chunk_size)
        .enumerate()
        .for_each(|(cn, cp)| {
            let mi = cn * chunk_size;
            cp.chunks_exact_mut(4).enumerate().for_each(|(i, cpp)| {
                cpp[0] = bufa.vec_dot((mi + i * 4) * k, bufb, 0, k);
                cpp[1] = bufa.vec_dot((mi + i * 4 + 1) * k, bufb, 0, k);
                cpp[2] = bufa.vec_dot((mi + i * 4 + 2) * k, bufb, 0, k);
                cpp[3] = bufa.vec_dot((mi + i * 4 + 3) * k, bufb, 0, k);
            });
        });
}
