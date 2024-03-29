use rayon::prelude::*;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::tensor::TensorStrider;

/// only dense GEMV is supported
/// (m, k) @ k -> (m, )
/// (m, k) @ (b, k) -> (b, m)
pub fn matmul_vec<'a>(
    device: &CpuTensorDeviceRef<'a>,
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) {
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());
    assert!(strider1.shape().last() == strider2.shape().last());

    let (m, k) = (strider1.shape()[0], strider1.shape()[1]);
    gemv_dense_2d_2d(device, bufa, bufb, bufc, m, k);
}

#[allow(clippy::too_many_arguments)]
fn gemv_dense_2d_2d(
    device: &CpuTensorDeviceRef,
    bufa: &CpuTensorBuf,
    bufb: &CpuTensorBuf,
    bufc: &mut CpuTensorBuf,
    m: usize,
    k: usize,
) {
    assert!(bufc.len() % 4 == 0);

    let bufc = bufc.as_f32_mut();
    let bufb = &bufb.quantize(bufa.vec_dot_rhs_dtype()).unwrap();
    let chunk = 16;
    let metrics = device.metrics.clone();
    let _t = metrics.matmul_walltime.track();
    bufc.par_chunks_exact_mut(chunk)
        .enumerate()
        .for_each(|(cn, cp)| {
            // a: m x k
            // b: b x k
            // c: b x m
            let mi = cn * chunk % m;
            let bi = (cn * chunk - mi) / m;
            for (i, cval) in cp.iter_mut().enumerate() {
                *cval = bufa.vec_dot((mi + i) * k, bufb, bi * k, k);
            }
        });
}
