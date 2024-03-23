use half::f16;
use rayon::prelude::*;

use crate::backends::cpu::buf::buf_f16::quantize_f32_f16;
use crate::backends::cpu::buf::buf_f16::vec_dot_f16_f16;
use crate::backends::cpu::buf::buf_f16::vec_fma_f16_f16;
use crate::backends::cpu::buf::buf_f32::vec_dot_f32_f32_strided;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::tensor::TensorStrider;

/// A (b, m, n) @ B (b, k, n) -> C (b, m, n)
/// A (b, m, n) @ B (k, n) -> C (b, m)
///
/// A is expected to be contiguous, B is allowed to be strided.
pub fn batch_matmul<'a>(
    _device: &CpuTensorDeviceRef<'a>,
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) {
    assert!(strider1.dims() == 3);
    assert!(strider2.dims() == 3 || strider2.dims() == 2);
    assert!(strider1.is_contiguous());

    let strider2 = if strider2.dims() == 3 {
        strider2.clone()
    } else {
        strider2
            .reshape(vec![strider2.shape()[0], strider2.shape()[1], 1])
            .unwrap()
    };

    match bufb {
        CpuTensorBuf::F32(bufb) => batch_matmul_naive_f32(
            bufa.as_f32_ref(),
            bufb,
            bufc.as_f32_mut(),
            strider1,
            &strider2,
        ),
        CpuTensorBuf::F16(bufb) => {
            let bufa = quantize_f32_f16(bufa.as_f32_ref());
            batch_matmul_naive_f16(&bufa, bufb, bufc.as_f32_mut(), strider1, &strider2)
        }
        _ => unreachable!(),
    }
}

fn batch_matmul_naive_f32(
    bufa: &[f32],     // b x m x k
    bufb: &[f32],     // b x k x n
    bufc: &mut [f32], // b x m x n
    stride1: &TensorStrider,
    stride2: &TensorStrider,
) {
    let (a_batch, b_batch) = (stride1.shape()[0], stride2.shape()[0]);
    let (m, k, n) = (stride1.shape()[1], stride1.shape()[2], stride2.shape()[2]);
    for bi in 0..b_batch {
        for mi in 0..m {
            for ni in 0..n {
                for ki in 0..k {
                    bufc[bi * (m * n) + mi * n + ni] += bufa[(bi % a_batch) * stride1.strides()[0]
                        + mi * stride1.strides()[1]
                        + ki * stride1.strides()[2]]
                        * bufb[bi * stride2.strides()[0]
                            + ki * stride2.strides()[1]
                            + ni * stride2.strides()[2]];
                }
            }
        }
    }
}

fn batch_matmul_naive_f16(
    bufa: &[f16],     // b x m x k
    bufb: &[f16],     // b x k x n
    bufc: &mut [f32], // b x m x n
    stride1: &TensorStrider,
    stride2: &TensorStrider,
) {
    let (a_batch, b_batch) = (stride1.shape()[0], stride2.shape()[0]);
    let (m, k, n) = (stride1.shape()[1], stride1.shape()[2], stride2.shape()[2]);
    for bi in 0..b_batch {
        for mi in 0..m {
            for ni in 0..n {
                for ki in 0..k {
                    bufc[bi * (m * n) + mi * n + ni] += (bufa[(bi % a_batch)
                        * stride1.strides()[0]
                        + mi * stride1.strides()[1]
                        + ki * stride1.strides()[2]]
                        * bufb[bi * stride2.strides()[0]
                            + ki * stride2.strides()[1]
                            + ni * stride2.strides()[2]])
                        .to_f32();
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemv_strided_3d_2d_f32(
    _device: &CpuTensorDeviceRef,
    abuf: &[f32],     // a_batch x M x K
    bbuf: &[f32],     // b_batch x K
    cbuf: &mut [f32], // b_batch x M
    a_batch: usize,
    _b_batch: usize,
    m: usize,
    k: usize,
    bi_stride: usize,
    mi_stride: usize,
    ki_stride: usize,
) {
    cbuf.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
        let mi = i % m;
        let bi = (i - mi) / m;
        *bufcp = vec_dot_f32_f32_strided(
            abuf,
            (bi % a_batch) * bi_stride + mi * mi_stride,
            ki_stride,
            k,
            &bbuf[bi * k..(bi + 1) * k],
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn gemv_strided_3d_2d_f16(
    device: &CpuTensorDeviceRef,
    abuf: &[f16],     // Batch x M x K
    bbuf: &[f16],     // Batch x K
    cbuf: &mut [f32], // Batch x M
    a_batch: usize,
    b_batch: usize, // b_batch is multiple of a_batch
    m: usize,
    k: usize,
    a_stride0: usize,
    a_stride1: usize,
    a_stride2: usize,
) {
    let mut tmpc = vec![f16::ZERO; b_batch * m]; // TODO: avoid allocation

    // if matrix A is row-wise contiguous, then we can use vec_dot_f16_f16
    // if matrix A is column-wise contiguous, then we can use vec_fma_f16_f16
    if a_stride2 == 1 {
        let _t = device.metrics.batch_matmul_rowwise_walltime.track();
        tmpc.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
            let mi = i % m;
            let bi = (i - mi) / m;
            *bufcp = f16::from_f32(vec_dot_f16_f16(
                abuf,
                (bi % a_batch) * a_stride0 + mi * a_stride1,
                &bbuf[bi * k..(bi + 1) * k],
                0,
                k,
            ));
        });
    } else {
        let _t = device.metrics.batch_matmul_colwise_walltime.track();
        for bi in 0..b_batch {
            for ki in 0..k {
                vec_fma_f16_f16(
                    abuf,
                    bbuf[bi * k + ki],
                    &mut tmpc[bi * m..],
                    (bi % a_batch) * a_stride0 + ki * a_stride2,
                    m,
                );
            }
        }
    }

    cbuf.iter_mut().zip(tmpc.iter()).for_each(|(c, tmp)| {
        *c = tmp.to_f32();
    });
}
