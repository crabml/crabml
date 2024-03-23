use half::f16;
use rayon::prelude::*;

use crate::backends::cpu::buf::buf_f16::quantize_f32_f16;
use crate::backends::cpu::buf::buf_f16::vec_dot_f16_f16;
use crate::backends::cpu::buf::buf_f16::vec_fma_f16_f16;
use crate::backends::cpu::buf::buf_f32::vec_dot_f32_f32_strided;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::tensor::TensorStrider;

/// (m, k) @ (k, ) -> (m, ) => gemv_2d_1d
/// (s, m, k) @ (s, k, ) -> (s, m, )  => bmm_3d_2d
/// (b, s, m, k) @ (b, s, k) -> (b, s, m) => bmm_4d_3d
///
/// the first dimension of A is expected to be always contiguous, and the last two dimensions
/// of A is allowed to be column-wise contiguous.
pub fn gemv<'a>(
    device: &CpuTensorDeviceRef<'a>,
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) {
    assert!(strider1.dims() == 4 || strider1.dims() == 3 || strider1.dims() == 2);
    assert!(strider2.dims() == strider1.dims() - 1);
    assert!(strider2.is_contiguous());

    // on the case of 2d x 1d matmul, we can expect it to be always contiguous.
    if strider1.dims() == 2 {
        assert!(strider1.is_contiguous());
        assert!(strider1.shape()[1] == strider2.shape()[0]);
        let (m, k) = (strider1.shape()[0], strider1.shape()[1]);
        let (mi_stride, ki_stride) = (strider1.strides()[0], strider1.strides()[1]);
        gemv_dense_3d_2d(
            device, bufa, bufb, bufc, 1, 1, m, k, 0, mi_stride, ki_stride,
        );
        return;
    }

    // 3d and 4d matmul could be handled by the same function
    let (a_batch, b_batch) = if strider1.dims() == 3 {
        // (s, m, k) @ (s, k) -> (s, m)
        (strider1.shape()[0], strider2.shape()[0])
    } else if strider1.dims() == 4 {
        // (b, s, m, k) @ (b, s, k) -> (b, s, m)
        // the b and s dimensions are considered as contiguous, and the m and k dimensions maybe not.
        // so the we can merge the b and s dimensions into a single batch dimension.
        (
            strider1.shape()[0] * strider1.shape()[1],
            strider2.shape()[0] * strider2.shape()[1],
        )
    } else {
        unreachable!();
    };

    // m, k are always the last two dimensions of strider1
    let (m, k) = {
        let shape_tail = &strider1.shape()[strider1.dims() - 2..strider1.dims()];
        (shape_tail[0], shape_tail[1])
    };
    // strides for the last three dimensions
    let (bi_stride, mi_stride, ki_stride) = {
        let strides_tail = &strider1.strides()[strider1.dims() - 3..strider1.dims()];
        (strides_tail[0], strides_tail[1], strides_tail[2])
    };

    match bufa {
        CpuTensorBuf::F32(bufa) => {
            let bufc = bufc.as_f32_mut();
            let bufb = bufb.as_f32_ref();
            gemv_3d_2d_f32(
                device, bufa, bufb, bufc, a_batch, b_batch, m, k, bi_stride, mi_stride, ki_stride,
            );
        }
        CpuTensorBuf::F16(bufa) => {
            let bufb = bufb.as_f32_ref();
            let bufc = bufc.as_f32_mut();
            let bufb = quantize_f32_f16(bufb);
            gemv_3d_2d_f16(
                device, bufa, &bufb, bufc, a_batch, b_batch, m, k, bi_stride, mi_stride, ki_stride,
            );
        }
        bufa => {
            // the quantized matmul is always dense
            assert!(strider1.is_contiguous());
            gemv_dense_3d_2d(
                device, bufa, bufb, bufc, a_batch, b_batch, m, k, bi_stride, mi_stride, ki_stride,
            );
        }
    }
}

fn gemv_dense_3d_2d(
    _device: &CpuTensorDeviceRef,
    bufa: &CpuTensorBuf,
    bufb: &CpuTensorBuf,
    bufc: &mut CpuTensorBuf,
    a_batch: usize,
    _b_batch: usize, // b_batch is multiple of a_batch
    m: usize,
    k: usize,
    bi_stride: usize,
    mi_stride: usize,
    _ki_stride: usize,
) {
    assert!(bufc.len() % 4 == 0);

    let bufc = bufc.as_f32_mut();
    let bufb = &bufb.quantize(bufa.vec_dot_rhs_dtype()).unwrap();
    bufc.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(cn, cp)| {
            // a: b x m x k
            // b: b x k
            // c: b x m
            let mi = cn * 4 % m;
            let bi = (cn * 4 - mi) / m;
            let bi_offset = (bi % a_batch) * bi_stride;
            cp[0] = bufa.vec_dot(bi_offset + mi * mi_stride, bufb, 0, k);
            cp[1] = bufa.vec_dot(bi_offset + (mi + 1) * mi_stride, bufb, 0, k);
            cp[2] = bufa.vec_dot(bi_offset + (mi + 2) * mi_stride, bufb, 0, k);
            cp[3] = bufa.vec_dot(bi_offset + (mi + 3) * mi_stride, bufb, 0, k);
        });
}

#[allow(clippy::too_many_arguments)]
fn gemv_3d_2d_f32(
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
fn gemv_3d_2d_f16(
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
