use half::f16;
use rayon::prelude::*;

use crate::backends::cpu::buf::buf_f16::quantize_f32_f16;
use crate::backends::cpu::buf::buf_f16::vec_dot_f16_f16;
use crate::backends::cpu::buf::buf_f16::vec_dot_f16_f16_strided;
use crate::backends::cpu::buf::buf_f32::vec_dot_f32_f32_strided;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDevice;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::TensorStrider;

// (b, m, k) @ (b, k, ) -> (b, m, )
// a is allowed to be not contiguous, but not quantized
pub fn batch_matmul_vec<'a>(
    device: &CpuTensorDeviceRef<'a>,
    a: &CpuTensorBuf<'a>,
    b: &CpuTensorBuf<'a>,
    c: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(strider1.shape().len() == 3);
    assert!(strider2.shape().len() == 2);
    // assert!(strider1.shape()[0] == strider2.shape()[0]);
    assert!(strider1.shape()[2] == strider2.shape()[1]);
    assert!(strider2.is_contiguous());

    let bufc = c.as_f32_mut();

    let batch = strider1.shape()[0];
    let m = strider1.shape()[1];
    let k = strider1.shape()[2];
    let bi_stride = strider1.strides()[0];
    let mi_stride = strider1.strides()[1];
    let ki_stride = strider1.strides()[2];

    match a {
        CpuTensorBuf::F32(bufa) => {
            let bufb = b.as_f32_ref();
            batch_matmul_vec_f32(
                device, bufa, bufb, bufc, batch, m, k, bi_stride, mi_stride, ki_stride,
            );
        }
        CpuTensorBuf::F16(bufa) => {
            let bufb = b.as_f32_ref();
            let bufb = quantize_f32_f16(bufb);
            batch_matmul_vec_f16(
                device, bufa, &bufb, bufc, batch, m, k, bi_stride, mi_stride, ki_stride,
            );
        }
        _ => return Err((ErrorKind::TensorError, "a must be f32 or 16").into()),
    };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_vec_f32(
    _device: &CpuTensorDeviceRef,
    abuf: &[f32],     // Batch x M x K
    bbuf: &[f32],     // Batch x K
    cbuf: &mut [f32], // Batch x M
    seq: usize,
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
            (bi % seq) * bi_stride + mi * mi_stride,
            ki_stride,
            k,
            &bbuf[bi * k..(bi + 1) * k],
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_vec_f16(
    device: &CpuTensorDeviceRef,
    abuf: &[f16],     // Batch x M x K
    bbuf: &[f16],     // Batch x K
    cbuf: &mut [f32], // Batch x M
    batch: usize,
    m: usize,
    k: usize,
    a_stride0: usize,
    a_stride1: usize,
    a_stride2: usize,
) {
    if a_stride2 == 1 {
        let _t = device.metrics.batch_matmul1_walltime.track();
        cbuf.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
            let mi = i % m;
            let bi = (i - mi) / m;
            *bufcp = vec_dot_f16_f16(
                abuf,
                (bi % batch) * a_stride0 + mi * a_stride1,
                &bbuf[bi * k..(bi + 1) * k],
                0,
                k,
            );
        });
    } else {
        let _t = device.metrics.batch_matmul2_walltime.track();
        cbuf.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
            let mi = i % m;
            let bi = (i - mi) / m;
            *bufcp = vec_dot_f16_f16_strided(
                abuf,
                (bi % batch) * a_stride0 + mi * a_stride1,
                a_stride2,
                k,
                &bbuf[bi * k..(bi + 1) * k],
            );
        })
    }
}
