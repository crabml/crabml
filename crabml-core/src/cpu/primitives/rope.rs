use std::borrow::Cow;

use crate::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::RopeMode;
use crate::tensor::TensorStrider;

// only support f32 yet
// TODO: support f16
pub fn rope_inplace(
    buf1: &mut CpuTensorBuf<'_>,
    strider1: &TensorStrider,
    mode: RopeMode,
    pos: usize,
    rope_dim: usize,
) -> Result<()> {
    assert!(strider1.is_contiguous());
    assert!(strider1.dims() == 2 || strider1.dims() == 3);

    let buf = match buf1 {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };

    let (n_batch, bi_stride, head_dim) = if strider1.dims() == 2 {
        (1, strider1.len(), strider1.shape()[1])
    } else {
        (
            strider1.shape()[0],
            strider1.strides()[0],
            strider1.shape()[2],
        )
    };

    for bi in 0..n_batch {
        let seq_pos = pos + bi;
        let buf_row = &mut buf[bi * bi_stride..(bi + 1) * bi_stride];
        match mode {
            RopeMode::Llama => rope_llama(buf_row, seq_pos, head_dim, rope_dim),
            RopeMode::Neox => rope_neox(buf_row, seq_pos, head_dim, rope_dim),
        }
    }

    Ok(())
}

fn rope_llama(buf: &mut [f32], pos: usize, head_dim: usize, rope_dim: usize) {
    let theta_scale = 10000_f32.powf(-2.0 / head_dim as f32);
    buf.chunks_exact_mut(head_dim).for_each(|chunk| {
        let mut theta: f32 = pos as f32;
        for i in (0..rope_dim).step_by(2) {
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            theta *= theta_scale;
            unsafe {
                let qp0 = *chunk.get_unchecked(i);
                let qp1 = *chunk.get_unchecked(i + 1);
                *chunk.get_unchecked_mut(i) = qp0 * cos_theta - qp1 * sin_theta;
                *chunk.get_unchecked_mut(i + 1) = qp0 * sin_theta + qp1 * cos_theta;
            }
        }
    });
}

fn rope_neox(buf: &mut [f32], pos: usize, head_dim: usize, rope_dim: usize) {
    buf.chunks_exact_mut(head_dim).for_each(|chunk| {
        for i in 0..rope_dim / 2 {
            let freq_exponents = 2.0 * i as f32 / head_dim as f32;
            let timescale = 10000_f32.powf(freq_exponents);
            let theta = pos as f32 / timescale;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let qp0 = chunk[i];
            let qp1 = chunk[i + head_dim / 2];
            chunk[i] = qp0 * cos_theta - qp1 * sin_theta;
            chunk[i + head_dim / 2] = qp0 * sin_theta + qp1 * cos_theta;
        }
    });
}
