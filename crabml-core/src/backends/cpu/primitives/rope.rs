use std::borrow::Cow;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// only support f32 yet
// TODO: support f16
pub fn rope_inplace(
    buf1: &mut CpuTensorBuf<'_>,
    strider1: &TensorStrider,
    pos: usize,
    rope_dims: usize,
) -> Result<()> {
    assert!(strider1.is_contiguous());
    assert!(strider1.shape().len() == 2);

    let head_dim = strider1.shape()[1];
    let qb = match buf1 {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };

    qb.chunks_exact_mut(head_dim).for_each(|chunk| {
        for i in 0..rope_dims / 2 {
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

    Ok(())
}
