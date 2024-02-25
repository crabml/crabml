use std::borrow::Cow;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// only support f32 yet
// TODO: support f16
pub fn rope_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    pos: usize,
    rope_dims: usize,
) -> Result<()> {
    assert!(strider1.is_contiguous());
    assert!(strider1.shape().len() == 2);

    let n_heads = strider1.shape()[0];
    let head_size = strider1.shape()[1];

    let qb = match buf1 {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };

    let theta_scale = 10000_f32.powf(-2.0 / head_size as f32);

    // apply RoPE rotation for each head
    for h in 0..n_heads {
        let mut theta: f32 = pos as f32;

        for i in 0..rope_dims / 2 {
            theta *= theta_scale;

            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            unsafe {
                let qp = qb.as_mut_ptr().add(h * head_size + i * 2);
                let qp0 = *qp;
                let qp1 = *qp.add(1);
                *qp = qp0 * cos_theta - qp1 * sin_theta;
                *qp.add(1) = qp0 * sin_theta + qp1 * cos_theta;
            }
        }
    }

    Ok(())
}
