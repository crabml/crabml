use std::borrow::Cow;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// (b, m, k) @ (b, k, ) -> (b, m, )
// a is allowed to be not contiguous, but not quantized
pub fn batch_matmul_vec<'a, 'b>(
    a: &CpuTensorBuf<'a>,
    b: &CpuTensorBuf<'b>,
    c: &mut CpuTensorBuf<'b>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()>
where
    'b: 'a,
{
    assert!(strider1.shape().len() == 3);
    assert!(strider2.shape().len() == 2);
    assert!(strider1.shape()[0] == strider2.shape()[0]);
    assert!(strider1.shape()[2] == strider2.shape()[1]);
    assert!(strider2.is_contiguous());

    let bufa = match a {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };
    let bufb = match b {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };
    let bufc = match c {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };

    let batch = strider1.shape()[0];
    let m = strider1.shape()[1];
    let k = strider1.shape()[2];
    let bi_stride = strider1.strides()[0];
    let mi_stride = strider1.strides()[1];
    let ki_stride = strider1.strides()[2];

    for bi in 0..batch {
        for mi in 0..m {
            let mut sum = 0.0;
            for ki in 0..k {
                sum += bufa[bi * bi_stride + mi * mi_stride + ki * ki_stride] * bufb[bi * k + ki];
            }
            bufc[bi * m + mi] = sum;
        }
    }
    return Ok(());
}
