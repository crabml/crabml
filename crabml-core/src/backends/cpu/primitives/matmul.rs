use std::borrow::Cow;

use rayon::prelude::*;

use crate::backends::cpu::buf::BufVecDot;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

// A (m,k) @ B (k,) -> xout (m,)
pub fn matmul<'a>(
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(strider1.shape().len() == 2);
    assert!(strider2.shape().len() == 1);
    assert!(strider1.shape()[1] == strider2.shape()[0]);
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    let m = strider1.shape()[0];
    let k = strider1.shape()[1];

    let ok = maybe_matmul_vec_2d_1d(bufa, bufb, bufc);
    if ok {
        return Ok(());
    }

    let out = match bufc {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };
    let a = match bufa {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };
    let b = match bufb {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };

    for mi in 0..m {
        let mut sum = 0.0;
        for ki in 0..k {
            sum += a[mi * k + ki] * b[ki];
        }
        out[mi] = sum;
    }

    return Ok(());
}

pub fn maybe_matmul_vec_2d_1d<'a, 'b: 'a>(
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'b>,
    bufc: &mut CpuTensorBuf<'a>,
) -> bool {
    let bufc = match bufc {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => return false,
    };

    match (bufa, bufb) {
        (CpuTensorBuf::Q8_0(bufa), CpuTensorBuf::F32(bufb)) => {
            if bufa.len() % 32 != 0 {
                return false;
            }
            matmul_vec_generic_xxx_f32_2d_1d(bufa, bufb, bufc);
        }
        (CpuTensorBuf::F32(bufa), CpuTensorBuf::F32(bufb)) => {
            if bufa.len() % 32 != 0 {
                return false;
            }
            matmul_vec_generic_xxx_f32_2d_1d(bufa, bufb, bufc);
        }
        _ => return false,
    };

    true
}

pub fn matmul_vec_generic_xxx_f32_2d_1d<'a, T: BufVecDot + Sync>(a: &T, b: &[f32], c: &mut [f32]) {
    // a: [m, k]
    // b: [k]
    // out: [m]
    let k = b.len();
    c.par_iter_mut().enumerate().for_each(|(mi, cp)| {
        let offset = mi * k;
        *cp = a.vec_dot_f32(offset, b);
    });
}
