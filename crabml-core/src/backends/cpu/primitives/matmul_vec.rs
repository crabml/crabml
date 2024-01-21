use std::borrow::Cow;

use rayon::prelude::*;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::buf::CpuTensorBufVecDot;
use crate::error::Result;
use crate::tensor::TensorStrider;

// matmul_vec is an implementation of GEMV: A (m,k) @ B (k,) -> xout (m,).
// A is allowed to be not contiguous, and quantized
pub fn matmul_vec<'a>(
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

    let m = strider1.shape()[0];
    let k = strider1.shape()[1];

    let ok = maybe_matmul_vec_simd(bufa, bufb, bufc, &strider1);
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

    let m_stride = strider1.strides()[0];
    let k_stride = strider1.strides()[1];

    for mi in 0..m {
        let mut sum = 0.0;
        for ki in 0..k {
            sum += a[mi * m_stride + ki * k_stride] * b[ki];
        }
        out[mi] = sum;
    }

    return Ok(());
}

pub fn maybe_matmul_vec_simd<'a, 'b: 'a>(
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'b>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
) -> bool {
    if !strider1.is_contiguous() {
        return false;
    }
    let bufc = match bufc {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => return false,
    };

    match (bufa, bufb) {
        (CpuTensorBuf::Q8_0(bufa), CpuTensorBuf::F32(bufb)) => {
            if bufa.len() % 32 != 0 {
                return false;
            }
            matmul_vec_generic_xxx_f32_simd(bufa, bufb, bufc);
        }
        (CpuTensorBuf::F32(bufa), CpuTensorBuf::F32(bufb)) => {
            if bufa.len() % 32 != 0 {
                return false;
            }
            matmul_vec_generic_xxx_f32_simd(bufa, bufb, bufc);
        }
        _ => return false,
    };

    true
}

pub fn matmul_vec_generic_xxx_f32_simd<'a, T: CpuTensorBufVecDot + Sync>(
    a: &T,
    b: &[f32],
    c: &mut [f32],
) {
    // a: [m, k]
    // b: [k]
    // out: [m]
    let k = b.len();
    c.par_iter_mut().enumerate().for_each(|(mi, cp)| {
        let offset = mi * k;
        *cp = a.vec_dot_f32(offset, b);
    });
}
