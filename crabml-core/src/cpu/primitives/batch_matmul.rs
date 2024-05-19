use half::f16;

use crate::cpu::buf::buf_f16::quantize_f32_f16;
use crate::cpu::buf::buf_f16::vec_dot_f16_f16;
use crate::cpu::buf::buf_f16::vec_fma_f16_f16;
use crate::cpu::buf::CpuTensorBuf;
use crate::cpu::CpuTensorDeviceRef;
use crate::gguf::GGMLType;
use crate::tensor::TensorStrider;

/// A (b, m, n) @ B (b, k, n) -> C (b, m, n)
///
/// A is expected to be contiguous, B is allowed to be strided, but B should
/// be contiguous on the K dimension or N dimension.
pub fn batch_matmul<'a>(
    _device: &CpuTensorDeviceRef<'a>,
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) {
    assert!(strider1.dims() == 3);
    assert!(strider2.dims() == 3);
    assert!(strider1.is_contiguous());
    assert!(strider2.strides()[1] == 1 || strider2.strides()[2] == 1);
    assert!(bufa.dtype() == GGMLType::F32 || bufa.dtype() == GGMLType::F16);
    assert!(bufb.dtype() == GGMLType::F32 || bufb.dtype() == GGMLType::F16);

    match bufb {
        CpuTensorBuf::F32(bufb) => batch_matmul_naive_f32(
            bufa.as_f32_ref(),
            bufb,
            bufc.as_f32_mut(),
            strider1,
            strider2,
        ),
        CpuTensorBuf::F16(bufb) => {
            let bufa = quantize_f32_f16(bufa.as_f32_ref());
            batch_matmul_simd_f16(&bufa, bufb, bufc.as_f32_mut(), strider1, strider2)
        }
        _ => unreachable!(),
    }
}

// TODO: use vec_dot and vec_fma to optimize this function
fn batch_matmul_naive_f32(
    bufa: &[f32],     // b x m x k
    bufb: &[f32],     // b x k x n
    bufc: &mut [f32], // b x m x n
    stride1: &TensorStrider,
    stride2: &TensorStrider,
) {
    let (a_batch, b_batch) = (stride1.shape()[0], stride2.shape()[0]);
    assert!(a_batch >= b_batch);
    let (m, k, n) = (stride1.shape()[1], stride1.shape()[2], stride2.shape()[2]);
    for bi in 0..a_batch {
        for mi in 0..m {
            for ni in 0..n {
                for ki in 0..k {
                    bufc[bi * (m * n) + mi * n + ni] += bufa[bi * stride1.strides()[0]
                        + mi * stride1.strides()[1]
                        + ki * stride1.strides()[2]]
                        * bufb[(bi % b_batch) * stride2.strides()[0]
                            + ki * stride2.strides()[1]
                            + ni * stride2.strides()[2]];
                }
            }
        }
    }
}

fn batch_matmul_simd_f16(
    bufa: &[f16],     // bA x m x k
    bufb: &[f16],     // bB x k x n, bA is multiple of bB
    bufc: &mut [f32], // bA x m x n
    stride1: &TensorStrider,
    stride2: &TensorStrider,
) {
    let (a_batch, b_batch) = (stride1.shape()[0], stride2.shape()[0]);
    assert!(a_batch >= b_batch);
    let (m, k, n) = (stride1.shape()[1], stride1.shape()[2], stride2.shape()[2]);
    let (stride_bb, stride_bk, stride_bn) = (
        stride2.strides()[0],
        stride2.strides()[1],
        stride2.strides()[2],
    );

    // On Grouped Query Attention, the batch size of A is always a multiple of the batch size of B.
    // batch demension of A / batch_broadcast = batch dimension of B.
    let batch_broadcast = a_batch / b_batch;

    // matrix A is always row-wise contiguous, matrix B should be contiguous on the K
    // dimension or N dimension.
    // if matrix B is contiguous on the k dimension, then we use vec_dot_f16_f16
    // if matrix B is contiguous on the n dimension, then we use vec_fma_f16_f16
    if stride_bk == 1 {
        bufc.iter_mut().enumerate().for_each(|(i, bufcp)| {
            let ni = i % n;
            let mi = (i - ni) / n % m;
            let bi_a = (i - ni - mi * n) / (m * n);
            let offset_a = bi_a * (m * k) + mi * k;
            let offset_b = (bi_a / batch_broadcast) * stride_bb + ni * stride_bn;
            *bufcp = vec_dot_f16_f16(bufa, offset_a, &bufb[offset_b..offset_b + k], 0, k);
        });
    } else if stride_bn == 1 {
        let mut tmpc = vec![f16::ZERO; a_batch * m * n]; // TODO: avoid allocation
        for bi_a in 0..a_batch {
            for mi in 0..m {
                for ki in 0..k {
                    let offset_a = bi_a * (m * k) + mi * k + ki;
                    let offset_b = (bi_a / batch_broadcast) * stride_bb + ki * stride_bk;
                    let offset_c = bi_a * (m * n) + mi * n;
                    vec_fma_f16_f16(
                        &bufb[offset_b..offset_b + n],
                        bufa[offset_a],
                        &mut tmpc[offset_c..offset_c + n],
                        0,
                        n,
                    );
                }
            }
        }

        bufc.iter_mut().zip(tmpc.iter()).for_each(|(c, tmp)| {
            *c = tmp.to_f32();
        });
    } else {
        unreachable!()
    }
}
