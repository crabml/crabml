use std::borrow::Cow;

use rayon::prelude::*;

use crate::backends::cpu::CpuTensorBuf;
use crate::tensor::TensorStrider;

pub fn contiguous(bufa: &CpuTensorBuf, stride_a: &TensorStrider, bufb: &mut CpuTensorBuf) {
    assert!(stride_a.shape().len() == 2 || stride_a.shape().len() == 3);

    match (bufa, bufb) {
        (CpuTensorBuf::F32(bufa), CpuTensorBuf::F32(Cow::Owned(bufb))) => {
            contiguous_buf(bufa, bufb, stride_a.shape(), stride_a.strides())
        }
        (CpuTensorBuf::F16(bufa), CpuTensorBuf::F16(Cow::Owned(bufb))) => {
            contiguous_buf(bufa, bufb, stride_a.shape(), stride_a.strides())
        }
        _ => unreachable!(),
    }
}

pub fn contiguous_buf<T: Copy>(a: &[T], b: &mut [T], shape: &[usize], stride: &[usize]) {
    match stride.len() {
        2 => contiguous_buf_2d(a, b, shape, stride),
        3 => contiguous_buf_3d(a, b, shape, stride),
        _ => unreachable!(),
    }
}

pub fn contiguous_buf_2d<T: Copy>(a: &[T], b: &mut [T], shape: &[usize], stride: &[usize]) {
    let vthreads = 8;
    let mut chunk = b.len() / vthreads;
    if chunk == 0 {
        chunk = b.len();
    }
    assert!(b.len() % chunk == 0);
    b.chunks_exact_mut(chunk).enumerate().for_each(|(i, bp)| {
        for j in 0..bp.len() {
            let pos = i * chunk + j;
            let idx1 = pos % shape[1];
            let idx0 = (pos - idx1) / shape[1];
            bp[j] = a[idx0 * stride[0] + idx1 * stride[1]];
        }
    });
}

pub fn contiguous_buf_3d<T: Copy>(a: &[T], b: &mut [T], shape: &[usize], stride: &[usize]) {
    let vthreads = 8;
    let mut chunk = b.len() / vthreads;
    if chunk == 0 {
        chunk = b.len();
    }
    assert!(b.len() % chunk == 0);

    b.chunks_exact_mut(chunk).enumerate().for_each(|(i, bp)| {
        for j in 0..bp.len() {
            let pos = i * chunk + j;
            let idx2 = pos % shape[2];
            let idx1 = ((pos - idx2) / shape[2]) % shape[1];
            let idx0 = (pos - idx2 - idx1) / (shape[2] * shape[1]);
            bp[j] = a[idx0 * stride[0] + idx1 * stride[1] + idx2 * stride[2]];
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_contiguous_buf_2d() {
        // 1 3
        // 2 4
        let a = vec![1, 3, 2, 4];
        let mut b = vec![0; 4];
        let shape = vec![2, 2];
        let stride = vec![1, 2];
        super::contiguous_buf_2d(&a, &mut b, &shape, &stride);
        assert_eq!(b, vec![1, 2, 3, 4]);
    }
}
