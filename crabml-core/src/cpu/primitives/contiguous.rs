use std::borrow::Cow;

use crate::cpu::CpuTensorBuf;
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

pub fn contiguous_buf<T: Copy + Send + Sync>(
    a: &[T],
    b: &mut [T],
    shape: &[usize],
    stride: &[usize],
) {
    match stride.len() {
        2 => contiguous_buf_2d(a, b, shape, stride),
        3 => contiguous_buf_3d(a, b, shape, stride),
        _ => unreachable!(),
    }
}

pub fn contiguous_buf_2d<T: Copy + Send + Sync>(
    a: &[T],
    b: &mut [T],
    shape: &[usize],
    stride: &[usize],
) {
    let mut index = 0;
    for i in 0..shape[0] {
        let offset = i * stride[0];
        for j in 0..shape[1] {
            b[index] = a[offset + j * stride[1]];
            index += 1;
        }
    }
}

pub fn contiguous_buf_3d<T: Copy + Send + Sync>(
    a: &[T],
    b: &mut [T],
    shape: &[usize],
    stride: &[usize],
) {
    let mut index = 0;
    for i in 0..shape[0] {
        let offset_i = i * stride[0];
        for j in 0..shape[1] {
            let offset_j = j * stride[1];
            for k in 0..shape[2] {
                b[index] = a[offset_i + offset_j + k * stride[2]];
                index += 1;
            }
        }
    }
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
