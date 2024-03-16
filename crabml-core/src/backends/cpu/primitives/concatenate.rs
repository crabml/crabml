use std::ops::MulAssign;

use half::vec;

use crate::backends::cpu::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

pub fn concatenate_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    todo!()
}

pub fn concatenate_1d<'a, T: Copy>(
    buf1: &mut [T],
    buf2: &[T],
    shape1: &[usize],
    shape2: &[usize],
    strides1: &[usize],
    strides2: &[usize],
) -> Result<()> {
    let buf1_offset = shape1[0];

    for x in 0..shape2[0] {
        let buf1_offset = x * strides1[0];
        let buf2_offset = x * strides2[0];
        buf1[buf1_offset] = buf2[buf2_offset];
    }
    Ok(())
}

pub fn concatenate_2d<'a, T: Copy + MulAssign>(
    buf1: &mut [T],
    buf2: &[T],
    shape1: &[usize],
    shape2: &[usize],
    strides1: &[usize],
    strides2: &[usize],
    axis: usize,
) -> Result<Vec<usize>> {
    let buf1_base = shape1[axis] * strides1[axis];

    for x in 0..shape2[0] {
        for y in 0..shape2[1] {
            let buf1_offset = buf1_base + x * strides1[0] + y * strides1[1];
            let buf2_offset = x * strides2[0] + y * strides2[1];
            buf1[buf1_offset] = buf2[buf2_offset];
        }
    }

    let mut new_shape = shape1.to_vec();
    new_shape[axis] += shape2[axis];
    Ok(new_shape)
}

pub fn concatenate_3d<'a, T: Copy>(
    buf1: &mut [T],
    buf2: &[T],
    shape1: &[usize],
    shape2: &[usize],
    strides1: &[usize],
    strides2: &[usize],
    axis: usize,
) -> Result<Vec<usize>> {
    let buf1_offset = shape1[axis] * strides1[axis];

    for x in 0..shape2[0] {
        for y in 0..shape2[1] {
            for z in 0..shape2[2] {
                let buf1_offset = buf1_offset + x * strides1[0] + y * strides1[1] + z * strides1[2];
                let buf2_offset = x * strides2[0] + y * strides2[1] + z * strides2[2];
                buf1[buf1_offset] = buf2[buf2_offset];
            }
        }
    }

    let mut new_shape = shape1.to_vec();
    new_shape[axis] += shape2[axis];
    Ok(new_shape)
}

#[cfg(test)]
mod test {
    #[test]
    fn test_concate_2d() {}
}
