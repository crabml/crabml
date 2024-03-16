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
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    let buf1_offset = strider1.shape()[0];
    for x in 0..strider2.shape()[0] {
        let buf1_offset = strider1.at_unchecked(&[buf1_offset + x]);
        let buf2_offset = strider2.at_unchecked(&[x]);
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
    let mut buf1_offset = vec![0, 0];
    buf1_offset[axis] = shape1[axis];

    for x in 0..shape2[0] {
        for y in 0..shape2[1] {
            let buf1_pos = (buf1_offset[0] + x) * strides1[0] + (buf1_offset[1] + y) * strides1[1];
            let buf2_pos = x * strides2[0] + y * strides2[1];
            buf1[buf1_pos] = buf2[buf2_pos];
        }
    }

    let mut new_shape = shape1.to_vec();
    new_shape[axis] += shape2[axis];
    Ok(new_shape)
}

pub fn concatenate_3d<'a, T: Copy>(
    buf1: &mut [T],
    buf2: &[T],
    strider1: &TensorStrider,
    strider2: &TensorStrider,
    axis: usize,
) -> Result<()> {
    let mut buf1_offset = vec![0, 0, 0];
    buf1_offset[axis] = strider1.shape()[axis];

    for x in 0..strider2.shape()[0] {
        for y in 0..strider2.shape()[1] {
            for z in 0..strider2.shape()[1] {
                let buf1_pos =
                    strider1.at(&[buf1_offset[0] + x, buf1_offset[1] + y, buf1_offset[2] + z])?;
                let buf2_pos = strider2.at(&[x, y, z])?;
                buf1[buf1_pos] = buf2[buf2_pos];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    #[test]
    fn test_concate_2d() {}
}
