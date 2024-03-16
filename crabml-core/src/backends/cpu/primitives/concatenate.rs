use std::borrow::Cow;

use half::f16;

use crate::backends::cpu::CpuTensorBuf;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::TensorStrider;

pub fn concatenate_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
    axis: usize,
) -> Result<TensorStrider> {
    let new_shape = match (buf1, buf2) {
        (CpuTensorBuf::F32(Cow::Owned(buf1)), CpuTensorBuf::F32(buf2)) => concatenate_inner(
            buf1,
            buf2,
            strider1.shape(),
            strider2.shape(),
            strider1.strides(),
            strider2.strides(),
            axis,
            |x| x,
        )?,
        (CpuTensorBuf::F16(Cow::Owned(buf1)), CpuTensorBuf::F16(buf2)) => concatenate_inner(
            buf1,
            buf2,
            strider1.shape(),
            strider2.shape(),
            strider1.strides(),
            strider2.strides(),
            axis,
            |x| x,
        )?,
        (CpuTensorBuf::F16(Cow::Owned(buf1)), CpuTensorBuf::F32(buf2)) => concatenate_inner(
            buf1,
            &buf2,
            strider1.shape(),
            strider2.shape(),
            strider1.strides(),
            strider2.strides(),
            axis,
            |x| f16::from_f32(x),
        )?,
        (buf1, buf2) => {
            return Err((
                ErrorKind::TensorError,
                format!("can not concatenate {} and {}", buf1.dtype(), buf2.dtype()),
            )
                .into());
        }
    };
    strider1.resize(&new_shape)
}

pub fn concatenate_inner<A, B: Copy>(
    buf1: &mut [A],
    buf2: &[B],
    shape1: &[usize],
    shape2: &[usize],
    strides1: &[usize],
    strides2: &[usize],
    axis: usize,
    f: impl Fn(B) -> A,
) -> Result<Vec<usize>> {
    match shape1.len() {
        1 => concatenate_1d(buf1, buf2, shape1, shape2, strides1, strides2, f),
        2 => concatenate_2d(buf1, buf2, shape1, shape2, strides1, strides2, axis, f),
        3 => concatenate_3d(buf1, buf2, shape1, shape2, strides1, strides2, axis, f),
        _ => unreachable!(),
    }
}

pub fn concatenate_1d<A, B: Copy>(
    buf1: &mut [A],
    buf2: &[B],
    shape1: &[usize],
    shape2: &[usize],
    strides1: &[usize],
    strides2: &[usize],
    f: impl Fn(B) -> A,
) -> Result<Vec<usize>> {
    let buf1_base = shape1[0] * strides1[0];
    for x in 0..shape2[0] {
        let buf1_offset = buf1_base + x * strides1[0];
        let buf2_offset = x * strides2[0];
        buf1[buf1_offset] = f(buf2[buf2_offset]);
    }
    Ok(vec![shape1[0] + shape2[0]])
}

pub fn concatenate_2d<A, B: Copy>(
    buf1: &mut [A],
    buf2: &[B],
    shape1: &[usize],
    shape2: &[usize],
    strides1: &[usize],
    strides2: &[usize],
    axis: usize,
    f: impl Fn(B) -> A,
) -> Result<Vec<usize>> {
    let buf1_base = shape1[axis] * strides1[axis];

    for x in 0..shape2[0] {
        for y in 0..shape2[1] {
            let buf1_offset = buf1_base + x * strides1[0] + y * strides1[1];
            let buf2_offset = x * strides2[0] + y * strides2[1];
            buf1[buf1_offset] = f(buf2[buf2_offset]);
        }
    }

    let mut new_shape = shape1.to_vec();
    new_shape[axis] += shape2[axis];
    Ok(new_shape)
}

pub fn concatenate_3d<A, B: Copy>(
    buf1: &mut [A],
    buf2: &[B],
    shape1: &[usize],
    shape2: &[usize],
    strides1: &[usize],
    strides2: &[usize],
    axis: usize,
    f: impl Fn(B) -> A,
) -> Result<Vec<usize>> {
    let buf1_offset = shape1[axis] * strides1[axis];

    for x in 0..shape2[0] {
        for y in 0..shape2[1] {
            for z in 0..shape2[2] {
                let buf1_offset = buf1_offset + x * strides1[0] + y * strides1[1] + z * strides1[2];
                let buf2_offset = x * strides2[0] + y * strides2[1] + z * strides2[2];
                buf1[buf1_offset] = f(buf2[buf2_offset]);
            }
        }
    }

    let mut new_shape = shape1.to_vec();
    new_shape[axis] += shape2[axis];
    Ok(new_shape)
}

#[cfg(test)]
mod test {
    use super::concatenate_2d;
    use super::Result;

    #[test]
    fn test_concate_2d() -> Result<()> {
        // v1: layout: 2x3
        // 1 0 0
        // 4 0 0
        let mut buf1 = vec![1, 0, 0, 4, 0, 0];
        let shape1: Vec<usize> = vec![2, 1];
        let strides1: Vec<usize> = vec![3, 1];
        let buf2 = vec![2, 5];
        let shape2: Vec<usize> = vec![2, 1];
        let strides2: Vec<usize> = vec![1, 1];
        let new_shape1 = concatenate_2d(
            &mut buf1,
            &buf2,
            &shape1,
            &shape2,
            &strides1,
            &strides2,
            1,
            |a| a,
        )?;
        assert_eq!(buf1, vec![1, 2, 0, 4, 5, 0]);
        assert_eq!(new_shape1, vec![2, 2]);

        // v1: layout: 2x3
        // 1 2 3
        // 0 0 0
        // v2: layout: 1x3
        let mut buf1 = vec![1, 2, 3, 0, 0, 0];
        let shape1: Vec<usize> = vec![1, 3];
        let strides1: Vec<usize> = vec![3, 1];
        let buf2 = vec![4, 5, 6];
        let shape2: Vec<usize> = vec![1, 3];
        let strides2: Vec<usize> = vec![3, 1];
        let new_shape1 = concatenate_2d(
            &mut buf1,
            &buf2,
            &shape1,
            &shape2,
            &strides1,
            &strides2,
            0,
            |a| a,
        )?;
        assert_eq!(buf1, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(new_shape1, vec![2, 3]);

        Ok(())
    }
}
