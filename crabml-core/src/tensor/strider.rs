use std::cell::RefCell;

use crate::error::ErrorKind;
use crate::error::Result;

#[derive(Clone, Debug, Default)]
pub struct TensorStrider {
    shape: Vec<usize>,
    strides: Vec<usize>,
    repeats: Option<Vec<usize>>,
}

impl TensorStrider {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        let dims = shape.len();
        Self {
            shape,
            strides,
            repeats: None,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn at(&self, idx: &[usize]) -> Result<usize> {
        if idx.len() != self.shape.len() {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "invalid index {:?} for tensor of shape {:?}",
                    idx, self.shape
                ),
            )
                .into());
        }
        for (i, &dim) in idx.iter().enumerate() {
            if dim >= self.shape[i] {
                return Err((
                    ErrorKind::TensorError,
                    format!(
                        "invalid index {:?} for tensor of shape {:?}",
                        idx, self.shape
                    ),
                )
                    .into());
            }
        }

        Ok(self.at_unchecked(idx))
    }

    pub fn at_unchecked(&self, idx: &[usize]) -> usize {
        // if repeats is set, divide the index by the repeat factor.
        // for example a tensor of shape [4, 3] with repeats [2, 1] will
        // be viewed as a tensor of shape [2, 3] with no repeats.
        if let Some(repeats) = &self.repeats {
            let mut offset = 0;
            for (dim, (stride, repeat)) in idx.iter().zip(self.strides.iter().zip(repeats)) {
                offset += dim / repeat * stride;
            }
            return offset;
        }

        let mut offset = 0;
        for (dim, stride) in idx.iter().zip(self.strides.iter()) {
            offset += dim * stride;
        }
        offset
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut pos = vec![0; self.shape.len()];
        let shape = &self.shape;
        (0..self.len()).into_iter().map(move |i| {
            let v = self.at_unchecked(&pos);
            Self::increment_pos(&mut pos, shape);
            v
        })
    }

    /// from the position, iterate until the end of the row / column
    pub fn iter_axis(
        &self,
        pos: &[usize],
        axis: usize,
    ) -> Result<impl Iterator<Item = usize> + '_> {
        let mut pos = pos.to_vec();
        let axis_pos = pos[axis];
        let axis_max = self.shape[axis];

        Ok((axis_pos..axis_max).map(move |i| {
            pos[axis] = i;
            self.at_unchecked(&pos)
        }))
    }

    pub fn into_iter_axis(self, pos: &[usize], axis: usize) -> Result<impl Iterator<Item = usize>> {
        let iter = self.iter_axis_inner(pos, axis)?;
        let iter = iter.map(move |pos| self.at_unchecked(&pos));
        Ok(iter)
    }

    fn iter_axis_inner(
        &self,
        pos: &[usize],
        axis: usize,
    ) -> Result<impl Iterator<Item = Vec<usize>>> {
        let mut pos = pos.to_vec();
        let axis_pos = pos[axis];
        let axis_max = self.shape[axis];

        Ok((axis_pos..axis_max).map(move |i| {
            pos[axis] = i;
            pos.clone()
        }))
    }

    pub fn view(&self, shape: Vec<usize>) -> Result<Self> {
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        let len: usize = shape.iter().product();
        if len != self.len() {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "invalid shape {:?} for a tensor has a length of {}",
                    shape, len
                ),
            )
                .into());
        }

        let strider = TensorStrider::new(shape);
        Ok(strider)
    }

    /// repeat the tensor along each axis. only used in the multi grouped query
    /// where each key/value head have multiple query heads.
    pub fn repeat(&self, repeats: Vec<usize>) -> Result<Self> {
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        if repeats.len() != self.shape.len() {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "invalid repeats {:?} for a tensor of shape {:?}",
                    repeats, self.shape
                ),
            )
                .into());
        }

        let new_shape = self
            .shape
            .iter()
            .zip(repeats.iter())
            .map(|(dim, repeat)| dim * repeat)
            .collect::<Vec<_>>();

        let strider = TensorStrider {
            shape: new_shape,
            strides: self.strides.clone(),
            repeats: Some(repeats.clone()),
        };
        Ok(strider)
    }

    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous_on_axis(0)
    }

    // if the tensor is contiguous on the given axis, you can safely iterate
    // the axis with a simple `.iter().step_by(strides[axis])`.
    pub fn is_contiguous_on_axis(&self, axis: usize) -> bool {
        if self.strides.len() == 0 {
            return true;
        }

        if let Some(repeats) = &self.repeats {
            let all_one = repeats[axis..].iter().all(|r| *r == 1);
            if !all_one {
                return false;
            }
        }

        if self.strides.last() != Some(&1) {
            return false;
        }

        let mut last_stride = 1;
        for i in (axis..self.shape.len()).rev() {
            if last_stride != self.strides[i] {
                return false;
            }
            last_stride *= self.shape[i];
        }

        true
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        strides.push(1);
        for i in 0..shape.len() - 1 {
            strides.push(strides.last().unwrap() * shape[shape.len() - i - 1]);
        }
        strides.reverse();
        strides
    }

    fn increment_pos(pos: &mut Vec<usize>, shape: &[usize]) {
        for i in (0..pos.len()).rev() {
            if pos[i] < shape[i] - 1 {
                pos[i] += 1;
                return;
            } else {
                pos[i] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strider_reshape() -> Result<()> {
        let s = TensorStrider::new(vec![3, 4]);
        assert_eq!(s.at(&[0, 0])?, 0);
        assert_eq!(s.at(&[0, 3])?, 3);
        assert_eq!(s.at(&[1, 0])?, 4);

        let r = s.view(vec![4, 2]);
        assert!(r.is_err());

        let s = s.view(vec![2, 6])?;
        assert_eq!(s.at(&[0, 0])?, 0);
        assert_eq!(s.at(&[0, 5])?, 5);
        assert_eq!(s.at(&[1, 0])?, 6);
        Ok(())
    }

    #[test]
    fn test_strider_iter_axis() -> Result<()> {
        let s = TensorStrider::new(vec![3, 4]);

        let r = s.iter_axis_inner(&[0, 0], 1)?.collect::<Vec<_>>();
        assert_eq!(r.len(), 4);
        assert_eq!(r[0], vec![0, 0]);
        assert_eq!(r[1], vec![0, 1]);
        assert_eq!(r[2], vec![0, 2]);
        assert_eq!(r[3], vec![0, 3]);

        let r = s.iter_axis_inner(&[0, 0], 0)?.collect::<Vec<_>>();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0], vec![0, 0]);
        assert_eq!(r[1], vec![1, 0]);
        assert_eq!(r[2], vec![2, 0]);
        Ok(())
    }

    #[test]
    fn test_repeat() -> Result<()> {
        let s_orig = TensorStrider::new(vec![2, 3]);
        let s = s_orig.repeat(vec![2, 1])?;
        assert_eq!(s.shape(), &[4, 3]);
        assert_eq!(s.strides(), &[3, 1]);
        assert_eq!(s.at_unchecked(&[0, 0]), 0);
        assert_eq!(s.at_unchecked(&[1, 0]), 0);
        assert_eq!(s.at_unchecked(&[2, 0]), 3);
        assert_eq!(s.at_unchecked(&[3, 0]), 3);
        Ok(())
    }

    #[test]
    fn test_is_contigous() -> Result<()> {
        let s = TensorStrider::new(vec![2, 3]);
        assert!(s.is_contiguous());
        assert!(s.is_contiguous_on_axis(1));
        assert!(s.is_contiguous_on_axis(0));

        let s = TensorStrider::new(vec![1, 2, 3]);
        let s = s.repeat(vec![2, 1, 1])?;
        assert!(!s.is_contiguous());
        assert!(s.is_contiguous_on_axis(2));
        assert!(s.is_contiguous_on_axis(1));
        assert!(!s.is_contiguous_on_axis(0));
        Ok(())
    }

    #[test]
    fn test_strider_increment_pos() -> Result<()> {
        let shape = vec![3, 2];
        let mut pos = vec![0, 0];

        TensorStrider::increment_pos(&mut pos, &shape);
        assert_eq!(pos, vec![0, 1]);
        TensorStrider::increment_pos(&mut pos, &shape);
        TensorStrider::increment_pos(&mut pos, &shape);
        assert_eq!(pos, vec![1, 1]);
        TensorStrider::increment_pos(&mut pos, &shape);
        TensorStrider::increment_pos(&mut pos, &shape);
        assert_eq!(pos, vec![2, 1]);
        Ok(())
    }
}
