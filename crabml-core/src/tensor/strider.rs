use crate::error::ErrorKind;
use crate::error::Result;

#[derive(Clone, Debug, Default)]
pub struct TensorStrider {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorStrider {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self { shape, strides }
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
        let mut offset = 0;
        for (dim, stride) in idx.iter().zip(self.strides.iter()) {
            offset += dim * stride;
        }
        offset
    }

    /// from the position, iterate until the end of the row / column
    pub fn iter_axis(
        &self,
        pos: Vec<usize>,
        axis: usize,
    ) -> Result<impl Iterator<Item = usize> + '_> {
        let iter = self.iter_axis_inner(pos, axis)?;
        let iter = iter.map(|pos| self.at_unchecked(&pos));
        Ok(iter)
    }

    pub fn into_iter_axis(
        self,
        pos: Vec<usize>,
        axis: usize,
    ) -> Result<impl Iterator<Item = usize>> {
        let iter = self.iter_axis_inner(pos, axis)?;
        let iter = iter.map(move |pos| self.at_unchecked(&pos));
        Ok(iter)
    }

    fn iter_axis_inner(
        &self,
        pos: Vec<usize>,
        axis: usize,
    ) -> Result<impl Iterator<Item = Vec<usize>>> {
        let mut pos = pos.to_vec();
        let axis_pos = pos[axis];
        let axis_max = self.shape[axis];
        for i in axis_pos..axis_max {
            pos[axis] = i;
        }

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

    pub fn is_contiguous(&self) -> bool {
        if self.strides.len() == 0 {
            return true;
        }

        if self.strides.last() != Some(&1) {
            return false;
        }

        let mut last_stride = 1;
        for i in (1..self.shape.len()).rev() {
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

        let r = s.iter_axis_inner(vec![0, 0], 1)?.collect::<Vec<_>>();
        assert_eq!(r.len(), 4);
        assert_eq!(r[0], vec![0, 0]);
        assert_eq!(r[1], vec![0, 1]);
        assert_eq!(r[2], vec![0, 2]);
        assert_eq!(r[3], vec![0, 3]);

        let r = s.iter_axis_inner(vec![0, 0], 0)?.collect::<Vec<_>>();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0], vec![0, 0]);
        assert_eq!(r[1], vec![1, 0]);
        assert_eq!(r[2], vec![2, 0]);
        Ok(())
    }
}
