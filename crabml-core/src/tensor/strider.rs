use crate::error::ErrorKind;
use crate::error::Result;

#[derive(Clone)]
pub struct TensorStrider {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl TensorStrider {
    pub fn new(shape: Vec<usize>, offset: usize) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            shape,
            strides,
            offset,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn offset_at(&self, idx: &[usize]) -> usize {
        let mut offset = 0;
        for (dim, stride) in idx.iter().zip(self.strides.iter()) {
            offset += dim * stride;
        }
        offset
    }

    pub fn row(&self, pos: &[usize]) -> Result<TensorStrider> {
        if pos.len() >= self.shape.len() {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "invalid row position {:?} for tensor of shape {:?}",
                    pos, self.shape
                ),
            )
                .into());
        }

        let offset = pos
            .iter()
            .zip(self.strides.iter())
            .map(|(&p, &s)| p * s)
            .sum();

        let shape = self.shape[pos.len()..].to_vec();
        let strides = self.strides[pos.len()..].to_vec();
        Ok(TensorStrider {
            shape,
            strides,
            offset,
        })
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
    #[test]
    fn test_strider() {
    }
}