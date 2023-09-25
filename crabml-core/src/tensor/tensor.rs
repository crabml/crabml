use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;
use std::borrow::Cow;
use std::fmt::Display;
use std::ops::Range;
use std::slice;

#[derive(Debug, Default, Clone)]
pub struct Tensor<'a> {
    buf: Cow<'a, [f32]>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    name: Option<String>,
}

impl<'a> Tensor<'a> {
    pub fn new(buf: impl Into<Cow<'a, [f32]>>, shape: Vec<usize>) -> Result<Self> {
        let buf = buf.into();
        if buf.len() != shape.iter().product() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: format!("invalid shape {:?} for data of length {}", shape, buf.len()),
                cause: None,
            });
        }

        let mut strides = Vec::with_capacity(shape.len());
        strides.push(1);
        for i in 0..shape.len() - 1 {
            strides.push(strides.last().unwrap() * shape[shape.len() - i - 1]);
        }
        strides.reverse();

        let tensor = Self {
            buf,
            shape,
            strides,
            name: None,
        };
        Ok(tensor)
    }

    pub fn from_raw_bytes(buf: &'a [u8], shape: Vec<usize>) -> Result<Self> {
        let len = buf.len();
        assert_eq!(
            len % std::mem::size_of::<f32>(),
            0,
            "Length of slice must be multiple of f32 size"
        );
        let new_len = len / std::mem::size_of::<f32>();
        let ptr = buf.as_ptr() as *const f32;
        let f32_buf = unsafe { slice::from_raw_parts(ptr, new_len) };
        Self::new(f32_buf, shape)
    }

    pub fn with_name(self, name: impl Into<String>) -> Self {
        Self {
            buf: self.buf,
            shape: self.shape,
            strides: self.strides,
            name: Some(name.into()),
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(|s| s.as_str())
    }

    pub fn iter<'b>(&'b self) -> Box<dyn Iterator<Item = f32> + 'b> {
        if self.shape.len() == 1 {
            return Box::new(Tensor1DIterator {
                tensor: self,
                logical_pos: 0,
            });
        }
        Box::new(TensorIterator {
            tensor: self,
            logical_pos: 0,
            idx_buf: vec![0; self.shape.len()],
        })
    }

    pub fn at(&self, idx: &[usize]) -> Result<f32> {
        if idx.len() != self.shape.len() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: format!(
                    "invalid index {:?} for tensor of shape {:?}",
                    idx, self.shape
                ),
                cause: None,
            });
        }
        for (i, &dim) in idx.iter().enumerate() {
            if dim >= self.shape[i] {
                return Err(Error {
                    kind: ErrorKind::TensorError,
                    message: format!(
                        "invalid index {:?} for tensor of shape {:?}",
                        idx, self.shape
                    ),
                    cause: None,
                });
            }
        }

        Ok(self.at_unchecked(idx))
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn view(&self, shape: &[usize]) -> Result<Self> {
        if shape.iter().product::<usize>() != self.len() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: format!(
                    "invalid shape {:?} for data of length {}",
                    shape,
                    self.len(),
                ),
                cause: None,
            });
        }
        if !self.is_contiguous() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: "cannot view a non-contiguous tensor".to_string(),
                cause: None,
            });
        }
        Self::new(self.buf.clone(), shape.to_vec())
    }

    pub fn at_unchecked(&self, idx: &[usize]) -> f32 {
        let offset = self.buf_offset(idx);
        self.buf[offset]
    }

    fn buf_offset(&self, idx: &[usize]) -> usize {
        let mut offset = 0;
        for (dim, stride) in idx.iter().zip(self.strides.iter()) {
            offset += dim * stride;
        }
        offset
    }

    pub fn transpose(&self, perm: &[usize]) -> Result<Self> {
        if perm.len() != self.shape.len() {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: format!(
                    "invalid transpose {:?} for tensor of shape {:?}",
                    perm, self.shape
                ),
                cause: None,
            });
        }
        let mut new_shape = vec![0; self.shape.len()];
        for (i, &dim) in perm.iter().enumerate() {
            new_shape[i] = self.shape[dim];
        }
        let mut new_strides = vec![0; self.shape.len()];
        for (i, &dim) in perm.iter().enumerate() {
            new_strides[i] = self.strides[dim];
        }
        let tensor = Self {
            buf: self.buf.clone(),
            shape: new_shape,
            strides: new_strides,
            name: self.name.clone(),
        };
        Ok(tensor)
    }

    // todo: test it
    pub fn crop(&self, limits: &[(usize, usize)]) -> Result<Self> {
        let offset = self.buf_offset(&limits.iter().map(|&(start, _)| start).collect::<Vec<_>>());
        let buf = self.slice_buf(offset..self.buf.len());
        let shape = limits
            .iter()
            .map(|&(start, end)| end - start)
            .collect::<Vec<_>>();
        Ok(Self {
            buf,
            shape,
            strides: self.strides.clone(),
            name: self.name.clone(),
        })
    }

    pub fn subtensor(&self, row: usize) -> Result<Self> {
        if self.shape.len() <= 1 {
            return Err(Error {
                kind: ErrorKind::TensorError,
                message: "cannot subtensor a 1D tensor".to_string(),
                cause: None,
            });
        }

        if self.is_contiguous() {
            let offset = row * self.strides[0];
            let buf = self.slice_buf(offset..offset + self.strides[0]);
            return Ok(Self {
                buf,
                shape: self.shape[1..].to_vec(),
                strides: self.strides[1..].to_vec(),
                name: self.name.clone(),
            });
        }

        let mut idx = vec![0; self.shape.len()];
        idx[0] = row;
        let offset = self.buf_offset(&idx);
        let buf = self.slice_buf(offset..self.buf.len());
        Ok(Self {
            buf,
            shape: self.shape[1..].to_vec(),
            strides: self.strides[1..].to_vec(),
            name: self.name.clone(),
        })
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

    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let buf = self.iter().collect::<Vec<_>>();
        Self::new(buf, self.shape.clone())
    }

    pub fn flat(&self) -> &[f32] {
        &self.buf
    }

    pub fn flat_mut(&mut self) -> Result<&mut [f32]> {
        match self.buf {
            Cow::Borrowed(_) => Err(Error {
                kind: ErrorKind::TensorError,
                message: "can not mut a borrowed tensor".into(),
                cause: None,
            }),
            Cow::Owned(_) => Ok(self.buf.to_mut()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn subtensors(&self) -> Result<Vec<Tensor<'a>>> {
        let mut result = Vec::with_capacity(self.shape[0]);
        for i in 0..self.shape[0] {
            result.push(self.subtensor(i)?);
        }
        Ok(result)
    }

    fn slice_buf(&self, range: Range<usize>) -> Cow<'a, [f32]> {
        match self.buf {
            Cow::Borrowed(data) => Cow::from(&data[range]),
            Cow::Owned(ref data) => Cow::from(Vec::from(&data[range])),
        }
    }
}

struct TensorIterator<'a, 'b>
where
    'a: 'b,
{
    tensor: &'b Tensor<'a>,
    logical_pos: usize,
    idx_buf: Vec<usize>,
}

impl<'a, 'b> Iterator for TensorIterator<'a, 'b> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.logical_pos >= self.tensor.buf.len() {
            return None;
        }

        self.idx_buf.fill(0);
        let mut lp = self.logical_pos;
        for (dim, idx) in self
            .tensor
            .shape()
            .iter()
            .rev()
            .zip(self.idx_buf.iter_mut().rev())
        {
            *idx = lp % dim;
            lp = (lp - *idx) / dim;
        }
        let offset = self.tensor.buf_offset(&self.idx_buf);

        self.logical_pos += 1;
        Some(self.tensor.buf[offset])
    }
}

struct Tensor1DIterator<'a, 'b>
where
    'a: 'b,
{
    tensor: &'b Tensor<'a>,
    logical_pos: usize,
}

impl<'a, 'b> Iterator for Tensor1DIterator<'a, 'b> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.logical_pos >= self.tensor.shape[0] {
            return None;
        }

        let physical_pos = self.logical_pos * self.tensor.strides[0];

        self.logical_pos += 1;
        Some(self.tensor.buf[physical_pos])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.tensor.shape[0] - self.logical_pos))
    }
}

impl<'a> Display for Tensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shape().len() == 1 {
            write!(f, "[")?;
            for (i, v) in self.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if i >= 8 {
                    write!(f, "...")?;
                    break;
                }
                write!(f, "{}", v)?;
            }
            write!(f, "]")?;
            return Ok(());
        }

        if self.shape().len() >= 2 {
            write!(f, "[")?;
            for (i, v) in self.subtensors().unwrap().iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", v)?;
                if i < self.shape()[0] - 1 {
                    write!(f, "\n")?;
                }
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() -> Result<()> {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&v, vec![2, 3]).unwrap();
        assert_eq!(
            t.subtensor(0)?.iter().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            t.subtensor(1)?.iter().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0]
        );

        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&v, vec![2, 3, 1]).unwrap();
        assert_eq!(format!("{:?}", t.strides), "[3, 1, 1]");
        assert_eq!(t.is_contiguous(), true);
        assert_eq!(t.subtensor(0)?.flat().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(t.subtensor(1)?.flat().to_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(t.subtensor(0)?.subtensor(0)?.flat().to_vec(), vec![1.0]);
        assert_eq!(t.subtensor(0)?.subtensor(1)?.flat().to_vec(), vec![2.0]);
        assert_eq!(t.subtensor(0)?.subtensor(2)?.flat().to_vec(), vec![3.0]);
        assert_eq!(t.subtensor(1)?.subtensor(0)?.flat().to_vec(), vec![4.0]);
        assert_eq!(t.subtensor(1)?.shape().to_vec(), vec![3, 1]);

        let v = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let t = Tensor::new(&v, vec![2, 3, 2, 1]).unwrap();
        assert_eq!(
            t.subtensor(0)?.flat().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            t.subtensor(1)?.flat().to_vec(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        Ok(())
    }

    #[test]
    fn test_tensor_transform() -> Result<()> {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&v, vec![2, 3]).unwrap();
        assert_eq!(t.strides.to_vec(), vec![3, 1]);
        assert_eq!(t.at(&[0, 0])?, 1.0);
        assert_eq!(t.at(&[0, 1])?, 2.0);
        assert_eq!(t.at(&[0, 2])?, 3.0);
        assert_eq!(t.at(&[0, 4]).unwrap_err().kind, ErrorKind::TensorError);
        assert_eq!(t.at(&[1, 0])?, 4.0); // offset = 1 * 3 + 0 * 1 = 2
        assert_eq!(t.at(&[1, 1])?, 5.0);
        assert_eq!(t.at(&[1, 2])?, 6.0);

        let t = t.transpose(&[1, 0])?;
        assert_eq!(t.strides.to_vec(), vec![1, 3]);
        assert_eq!(t.at(&[0, 0])?, 1.0);
        assert_eq!(t.at(&[1, 0])?, 2.0);
        assert_eq!(t.at(&[2, 0])?, 3.0);
        assert_eq!(t.at(&[4, 0]).unwrap_err().kind, ErrorKind::TensorError);
        assert_eq!(t.at(&[0, 1])?, 4.0);
        assert_eq!(t.at(&[1, 1])?, 5.0);
        assert_eq!(t.at(&[2, 1])?, 6.0);

        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&v, vec![2, 3]).unwrap(); // 2x3
        let t1 = t.subtensor(0)?; // (3, )
        assert_eq!(t1.shape(), &[3]);
        assert_eq!(t1.at(&[0])?, 1.0); // offset = 1 * 3 + 0 * 1 = 2
        assert_eq!(t1.at(&[1])?, 2.0);
        assert_eq!(t1.at(&[2])?, 3.0);
        let t2 = t.transpose(&[1, 0])?;
        assert_eq!(t2.shape.to_vec(), vec![3, 2]);
        let t3 = t.subtensor(1)?; // (2, )
        assert_eq!(t3.at(&[0])?, 4.0);

        Ok(())
    }

    #[test]
    fn test_tensor_iterator() -> Result<()> {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&v, vec![2, 3]).unwrap();
        let tv = t.iter().collect::<Vec<_>>();
        assert_eq!(tv, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t = t.transpose(&[1, 0])?;
        let tv = t.iter().collect::<Vec<_>>();
        assert_eq!(tv, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let v = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::new(&v, vec![4]).unwrap();
        let tv = t.iter().collect::<Vec<_>>();
        assert_eq!(tv, vec![1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_contiguous() -> Result<()> {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // 1, 2, 3
        // 4, 5, 6
        let t = Tensor::new(&v, vec![2, 3]).unwrap();
        assert_eq!(t.to_string(), "[[1, 2, 3]\n,[4, 5, 6]]");

        // 1, 4,
        // 2, 5,
        // 3, 6,
        let t = t.transpose(&[1, 0])?;
        let t = t.contiguous()?;
        assert_eq!(t.flat(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // 1, 2, 3
        // 4, 5, 6
        let t = t.transpose(&[1, 0])?;
        let t = t.contiguous()?;
        assert_eq!(t.flat(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        Ok(())
    }
}
