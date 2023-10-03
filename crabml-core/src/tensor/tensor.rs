use crate::error::{ErrorKind, Result};
use std::{cell::RefCell, rc::Rc};

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

type TensorID = usize;

pub enum TensorDeviceOp {
    AllocTensor {
        strider: TensorStrider,
    },

    RecycleTensor {
        t: TensorID,
    },

    EditTensor {
        t: TensorID,
        strider: TensorStrider,
    },

    CopyFrom {
        dst: TensorID,
        pos: Vec<usize>,
        src: TensorID,
    },

    MatMul {
        out: TensorID,
        lhs: TensorID,
        rhs: TensorID,
    },

    RopeInplace {
        q: TensorID,
        k: TensorID,
        pos: usize,
        freq_base: f32,
        freq_scale: f32,
    },

    SiluInplace {
        t: TensorID,
    },

    MulInplace {
        t1: TensorID,
        t2: TensorID,
    },

    AddInplace {
        t1: TensorID,
        t2: TensorID,
    },

    RmsNormInplace {
        t: TensorID,
    }
}

// if we want to support GPU, we need to add a new type of device
pub trait TensorDevice {
    type DataType;

    fn process_op(&mut self, op: TensorDeviceOp) -> Result<Option<TensorID>>;

    fn register_tensor(&mut self, shape: &[usize], data: &[Self::DataType]) -> TensorID;

    fn export_tensor(self, t: TensorID, buf: &mut [Self::DataType]) -> Result<()>;

    fn data_type(&self) -> Self::DataType;

    fn name(&self) -> &'static str;
}

#[derive(Clone)]
pub struct Tensor<D: TensorDevice> {
    id: TensorID,
    strider: TensorStrider,
    device: Rc<RefCell<D>>,
}

impl<D: TensorDevice> Tensor<D> {
    pub fn zeros(shape: Vec<usize>, device: Rc<RefCell<D>>) -> Result<Self> {
        let strider: TensorStrider = TensorStrider::new(shape.clone(), 0);
        let id = device
            .borrow_mut()
            .process_op(TensorDeviceOp::AllocTensor {
                strider: strider.clone(),
            })?
            .unwrap();
        Ok(Self {
            id,
            strider,
            device,
        })
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn copy_from(&mut self, pos: &[usize], t: &Self) -> Result<()> {
        let strider = self.strider.row(pos)?;
        let n_elems = strider.len();

        todo!();
        Ok(())
    }

    pub fn reshape(self, shape: Vec<usize>) -> Result<Self> {
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
        let strider = TensorStrider::new(shape, self.strider.offset);

        self.device
            .borrow_mut()
            .process_op(TensorDeviceOp::EditTensor {
                t: self.id,
                strider: strider.clone(),
            })?;
        Ok(Self {
            strider,
            id: self.id,
            device: self.device.clone(),
        })
    }

    pub fn repeat(self, n: usize, axis: usize) -> Result<Self> {
        todo!()
    }

    pub fn mul(self, t: &Self) -> Result<Self> {
        todo!()
    }

    pub fn add(self, t: &Self) -> Result<Self> {
        todo!()
    }

    pub fn matmul(&self, t: &Self) -> Result<Self> {
        todo!()
    }
}

impl<D: TensorDevice> Drop for Tensor<D> {
    fn drop(&mut self) {
        self.device
            .borrow_mut()
            .process_op(TensorDeviceOp::RecycleTensor { t: self.id })
            .unwrap();
    }
}
