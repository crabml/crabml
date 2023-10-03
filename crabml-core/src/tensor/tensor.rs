use crate::error::{ErrorKind, Result};
use crate::tensor::strider::TensorStrider;

use std::{cell::RefCell, rc::Rc};

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
    },
}

/// A tensor device is responsible for manage the buffer of tensors, and performing
/// platform specific operations on tensors.
///
/// A tensor device might be a CPU, GPU. Besides that, if we want to inference a model
/// with quantization, we can also have a specialized quantized CPU/GPU tensor device.
///
/// The tensors are managed in a handle based way. The device is responsible for manage
/// the pool of the tensors, each tensor is identified by a unique TensorID. The tensor
/// may located in the CPU or GPU memory, you can not directly acccess its data except
/// calling `export_tensor()` to load the tensor's data into the host's memory.
pub trait TensorDevice {
    type DataType;

    fn process_op(&mut self, op: TensorDeviceOp) -> Result<Option<TensorID>>;

    fn import_tensor(&mut self, shape: &[usize], data: &[Self::DataType]) -> TensorID;

    fn export_tensor(self, t: TensorID, data: &mut [Self::DataType]) -> Result<()>;

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

        let strider = TensorStrider::new(shape, self.strider.offset());

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
