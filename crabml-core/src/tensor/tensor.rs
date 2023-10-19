use std::cell::RefCell;
use std::rc::Rc;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::strider::TensorStrider;

type TensorID = usize;

pub enum TensorComputeOp {
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

    RefTensor {
        t: TensorID,
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
        t: TensorID,
        pos: usize,
        rope_dim: usize,
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
pub trait TensorBackend {
    fn append_op(&mut self, op: TensorComputeOp) -> Result<Option<TensorID>>;

    fn process_op(&mut self, op: TensorComputeOp) -> Result<Option<TensorID>>;

    fn import_tensor(&mut self, shape: &[usize], buf: &CpuTensorBuf) -> TensorID;

    fn export_tensor(self, t: TensorID, data: &mut [f32]) -> Result<()>;

    fn name(&self) -> &'static str;
}

#[derive(Clone)]
pub struct Tensor<D: TensorBackend> {
    id: TensorID,
    strider: TensorStrider,
    backend: Rc<RefCell<D>>,
}

impl<D: TensorBackend> Tensor<D> {
    pub fn zeros(shape: Vec<usize>, device: Rc<RefCell<D>>) -> Result<Self> {
        let strider: TensorStrider = TensorStrider::new(shape.clone());
        let id = device
            .borrow_mut()
            .append_op(TensorComputeOp::AllocTensor {
                strider: strider.clone(),
            })?
            .unwrap();
        Ok(Self {
            id,
            strider,
            backend: device,
        })
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn copy_from(&mut self, pos: &[usize], src: &Self) -> Result<()> {
        self.backend
            .borrow_mut()
            .append_op(TensorComputeOp::CopyFrom {
                dst: self.id,
                pos: pos.to_vec(),
                src: src.id,
            })?;
        Ok(())
    }

    pub fn view(self, shape: Vec<usize>) -> Result<Self> {
        let strider = self.strider.view(shape)?;

        self.backend
            .borrow_mut()
            .append_op(TensorComputeOp::EditTensor {
                t: self.id,
                strider: strider.clone(),
            })?;
        Ok(Self {
            strider,
            id: self.id,
            backend: self.backend.clone(),
        })
    }

    pub fn repeat(self, _n: usize, _axis: usize) -> Result<Self> {
        todo!()
    }

    pub fn mul(self, _t: &Self) -> Result<Self> {
        todo!()
    }

    pub fn add(self, _t: &Self) -> Result<Self> {
        todo!()
    }

    pub fn matmul(&self, _t: &Self) -> Result<Self> {
        todo!()
    }
}

impl<D: TensorBackend> Drop for Tensor<D> {
    fn drop(&mut self) {
        self.backend
            .borrow_mut()
            .append_op(TensorComputeOp::RecycleTensor { t: self.id })
            .unwrap();
    }
}
