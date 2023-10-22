use std::cell::RefCell;
use std::rc::Rc;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::CpuTensor;
use crate::error::Result;
use crate::tensor::strider::TensorStrider;

pub type TensorBufID = usize;

#[derive(Clone, Debug)]
pub struct TensorOpVar {
    pub buf_id: usize,
    pub strider: TensorStrider,
}

#[derive(Clone, Debug)]
pub enum TensorOp {
    AllocTensor {
        shape: Vec<usize>,
    },

    RecycleTensor {
        t: TensorOpVar,
    },

    CopyFrom {
        src: TensorOpVar,
        pos: Vec<usize>,
        dst: TensorOpVar,
    },

    MatMul {
        out: TensorOpVar,
        lhs: TensorOpVar,
        rhs: TensorOpVar,
    },

    RopeInplace {
        t: TensorOpVar,
        pos: usize,
        rope_dim: usize,
    },

    SiluInplace {
        t: TensorOpVar,
    },

    MulInplace {
        lhs: TensorOpVar,
        rhs: TensorOpVar,
    },

    SoftmaxInplace {
        t: TensorOpVar,
        axis: usize,
    },

    DivScalarInplace {
        t: TensorOpVar,
        scalar: f32,
    },

    AddInplace {
        lhs: TensorOpVar,
        rhs: TensorOpVar,
    },

    RmsNormInplace {
        t: TensorOpVar,
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
pub trait TensorBackend<'a> {
    fn process_op(&mut self, op: TensorOp) -> Result<Option<TensorOpVar>>;

    fn import_buf(&mut self, buf: CpuTensorBuf<'a>) -> Result<TensorBufID>;

    fn export_buf(&self, buf_id: TensorBufID, data: &mut [f32]) -> Result<()>;

    fn name(&self) -> &'static str;
}

#[derive(Clone)]
pub struct Tensor<'a, D: TensorBackend<'a>> {
    buf_id: TensorBufID,
    strider: TensorStrider,
    backend: Rc<RefCell<D>>,
    _phantom: std::marker::PhantomData<&'a D>,
}

impl<'a, D: TensorBackend<'a>> Tensor<'a, D> {
    pub fn from_cpu(src: CpuTensor<'a>, backend: Rc<RefCell<D>>) -> Result<Self> {
        let strider = src.strider().clone();
        let buf_id = backend.borrow_mut().import_buf(src.into_buf())?;
        Ok(Self {
            buf_id,
            strider,
            backend: backend.clone(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn zeros(shape: &[usize], backend: Rc<RefCell<D>>) -> Result<Self> {
        let strider: TensorStrider = TensorStrider::new(shape.to_vec());
        let op_var = backend
            .borrow_mut()
            .process_op(TensorOp::AllocTensor {
                shape: strider.shape().to_vec(),
            })?
            .unwrap();
        Ok(Self {
            buf_id: op_var.buf_id,
            strider,
            backend,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn export(&self, dst: &mut [f32]) -> Result<()> {
        self.backend.borrow_mut().export_buf(self.buf_id, dst)?;
        Ok(())
    }

    fn as_op_var(&self) -> TensorOpVar {
        TensorOpVar {
            buf_id: self.buf_id,
            strider: self.strider.clone(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn copy_from(&mut self, pos: &[usize], src: &Self) -> Result<()> {
        self.backend.borrow_mut().process_op(TensorOp::CopyFrom {
            dst: self.as_op_var(),
            pos: pos.to_vec(),
            src: src.as_op_var(),
        })?;
        Ok(())
    }

    pub fn view(self, shape: Vec<usize>) -> Result<Self> {
        let strider = self.strider.view(shape)?;
        Ok(Self {
            strider,
            buf_id: self.buf_id,
            backend: self.backend.clone(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn repeat(self, repeats: Vec<usize>) -> Result<Self> {
        let strider = self.strider.repeat(repeats)?;
        Ok(Self {
            strider,
            buf_id: self.buf_id,
            backend: self.backend.clone(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn transpose(self, perm: &[usize]) -> Result<Self> {
        let strider = self.strider.transpose(perm)?;
        Ok(Self {
            strider,
            buf_id: self.buf_id,
            backend: self.backend.clone(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn mul(self, rhs: &Self) -> Result<Self> {
        self.backend.borrow_mut().process_op(TensorOp::MulInplace {
            lhs: self.as_op_var(),
            rhs: rhs.as_op_var(),
        })?;
        Ok(self)
    }

    pub fn add(self, rhs: &Self) -> Result<Self> {
        self.backend.borrow_mut().process_op(TensorOp::AddInplace {
            lhs: self.as_op_var(),
            rhs: rhs.as_op_var(),
        })?;
        Ok(self)
    }

    pub fn softmax(self, axis: usize) -> Result<Self> {
        self.backend
            .borrow_mut()
            .process_op(TensorOp::SoftmaxInplace {
                t: self.as_op_var(),
                axis,
            })?;
        Ok(self)
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        // TODO: validate shape here

        let out_shape = if rhs.shape().len() == 2 {
            vec![self.shape()[0], rhs.shape()[1]]
        } else {
            vec![self.shape()[0]]
        };

        let out = self.backend.borrow_mut().process_op(TensorOp::AllocTensor {
            shape: out_shape,
        })?.unwrap();

        self.backend.borrow_mut().process_op(TensorOp::MatMul {
            out: out.clone(),
            lhs: self.as_op_var(),
            rhs: rhs.as_op_var(),
        })?;

        Ok(Self {
            buf_id: out.buf_id,
            strider: out.strider.clone(),
            backend: self.backend.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, D: TensorBackend<'a>> Drop for Tensor<'a, D> {
    fn drop(&mut self) {
        self.backend
            .borrow_mut()
            .process_op(TensorOp::RecycleTensor {
                t: self.as_op_var(),
            })
            .unwrap();
    }
}
