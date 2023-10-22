use crate::tensor::tensor::TensorBackend;
use crate::tensor::tensor::TensorOp;
use crate::tensor::tensor::TensorOpVar;
use crate::error::Result;

use super::buf::CpuTensorBuf;

pub struct CpuTensorBackend<'a> {
    bufs: Vec<CpuTensorBuf<'a>>,
}

impl<'a> TensorBackend for CpuTensorBackend<'a> {
    fn append_op(&mut self, op: TensorOp) -> Result<Option<TensorOpVar>> {
        todo!()
    }

    fn import_tensor<'b>(&'b mut self, shape: &[usize], buf: &CpuTensorBuf<'b>) -> TensorOpVar {
        todo!()
    }

    fn export_tensor(self, t: TensorOpVar, data: &mut [f32]) -> Result<()> {
        todo!()
    }

    fn name(&self) -> &'static str {
        todo!()
    }
}