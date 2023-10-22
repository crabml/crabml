use super::arithmetic::add_inplace;
use super::buf::CpuTensorBuf;
use super::pool::CpuTensorPool;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::tensor::TensorBackend;
use crate::tensor::tensor::TensorBufID;
use crate::tensor::tensor::TensorOp;
use crate::tensor::tensor::TensorOpVar;

pub struct CpuTensorBackend<'a> {
    pool: CpuTensorPool<'a>,
}

impl CpuTensorBackend<'_> {
    pub fn new() -> Self {
        Self { pool: CpuTensorPool::new() }
    }
}

impl<'a> TensorBackend<'a> for CpuTensorBackend<'a> {
    fn process_op(&mut self, op: TensorOp) -> Result<Option<TensorOpVar>> {
        match &op {
            TensorOp::AddInplace { lhs, rhs } => {
                let mut lhs = self.pool.load(lhs)?;
                let rhs = self.pool.load(rhs)?;
                add_inplace(&mut lhs, &rhs)?;
            }
            _ => todo!(),
        }
        Ok(None)
    }

    fn import_buf(&mut self, buf: CpuTensorBuf<'a>) -> Result<TensorBufID> {
        self.pool.import(buf)
    }

    fn export_buf(self, buf_id: TensorBufID, dst: &mut [f32]) -> Result<()> {
        self.pool.export(buf_id, dst)?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {}
}
