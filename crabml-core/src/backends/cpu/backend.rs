use std::cell::RefCell;
use std::rc::Rc;

use rayon::slice::RChunks;

use super::arithmetic::add_inplace;
use super::arithmetic::batch_matmul_no_alloc;
use super::arithmetic::div_scalar_inplace;
use super::arithmetic::matmul_2d_1d_no_alloc;
use super::arithmetic::mul_inplace;
use super::arithmetic::rms_norm_inplace;
use super::arithmetic::rope_inplace;
use super::arithmetic::softmax_inplace;
use super::buf::CpuTensorBuf;
use super::pool::CpuTensorPool;
use crate::error::Result;
use crate::tensor::strider::TensorStrider;
use crate::tensor::tensor::TensorBackend;
use crate::tensor::tensor::TensorBufID;
use crate::tensor::tensor::TensorOp;
use crate::tensor::tensor::TensorOpVar;

pub struct CpuTensorBackend<'a> {
    pool: CpuTensorPool<'a>,
}

impl CpuTensorBackend<'_> {
    pub fn new() -> Rc<RefCell<Self>> {
        let backend = Self {
            pool: CpuTensorPool::new(),
        };
        Rc::new(RefCell::new(backend))
    }
}

impl<'a> TensorBackend<'a> for CpuTensorBackend<'a> {
    fn process_op(&mut self, op: TensorOp) -> Result<Option<TensorOpVar>> {
        match &op {
            TensorOp::RecycleTensor { t } => {
                self.pool.recycle(t)?;
            }
            TensorOp::AllocTensor { shape, zeros } => {
                let buf_id = self.pool.alloc(shape, *zeros)?;
                return TensorOpVar::new(buf_id, TensorStrider::new(shape.to_vec()));
            }
            TensorOp::ExtendTensor { dst, src } => {
                let mut dst = self.pool.load(dst)?;
                let src = self.pool.load(src)?;
                dst.extend(&src)?;
            }
            TensorOp::MatMul { out, lhs, rhs } => {
                let mut out = self.pool.load(out)?;
                let lhs = self.pool.load(lhs)?;
                let rhs = self.pool.load(rhs)?;
                matmul_2d_1d_no_alloc(&mut out, &lhs, &rhs)?;
            }
            TensorOp::BatchMatMul { out, lhs, rhs } => {
                let mut out = self.pool.load(out)?;
                let lhs = self.pool.load(lhs)?;
                let rhs = self.pool.load(rhs)?;
                batch_matmul_no_alloc(&mut out, &lhs, &rhs)?;
            }
            TensorOp::RmsNormInplace { t, eps } => {
                let mut t = self.pool.load(t)?;
                rms_norm_inplace(&mut t, *eps)?;
            }
            TensorOp::RopeInplace { t, pos, rope_dim } => {
                let mut t = self.pool.load(t)?;
                rope_inplace(&mut t, *pos, *rope_dim)?;
            }
            TensorOp::MulInplace { lhs, rhs } => {
                let mut lhs = self.pool.load(lhs)?;
                let rhs = self.pool.load(rhs)?;
                mul_inplace(&mut lhs, &rhs)?;
            }
            TensorOp::DivScalarInplace { lhs, rhs } => {
                let mut lhs = self.pool.load(lhs)?;
                div_scalar_inplace(&mut lhs, *rhs)?
            }
            TensorOp::SoftmaxInplace { t, axis } => {
                let mut t= self.pool.load(t)?;
                softmax_inplace(&mut t, *axis)?;
            }
            TensorOp::AddInplace { lhs, rhs } => {
                let mut lhs = self.pool.load(lhs)?;
                let rhs = self.pool.load(rhs)?;
                add_inplace(&mut lhs, &rhs)?;
            }
            _ => todo!("unimplemented: {:?}", op),
        }
        Ok(None)
    }

    fn import_buf(&mut self, buf: CpuTensorBuf<'a>) -> Result<TensorBufID> {
        self.pool.import(buf)
    }

    fn export_buf(&self, buf_id: TensorBufID, dst: &mut [f32]) -> Result<()> {
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
    use crate::backends::CpuTensor;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_cpu_backend() -> Result<()> {
        // 1, 2, 3
        // 4, 5, 6
        let raw_t1 = CpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let raw_t2 = CpuTensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3])?;
        let backend = CpuTensorBackend::new();

        let t1 = Tensor::from_cpu(raw_t1, backend.clone())?;
        let t2 = Tensor::from_cpu(raw_t2, backend.clone())?;
        let t1 = t1.add(&t2)?;

        let mut result = vec![0.0; 6];
        t1.export(&mut result)?;
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        Ok(())
    }
}
