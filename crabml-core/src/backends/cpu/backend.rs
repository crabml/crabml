use crate::error::ErrorKind;
use crate::tensor::tensor::TensorBackend;
use crate::tensor::tensor::TensorBufID;
use crate::tensor::tensor::TensorOp;
use crate::tensor::tensor::TensorOpVar;
use crate::error::Result;

use super::buf::CpuTensorBuf;

pub struct CpuTensorBackend<'a> {
    bufs: Vec<CpuTensorBuf<'a>>,
}

impl<'a> TensorBackend<'a> for CpuTensorBackend<'a> {
    fn process_op(&mut self, op: TensorOp) -> Result<Option<TensorOpVar>> {
        todo!()
    }

    fn import_buf(&mut self, buf: CpuTensorBuf<'a>) -> Result<TensorBufID> {
        let next_buf_id = self.bufs.len();
        self.bufs.push(buf);
        Ok(next_buf_id)
    }

    fn export_buf(self, buf_id: TensorBufID, dst: &mut [f32]) -> Result<()> {
        if buf_id >= self.bufs.len() {
            return Err((ErrorKind::InvalidArgs, "invalid buf_id").into());
        }
        let buf = &self.bufs[buf_id];
        let buf = match buf {
            CpuTensorBuf::F32(buf) => buf,
            _ => return Err((ErrorKind::InvalidArgs, format!("only f32 buf can be exported, but got {}", buf.typ())).into()),
        };

        if buf.len() != dst.len() {
            return Err((ErrorKind::InvalidArgs, format!("mismatched buf len, want {}, but got {}", dst.len(), buf.len())).into());
        }

        dst.copy_from_slice(buf);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}