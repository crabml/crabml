use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;

use super::buf::CpuTensorBuf;
use super::CpuTensor;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::tensor::TensorBufID;
use crate::tensor::tensor::TensorOpVar;

pub struct CpuTensorPooledRef<'a> {
    buf_id: TensorBufID,
    tensor: Option<CpuTensor<'a>>,
    bufs: Rc<RefCell<HashMap<TensorBufID, CpuTensorBuf<'a>>>>,
}

impl<'a> Deref for CpuTensorPooledRef<'a> {
    type Target = CpuTensor<'a>;

    fn deref(&self) -> &Self::Target {
        self.tensor.as_ref().unwrap()
    }
}

impl<'a> DerefMut for CpuTensorPooledRef<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor.as_mut().unwrap()
    }
}

impl Drop for CpuTensorPooledRef<'_> {
    fn drop(&mut self) {
        let buf = self.tensor.take().unwrap().into_buf();
        self.bufs.borrow_mut().insert(self.buf_id, buf);
    }
}

pub struct CpuTensorPool<'a> {
    bufs: Rc<RefCell<HashMap<TensorBufID, CpuTensorBuf<'a>>>>,
    next_buf_id: usize,
    recycled_bufs: Vec<TensorBufID>,
}

impl<'a> CpuTensorPool<'a> {
    pub fn new() -> Self {
        Self {
            bufs: Rc::new(RefCell::new(HashMap::new())),
            recycled_bufs: vec![],
            next_buf_id: 0,
        }
    }

    pub fn import(&mut self, buf: CpuTensorBuf<'a>) -> Result<TensorBufID> {
        let buf_id = self.next_buf_id;
        let mut bufs = self.bufs.borrow_mut();
        bufs.insert(buf_id, buf);
        self.next_buf_id += 1;
        Ok(buf_id)
    }

    // TODO: a buffer might be recycled
    pub fn alloc(&mut self, shape: &[usize], _zeros: bool) -> Result<TensorBufID> {
        let tensor = CpuTensor::zeros(shape)?;
        let buf = tensor.into_buf();

        let buf_id = self.next_buf_id;
        let mut bufs = self.bufs.borrow_mut();
        bufs.insert(buf_id, buf);
        self.next_buf_id += 1;
        Ok(buf_id)
    }

    pub fn load(&self, op_var: &TensorOpVar) -> Result<CpuTensorPooledRef<'a>> {
        let mut bufs = self.bufs.borrow_mut();
        let buf = bufs.remove(&op_var.buf_id);
        let buf = match buf {
            Some(buf) => buf,
            None => return Err((ErrorKind::TensorNotFound, "invalid buf_id").into()),
        };

        let tensor = CpuTensor::new(buf, op_var.strider.clone())?;
        Ok(CpuTensorPooledRef {
            buf_id: op_var.buf_id,
            tensor: Some(tensor),
            bufs: self.bufs.clone(),
        })
    }

    pub fn export(&self, buf_id: TensorBufID, dst: &mut [f32]) -> Result<()> {
        let bufs = self.bufs.borrow();
        let buf = bufs.get(&buf_id);
        let buf = match buf {
            Some(buf) => buf,
            None => return Err((ErrorKind::TensorNotFound, "invalid buf_id").into()),
        };

        let buf = match buf {
            CpuTensorBuf::F32(buf) => buf,
            _ => return Err((ErrorKind::TensorNotFound, "only f32 buf can be exported").into()),
        };
        dst.copy_from_slice(buf);
        Ok(())
    }

    pub fn recycle(&mut self, op_var: &TensorOpVar) -> Result<()> {
        self.recycled_bufs.push(op_var.buf_id);
        Ok(())
    }
}
