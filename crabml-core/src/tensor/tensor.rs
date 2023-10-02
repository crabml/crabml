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
        Self { shape, strides, offset }
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
                )).into()
            );
        }

        let offset = pos
            .iter()
            .zip(self.strides.iter())
            .map(|(&p, &s)| p * s)
            .sum();

        let shape = self.shape[pos.len()..].to_vec();
        let strides = self.strides[pos.len()..].to_vec();
        Ok(TensorStrider { shape, strides, offset })
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

pub type TensorBufferHandle = usize;

#[derive(Clone)]
pub struct TensorBuffer<D: TensorDevice> {
    handle: TensorBufferHandle,
    device: Rc<RefCell<D>>,
}

impl<D: TensorDevice> TensorBuffer<D> {
    fn alloc(size: usize, device: &Rc<RefCell<D>>) -> Rc<RefCell<Self>> {
        let handle = device.borrow_mut().alloc_buffer(size);
        let buf = Self {
            handle,
            device: device.clone(),
        };
        Rc::new(RefCell::new(buf))
    }
}

impl<D: TensorDevice> Drop for TensorBuffer<D> {
    fn drop(&mut self) {
        self.device.borrow_mut().recycle_buffer(self.handle);
    }
}

pub trait TensorDevice {
    fn alloc_buffer(&mut self, size: usize) -> TensorBufferHandle;

    fn recycle_buffer(&mut self, handle: TensorBufferHandle);

    fn copy_buffer(&mut self, dst: TensorBufferHandle, src: TensorBufferHandle, offset: usize, len: usize);

    fn register_buffer(&mut self, buf: &[u8]) -> TensorBufferHandle;

    fn retrieve_buffer(&self, handle: TensorBufferHandle) -> &[u8];
}

#[derive(Clone)]
pub struct Tensor<D: TensorDevice> {
    strider: TensorStrider,
    elem_size: usize,
    buf: Rc<RefCell<TensorBuffer<D>>>,
}

impl<D: TensorDevice> Tensor<D> {
    pub fn zeros(shape: Vec<usize>, device: &Rc<RefCell<D>>) -> Self {
        let strider: TensorStrider = TensorStrider::new(shape, 0);
        let buf = TensorBuffer::alloc(strider.len(), device);
        Self { strider, buf, elem_size: 8 }
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn row(&self, pos: &[usize]) -> Result<Self> {
        let strider = self.strider.row(pos)?;
        Ok(Self {
            strider,
            elem_size: self.elem_size,
            buf: self.buf.clone(),
        })
    }

    pub fn copy_row_from(&mut self, pos: &[usize], t: &Self) -> Result<()> {
        let strider = self.strider.row(pos)?;
        let n_elems = strider.len();
        let src_len = n_elems * self.elem_size;
        let dst_offset = strider.offset * self.elem_size;
        let dst = self.buf.borrow_mut();
        let src = t.buf.borrow();
        dst.device.borrow_mut().copy_buffer(
            dst.handle,
            dst_offset,
            src.handle,
            src_len,
        );
        Ok(())
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
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
        Ok(Self {
            strider,
            elem_size: self.elem_size,
            buf: self.buf.clone(),
        })
    }
}
