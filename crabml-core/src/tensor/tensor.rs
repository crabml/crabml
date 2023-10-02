use crate::error::{ErrorKind, Result};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub struct TensorStrider {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorStrider {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self { shape, strides }
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

    fn copy_buffer(&mut self, handle: TensorBufferHandle, offset: usize, buf: &[u8]);

    fn register_buffer(&mut self, buf: &[u8]);

    fn retrieve_buffer(&self, handle: TensorBufferHandle) -> &[u8];
}

#[derive(Clone)]
pub struct Tensor<D: TensorDevice> {
    strider: TensorStrider,
    buf: Rc<RefCell<TensorBuffer<D>>>,
}

impl<D: TensorDevice> Tensor<D> {
    pub fn zeros(shape: Vec<usize>, device: &Rc<RefCell<D>>) -> Self {
        let strider: TensorStrider = TensorStrider::new(shape);
        let buf = TensorBuffer::alloc(strider.len(), device);
        Self { strider, buf }
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    pub fn len(&self) -> usize {
        self.strider.len()
    }

    pub fn view(&self, shape: Vec<usize>) -> Result<Self> {
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
        let strider = TensorStrider::new(shape);
        Ok(Self {
            strider,
            buf: self.buf.clone(),
        })
    }
}
