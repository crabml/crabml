use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::CpuTensor;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct CpuTensorDeviceOptions {
    /// when enabled, whenever tensor called with `with_name`, the name and the
    /// tensor will be recorded in the device. only used in test.
    pub debug_named_tensors: bool,
}

impl Default for CpuTensorDeviceOptions {
    fn default() -> Self {
        Self {
            debug_named_tensors: false,
        }
    }
}

#[derive(Debug)]
pub struct CpuTensorDevice<'a> {
    pub(crate) opts: CpuTensorDeviceOptions,
    _bufs: Vec<CpuTensorBuf<'a>>,
    pub(crate) debug_tensors: RefCell<HashMap<String, Vec<f32>>>,
}

pub type CpuTensorDeviceRef<'a> = Rc<CpuTensorDevice<'a>>;

impl<'a> CpuTensorDevice<'a> {
    pub fn new() -> CpuTensorDeviceRef<'a> {
        let device = Self {
            opts: CpuTensorDeviceOptions::default(),
            _bufs: vec![],
            debug_tensors: RefCell::new(HashMap::new()),
        };
        Rc::new(device)
    }

    pub fn with_options(opts: CpuTensorDeviceOptions) -> CpuTensorDeviceRef<'a> {
        let device = Self {
            opts,
            _bufs: vec![],
            debug_tensors: RefCell::new(HashMap::new()),
        };
        Rc::new(device)
    }

    pub fn export_tensor(self: Rc<Self>, tensor: &CpuTensor<'a>, dst: &mut [f32]) -> Result<()> {
        tensor.iter().zip(dst.iter_mut()).for_each(|(src, dst)| {
            *dst = src;
        });
        Ok(())
    }

    pub fn dump_debug_tensor(&self, name: &str) -> Option<Vec<f32>> {
        self.debug_tensors.borrow().get(name).cloned()
    }

    pub(crate) fn add_debug_tensor(&self, tensor: &CpuTensor<'a>) {
        let buf = tensor.buf().iter().collect::<Vec<_>>();
        self.debug_tensors
            .borrow_mut()
            .insert(tensor.name.clone().unwrap(), buf);
    }
}
