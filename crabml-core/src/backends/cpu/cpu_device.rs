use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use half::f16;

use super::CpuTensor;
use crate::tensor::TensorDeviceMetrics;

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
    pub(crate) metrics: TensorDeviceMetrics,
    pub(crate) debug_tensors: RefCell<HashMap<String, Vec<f32>>>,
    pub(crate) wbuf: RefCell<Option<Vec<f32>>>,
    pub(crate) exp_cache: Vec<f16>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

pub type CpuTensorDeviceRef<'a> = Rc<CpuTensorDevice<'a>>;

impl<'a> CpuTensorDevice<'a> {
    pub fn new() -> CpuTensorDeviceRef<'a> {
        let device = Self {
            opts: CpuTensorDeviceOptions::default(),
            debug_tensors: RefCell::new(HashMap::new()),
            metrics: TensorDeviceMetrics::default(),
            wbuf: RefCell::new(Some(vec![0.0; 32000])),
            exp_cache: Self::init_exp_cache(),
            _phantom: std::marker::PhantomData,
        };
        Rc::new(device)
    }

    pub fn with_options(opts: CpuTensorDeviceOptions) -> CpuTensorDeviceRef<'a> {
        let device = Self {
            opts,
            debug_tensors: RefCell::new(HashMap::new()),
            metrics: TensorDeviceMetrics::default(),
            wbuf: RefCell::new(Some(vec![0.0; 32000])),
            exp_cache: Self::init_exp_cache(),
            _phantom: std::marker::PhantomData,
        };
        Rc::new(device)
    }

    pub fn with_metrics(self: Rc<Self>, metrics: TensorDeviceMetrics) -> CpuTensorDeviceRef<'a> {
        let device = Self {
            opts: self.opts.clone(),
            debug_tensors: self.debug_tensors.clone(),
            wbuf: self.wbuf.clone(),
            exp_cache: self.exp_cache.clone(),
            metrics,
            _phantom: std::marker::PhantomData,
        };
        Rc::new(device)
    }

    pub fn metrics(&self) -> &TensorDeviceMetrics {
        &self.metrics
    }

    pub fn dump_debug_tensor(&self, name: &str) -> Option<Vec<f32>> {
        self.debug_tensors.borrow().get(name).cloned()
    }

    fn init_exp_cache() -> Vec<f16> {
        (0..65536)
            .map(|x| {
                let exp32 = f16::from_bits(x as u16).to_f32().exp();
                f16::from_f32(exp32)
            })
            .collect()
    }

    pub(crate) fn add_debug_tensor(&self, tensor: &CpuTensor<'a>) {
        let buf = tensor.buf().iter_f32().collect::<Vec<_>>();
        self.debug_tensors
            .borrow_mut()
            .insert(tensor.name.clone().unwrap(), buf);
    }
}
