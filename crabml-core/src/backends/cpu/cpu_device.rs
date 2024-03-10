use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use half::f16;

use super::CpuTensor;
use crate::tensor::TensorMetrics;

#[derive(Debug, Clone, Default)]
pub struct CpuTensorDeviceOptions {
    /// when enabled, whenever tensor called with `with_name`, the name and the
    /// tensor will be recorded in the device. only used in test.
    pub debug_named_tensors: bool,

    /// thread pool size for parallel computation
    pub thread_pool_size: Option<usize>,

    /// the metrics collector
    pub metrics: Option<TensorMetrics>,
}

#[derive(Debug)]
pub struct CpuTensorDevice<'a> {
    pub(crate) opts: CpuTensorDeviceOptions,
    pub(crate) metrics: TensorMetrics,
    pub(crate) debug_tensors: RefCell<HashMap<String, Vec<f32>>>,
    pub(crate) exp_cache: Rc<Vec<f16>>,
    /// not used yet, maybe can be used for storing logits
    pub(crate) _wbuf: RefCell<Option<Vec<f32>>>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

pub type CpuTensorDeviceRef<'a> = Rc<CpuTensorDevice<'a>>;

impl<'a> CpuTensorDevice<'a> {
    pub fn new(opts: CpuTensorDeviceOptions) -> CpuTensorDeviceRef<'a> {
        let device = Self {
            opts: opts.clone(),
            debug_tensors: RefCell::new(HashMap::new()),
            metrics: opts.metrics.unwrap_or_default(),
            _wbuf: RefCell::new(Some(vec![0.0; 32000])),
            exp_cache: Rc::new(Self::init_exp_cache()),
            _phantom: std::marker::PhantomData,
        };
        Rc::new(device)
    }

    pub fn metrics(&self) -> &TensorMetrics {
        &self.metrics
    }

    pub fn dump_debug_tensor(&self, name: &str) -> Option<Vec<f32>> {
        self.debug_tensors.borrow().get(name).cloned()
    }

    pub fn exp_cache(&self) -> Rc<Vec<f16>> {
        self.exp_cache.clone()
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
