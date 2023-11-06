use std::rc::Rc;

use wgpu;

use crate::tensor::{TensorStrider, Tensor, TensorArithmetics};
use crate::error::Result;

struct WgpuTensorDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuTensorDevice {
    fn new() -> Self {
        let (device, queue) = pollster::block_on(Self::init_wgpu());
        Self { device, queue }
    }

    async fn init_wgpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();
        (device, queue)
    }
}

type WgpuTensorDeviceRef = Rc<WgpuTensorDevice>;

#[derive(Clone)]
struct WgpuTensor {
    buf: Rc<wgpu::Buffer>,
    strider: TensorStrider,
    device: WgpuTensorDeviceRef,
}

impl Tensor for WgpuTensor {
    type Device = WgpuTensorDeviceRef;

    fn alloc(shape: &[usize], device: Self::Device) -> Result<Self> {
        todo!()
    }

    fn with_strider(self, strider: TensorStrider) -> Result<Self> {
        todo!()
    }

    fn reshape(self, shape: &[usize]) -> Result<Self> {
        todo!()
    }

    fn repeat(self, repeats: &[usize]) -> Result<Self> {
        todo!()
    }

    fn transpose(self, shape: &[usize]) -> Result<Self> {
        todo!()
    }

    fn strider(&self) -> &TensorStrider {
        todo!()
    }

    fn extend(&mut self, rhs: &Self) -> Result<()> {
        todo!()
    }

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()> {
        todo!()
    }

    fn export(&self) -> Result<Box<dyn Iterator<Item = f32> + '_>> {
        todo!()
    }
}

impl TensorArithmetics for WgpuTensor {
    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self> {
        todo!()
    }

    fn rms_norm_inplace(self, eps: f32) -> Result<Self> {
        todo!()
    }

    fn softmax_inplace(self, axis: usize) -> Result<Self> {
        todo!()
    }

    fn silu_inplace(self) -> Result<Self> {
        todo!()
    }

    fn mul_inplace(self, rhs: &Self) -> Result<Self> {
        todo!()
    }

    fn add_inplace(self, rhs: &Self) -> Result<Self> {
        todo!()
    }

    fn div_scalar_inplace(self, rhs: f32) -> Result<Self> {
        todo!()
    }

    fn matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }

    fn batch_matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }
}