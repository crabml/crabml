use std::rc::Rc;

use wgpu;
use wgpu::util::DeviceExt;

use crate::tensor::{TensorStrider, Tensor, TensorArithmetics};
use crate::error::Result;

pub struct WgpuTensorDevice {
    inner: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuTensorDevice {
    fn new() -> Self {
        let (device, queue) = pollster::block_on(Self::init_wgpu());
        Self { inner: device, queue }
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

pub type WgpuTensorDeviceRef = Rc<WgpuTensorDevice>;

#[derive(Clone)]
pub struct WgpuTensor {
    buf: Rc<wgpu::Buffer>,
    strider: TensorStrider,
    device: WgpuTensorDeviceRef,
}

impl WgpuTensor {
    fn new(buf: Vec<f32>, strider: TensorStrider, device: WgpuTensorDeviceRef) -> Self {
        let buf = device.inner
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input buffer"),
                contents: bytemuck::cast_slice(&buf),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Self { buf: Rc::new(buf), strider, device }
    }
}

impl Tensor for WgpuTensor {
    type Device = WgpuTensorDeviceRef;

    fn alloc(shape: &[usize], device: Self::Device) -> Result<Self> {
        todo!()
    }

    fn with_strider(self, strider: TensorStrider) -> Result<Self> {
        Ok(Self {
            buf: self.buf,
            strider: self.strider,
            device: self.device,
        })
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
