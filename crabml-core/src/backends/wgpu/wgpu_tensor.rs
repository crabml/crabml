use std::borrow::BorrowMut;
use std::rc::Rc;

use wgpu;
use wgpu::util::DeviceExt;

use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::TensorArithmetics;
use crate::tensor::TensorStrider;

pub struct WgpuTensorDevice {
    inner: wgpu::Device,
    queue: wgpu::Queue,
    staging_buf: wgpu::Buffer,
    staging_buf_size: usize,
}

impl WgpuTensorDevice {
    fn new(staging_buf_size: usize) -> Self {
        let (device, queue) = pollster::block_on(Self::init_wgpu());
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: staging_buf_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            inner: device,
            queue,
            staging_buf,
            staging_buf_size,
        }
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
        let buf = device
            .inner
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input buffer"),
                contents: bytemuck::cast_slice(&buf),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Self {
            buf: Rc::new(buf),
            strider,
            device,
        }
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
        let buf_size = self.strider.len() * 8;
        if buf_size > self.device.staging_buf_size {
            return Err((ErrorKind::TensorError, "buffer size too large").into());
        }

        let mut encoder = self
            .device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            &self.buf,
            0,
            &self.device.staging_buf,
            0,
            buf_size as u64,
        );

        self.device.queue.submit(Some(encoder.finish()));
    todo!()
}
}
