use std::future::Future;
use std::rc::Rc;

use wgpu;

use crate::tensor::TensorStrider;

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

struct WgpuTensor {
    buf: Rc<wgpu::Buffer>,
    strider: TensorStrider,
    device: WgpuTensorDeviceRef,
}
