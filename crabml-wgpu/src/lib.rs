#![feature(thread_local)]
#![feature(lazy_cell)]

mod meta;
mod wgpu_device;
mod wgpu_tensor;

pub use wgpu_device::WgpuTensorDevice;
pub use wgpu_device::WgpuTensorDeviceOptions;
pub use wgpu_device::WgpuTensorDeviceRef;
pub use wgpu_tensor::WgpuTensor;
