use std::borrow::BorrowMut;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use wgpu;
use wgpu::util::DeviceExt;

use super::meta::RmsNormMeta;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::TensorArithmetics;
use crate::tensor::TensorStrider;

pub struct WgpuTensorDevice {
    inner: wgpu::Device,
    queue: wgpu::Queue,
    staging_buf: wgpu::Buffer,
    staging_buf_bytes: usize,
    modules: HashMap<&'static str, wgpu::ShaderModule>,
}

impl WgpuTensorDevice {
    fn new(staging_buf_bytes: usize) -> WgpuTensorDeviceRef {
        let (device, queue) = pollster::block_on(Self::init_wgpu());
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: staging_buf_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut d = Self {
            inner: device,
            queue,
            staging_buf,
            staging_buf_bytes,
            modules: HashMap::new(),
        };
        d.load_modules();
        Rc::new(d)
    }

    fn load_modules(&mut self) {
        let module_sources = vec![
            ("add_inplace", include_str!("shaders/add.wgsl")),
            ("mul_inplace", include_str!("shaders/mul.wgsl")),
            ("div_inplace", include_str!("shaders/div.wgsl")),
            ("rms_norm_inplace", include_str!("shaders/rms_norm.wgsl")),
        ];
        let mut modules = HashMap::new();
        for (module_name, module_source) in module_sources {
            let module = self
                .inner
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(module_source)),
                });
            modules.insert(module_name, module);
        }
        self.modules = modules
    }

    fn make_storage_buffer(&self, content: &[u8]) -> wgpu::Buffer {
        self.inner
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: content,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn pipeline_for(&self, key: &'static str) -> wgpu::ComputePipeline {
        let module = self.modules.get(key).unwrap();
        self.inner
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: "main",
            })
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

    fn encode_pipeline_commnad(
        &self,
        key: &'static str,
        entries: &[wgpu::BindGroupEntry],
        work_group_size: (u32, u32, u32),
    ) -> wgpu::CommandEncoder {
        let pipeline = self.pipeline_for(key);
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.inner.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries,
        });

        let mut encoder = self
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(work_group_size.0, work_group_size.1, work_group_size.2);
        }
        encoder
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
    fn new(src: &[f32], shape: &[usize], device: WgpuTensorDeviceRef) -> Result<Self> {
        let buf = device
            .inner
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input buffer"),
                contents: bytemuck::cast_slice(src),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let strider = TensorStrider::new(shape.to_vec());
        if strider.len() != src.len() {
            return Err((ErrorKind::TensorError, "buffer size mismatch").into());
        };
        Ok(Self {
            buf: Rc::new(buf),
            strider,
            device,
        })
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
        let strider = self.strider.reshape(shape.to_vec())?;
        self.with_strider(strider)
    }

    fn repeat(self, repeats: &[usize]) -> Result<Self> {
        let strider = self.strider.repeat(repeats.to_vec())?;
        self.with_strider(strider)
    }

    fn transpose(self, dims: &[usize]) -> Result<Self> {
        let strider = self.strider.transpose(dims)?;
        self.with_strider(strider)
    }

    fn strider(&self) -> &TensorStrider {
        &self.strider
    }

    fn extend(&mut self, rhs: &Self) -> Result<()> {
        todo!()
    }

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()> {
        todo!()
    }

    fn export(&self, dst: &mut [f32]) -> Result<()> {
        let buf_size = self.strider.len() * std::mem::size_of::<f32>();
        if buf_size != self.device.staging_buf_bytes {
            return Err((ErrorKind::TensorError, "buffer size mismatch").into());
        }

        // enqueue copy from self.buf to staging buffer
        let mut encoder = self
            .device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buf, 0, &self.device.staging_buf, 0, buf_size as u64);
        self.device.queue.submit(Some(encoder.finish()));

        // await from the staging buf
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let staging_slice = self.device.staging_buf.slice(..);
        staging_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.inner.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            // Gets contents of buffer
            let data = staging_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            dst.copy_from_slice(bytemuck::cast_slice(&data));

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            self.device.staging_buf.unmap();
        } else {
            panic!("failed to run compute on gpu!")
        }

        Ok(())
    }
}

impl TensorArithmetics for WgpuTensor {
    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self> {
        todo!()
    }

    fn rms_norm_inplace(self, eps: f32) -> Result<Self> {
        let meta_buf = self
            .device
            .make_storage_buffer(bytemuck::bytes_of(&RmsNormMeta {
                M: 1,
                N: self.strider.len() as u32,
                eps: eps,
                _padding: 0.0,
            }));
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: meta_buf.as_entire_binding(),
            },
        ];
        let encoder = self.device.encode_pipeline_commnad(
            "rms_norm_inplace",
            entries,
            (self.strider.len() as u32 / 64, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn softmax_inplace(self, axis: usize) -> Result<Self> {
        todo!()
    }

    fn silu_inplace(self) -> Result<Self> {
        todo!()
    }

    fn mul_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.strider().len() % 64 == 0);
        let meta_buf = self
            .device
            .make_storage_buffer(bytemuck::cast_slice(&[1u32, self.strider.len() as u32]));
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: meta_buf.as_entire_binding(),
            },
        ];
        let encoder = self.device.encode_pipeline_commnad(
            "mul_inplace",
            entries,
            (self.strider.len() as u32 / 64, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn add_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.strider().len() % 64 == 0);
        let meta_buf = self
            .device
            .make_storage_buffer(bytemuck::cast_slice(&[1u32, self.strider.len() as u32]));
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: meta_buf.as_entire_binding(),
            },
        ];
        let encoder = self
            .device
            .encode_pipeline_commnad("add_inplace", entries, (1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn div_scalar_inplace(self, rhs: f32) -> Result<Self> {
        assert!(self.strider().len() % 64 == 0);
        let meta_buf = self
            .device
            .make_storage_buffer(bytemuck::cast_slice(&[1u32, self.strider.len() as u32]));
        let rhs_buf = self
            .device
            .make_storage_buffer(bytemuck::cast_slice(&[rhs]));
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: meta_buf.as_entire_binding(),
            },
        ];
        let encoder = self.device.encode_pipeline_commnad(
            "div_inplace",
            entries,
            (self.strider.len() as u32 / 64, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }

    fn batch_matmul(&self, y: &Self) -> Result<Self> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuTensor;
    use super::WgpuTensorDevice;
    use crate::error::Result;
    use crate::tensor::Tensor;
    use crate::tensor::TensorArithmetics;

    #[test]
    fn test_wgpu_tensor_new_and_export() -> Result<()> {
        let device = WgpuTensorDevice::new(6 * 4);
        let t1 = WgpuTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)?;
        let mut dst = vec![0.0; 6];

        t1.export(&mut dst)?;

        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_add() -> Result<()> {
        let device = WgpuTensorDevice::new(64 * 4);
        let t1 = WgpuTensor::new(&[2.0; 64], &[16, 4], device.clone())?;
        let t2 = WgpuTensor::new(&[3.0; 64], &[16, 4], device)?;
        let t1 = t1.add_inplace(&t2)?;

        let mut dst = vec![0.0; 64];
        t1.export(&mut dst)?;

        assert_eq!(dst, vec![5.0; 64]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_mul() -> Result<()> {
        let device = WgpuTensorDevice::new(1024 * 4);
        let t1 = WgpuTensor::new(&[3.0; 1024], &[512, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 1024], &[512, 2], device)?;
        let t1 = t1.mul_inplace(&t2)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;

        assert_eq!(&dst[0..6], [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
        assert!(dst.iter().all(|v| *v == 6.0));
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_div_scalar() -> Result<()> {
        let device = WgpuTensorDevice::new(1024 * 4);
        let t1 = WgpuTensor::new(&[6.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.div_scalar_inplace(2.0)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;

        assert_eq!(&dst[0..3], [3.0, 3.0, 3.0]);
        Ok(())
    }
}
