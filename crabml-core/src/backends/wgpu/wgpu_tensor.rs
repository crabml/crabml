use std::borrow::BorrowMut;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use wgpu;
use wgpu::util::DeviceExt;

use super::meta::MatmulMeta;
use super::meta::RmsNormMeta;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::TensorArithmetics;
use crate::tensor::TensorStrider;

pub struct WgpuTensorDeviceOptions {
    pub staging_buf_bytes: usize,

    pub debug_named_tensor: bool,
}

impl WgpuTensorDeviceOptions {
    pub fn new(staging_buf_bytes: usize) -> Self {
        Self {
            staging_buf_bytes,
            debug_named_tensor: false,
        }
    }

    pub fn with_debug_named_tensor(mut self, v: bool) -> Self {
        self.debug_named_tensor = v;
        self
    }
}

pub struct WgpuTensorDevice {
    opts: WgpuTensorDeviceOptions,
    inner: wgpu::Device,
    queue: wgpu::Queue,
    staging_buf: wgpu::Buffer,
    modules: HashMap<&'static str, wgpu::ShaderModule>,

    /// used for test only
    pub debug_tensors: RefCell<HashMap<String, Vec<f32>>>,
}

impl WgpuTensorDevice {
    pub fn new(opts: WgpuTensorDeviceOptions) -> WgpuTensorDeviceRef {
        let (device, queue) = pollster::block_on(Self::init_wgpu());
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: opts.staging_buf_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut d = Self {
            inner: device,
            opts,
            queue,
            staging_buf,
            modules: HashMap::new(),
            debug_tensors: RefCell::new(HashMap::new()),
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
            ("matmul_naive", include_str!("shaders/matmul_naive.wgsl")),
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

    fn record_debug_tensor(&self, name: String, tensor: &WgpuTensor) {
        let mut dst = vec![0.0; tensor.strider().len()];
        tensor.export(&mut dst).unwrap();
        self.debug_tensors.borrow_mut().insert(name, dst);
    }

    pub fn dump_debug_tensor(&self, name: &str) -> Option<Vec<f32>> {
        self.debug_tensors.borrow().get(name).cloned()
    }
}

pub type WgpuTensorDeviceRef = Rc<WgpuTensorDevice>;

#[derive(Clone)]
pub struct WgpuTensor {
    buf: Rc<wgpu::Buffer>,
    strider: TensorStrider,
    device: WgpuTensorDeviceRef,
    name: Option<String>,
}

impl WgpuTensor {
    pub fn new(src: &[f32], shape: &[usize], device: WgpuTensorDeviceRef) -> Result<Self> {
        let buf = device
            .inner
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor weights buffer"),
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
            name: None,
        })
    }

    pub fn is_contiguous(&self) -> bool {
        self.strider.is_contiguous()
    }

    pub fn shape(&self) -> &[usize] {
        self.strider.shape()
    }
}

impl Tensor for WgpuTensor {
    type Device = WgpuTensorDeviceRef;

    fn alloc(shape: &[usize], device: Self::Device) -> Result<Self> {
        let buf_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let buf = device.inner.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor storage buffer"),
            size: buf_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let strider = TensorStrider::new(shape.to_vec());
        Ok(Self {
            buf: Rc::new(buf),
            strider,
            device,
            name: None,
        })
    }

    fn with_strider(self, strider: TensorStrider) -> Result<Self> {
        Ok(Self {
            buf: self.buf,
            strider: strider,
            device: self.device,
            name: None,
        })
    }

    fn with_name(mut self, name: String) -> Self {
        if self.device.opts.debug_named_tensor {
            self.device.record_debug_tensor(name.clone(), &self);
        }

        self.name = Some(name);
        self
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
        return Err((ErrorKind::NotImplemented, "not implemented").into());
    }

    fn copy_from(&mut self, rhs: &Self, pos: &[usize], len: usize) -> Result<()> {
        // TODO: check is_owned
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }

        let f32_size = std::mem::size_of::<f32>();
        let offset = rhs.strider.at(pos).unwrap() * f32_size;
        let bytes_len = len * f32_size;

        // enqueue copy from rhs to self's buffer
        let mut encoder = self
            .device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&rhs.buf, offset as u64, &self.buf, 0, bytes_len as u64);
        self.device.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    fn export(&self, dst: &mut [f32]) -> Result<()> {
        let buf_size = self.strider.len() * std::mem::size_of::<f32>();
        if buf_size != self.device.opts.staging_buf_bytes {
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

    fn dup(&self) -> Result<Self> {
        let mut new_tensor = Self::alloc(self.strider.shape(), self.device.clone())?;
        new_tensor
            .copy_from(&self, &vec![0; self.strider.len()], self.strider.len())
            .unwrap();
        Ok(new_tensor)
    }
}

impl TensorArithmetics for WgpuTensor {
    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self> {
        return Err((ErrorKind::NotImplemented, "not implemented").into());
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
        let encoder = self
            .device
            .encode_pipeline_commnad("rms_norm_inplace", entries, (1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn softmax_inplace(self, axis: usize) -> Result<Self> {
        return Err((ErrorKind::NotImplemented, "not implemented").into());
    }

    fn silu_inplace(self) -> Result<Self> {
        return Err((ErrorKind::NotImplemented, "not implemented").into());
    }

    fn mul_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.strider().len() % 32 == 0);
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
            .encode_pipeline_commnad("mul_inplace", entries, (1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn add_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.strider().len() % 32 == 0);
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
        assert!(self.strider().len() % 32 == 0);
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
            (self.strider.len() as u32 / 32, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn matmul(&self, y: &Self) -> Result<Self> {
        assert!(self.shape().len() == 2);
        assert!(self.shape()[1] == y.shape()[0]);
        assert!(y.shape().len() == 1);
        assert!(self.is_contiguous());
        assert!(y.is_contiguous());

        let output = Self::alloc(&[self.strider.shape()[0]], self.device.clone())?;
        let meta = MatmulMeta {
            M: self.strider.shape()[0] as u32,
            N: self.strider.shape()[1] as u32,
            K: 1,
            _padding: 0,
        };

        let meta_buf = self.device.make_storage_buffer(bytemuck::bytes_of(&meta));
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buf.as_entire_binding(),
            },
        ];
        let encoder =
            self.device
                .encode_pipeline_commnad("matmul_naive", entries, (meta.M / 32, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));

        Ok(output)
    }

    fn batch_matmul(&self, y: &Self) -> Result<Self> {
        return Err((ErrorKind::NotImplemented, "not implemented").into());
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::WgpuTensor;
    use super::WgpuTensorDevice;
    use crate::backends::wgpu::wgpu_tensor::WgpuTensorDeviceOptions;
    use crate::error::Result;
    use crate::tensor::Tensor;
    use crate::tensor::TensorArithmetics;

    #[test]
    fn test_wgpu_tensor_new_and_export() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new(6 * 4));
        let t1 = WgpuTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)?;
        let mut dst = vec![0.0; 6];

        t1.export(&mut dst)?;

        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_add() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new(64 * 4));
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
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new(1024 * 4));
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
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new(1024 * 4));
        let t1 = WgpuTensor::new(&[6.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.div_scalar_inplace(2.0)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;

        assert_eq!(&dst[0..3], [3.0, 3.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_alloc() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new(1024 * 4));
        let t1 = WgpuTensor::alloc(&[512, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[1.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.add_inplace(&t2)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;

        assert_eq!(&dst[0..3], [1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_with_name() -> Result<()> {
        let device_opts = WgpuTensorDeviceOptions::new(1024 * 4).with_debug_named_tensor(true);
        let device = WgpuTensorDevice::new(device_opts);

        let t1 = WgpuTensor::alloc(&[512, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[1.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.add_inplace(&t2)?;
        let _ = t1.with_name("t1".to_string());

        let dst = device.dump_debug_tensor("t1").unwrap();
        assert_eq!(dst, vec![1.0; 1024]);
        Ok(())
    }

    #[test]
    fn test_wgpu_copy_from() -> Result<()> {
        let device_opts = WgpuTensorDeviceOptions::new(1024 * 4).with_debug_named_tensor(true);
        let device = WgpuTensorDevice::new(device_opts);

        let mut t1 = WgpuTensor::alloc(&[256, 4], device.clone())?;
        let t2 = WgpuTensor::new(
            &(0..1024).map(|d| d as f32).collect::<Vec<f32>>(),
            &[256, 4],
            device.clone(),
        )?;

        assert_eq!(t2.strider.at(&[1, 0])?, 4);
        t1.copy_from(&t2, &[1, 0], 4)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;
        assert_eq!(&dst[0..4], [4.0, 5.0, 6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_rms_norm() -> Result<()> {
        // it seems that webgpu have a different rounding method on dividing f32:
        // https://stackoverflow.com/questions/73674463/does-rust-f64-f32-round-correctly
        pub fn simple_rmsnorm(x: &mut [f32]) {
            let ss = x.iter().fold(0.0, |s, n| s + n * n);
            let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
            let scale = 1.0 / rms;
            // normalize and scale
            for i in 0..x.len() {
                x[i] *= scale;
            }
        }

        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new(128 * 4));
        let v1 = (1..129).map(|i| i as f32).collect::<Vec<_>>();

        let t1 = WgpuTensor::new(&v1.clone(), &[128], device.clone())?;
        let t1 = t1.rms_norm_inplace(1e-5)?;
        let mut dst1 = vec![0.0; 128];
        t1.export(&mut dst1)?;

        let mut dst2 = v1.clone();
        simple_rmsnorm(&mut dst2);

        assert_relative_eq!(&dst1[0..10], &dst2[0..10], epsilon = 1e-7);
        Ok(())
    }

    #[test]
    fn test_wgpu_matmul() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new(128 * 4));
        let v1 = (0..256).map(|i| i as f32).collect::<Vec<_>>();

        // 0.0, 1.0
        // 2.0, 3.0
        //...
        let t1 = WgpuTensor::new(&v1, &[128, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 2], &[2], device.clone())?;
        let t3 = t1.matmul(&t2)?;
        let mut dst1 = vec![0.0; 128];
        t3.export(&mut dst1)?;
        assert_eq!(dst1[0..8], vec![
            2.0, 10.0, 18.0, 26.0, 34.0, 42.0, 50.0, 58.0
        ]);
        Ok(())
    }
}
