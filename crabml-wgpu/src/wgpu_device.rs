use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crabml::tensor::Tensor;
use wgpu::util::DeviceExt;

pub struct WgpuTensorDeviceOptions {
    pub staging_buf_bytes: usize,

    pub debug_named_tensor: bool,
}

impl Default for WgpuTensorDeviceOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuTensorDeviceOptions {
    pub fn new() -> Self {
        Self {
            staging_buf_bytes: 1024 * 4,
            debug_named_tensor: false,
        }
    }

    pub fn with_staging_buf_bytes(mut self, v: usize) -> Self {
        self.staging_buf_bytes = v;
        self
    }

    pub fn with_debug_named_tensor(mut self, v: bool) -> Self {
        self.debug_named_tensor = v;
        self
    }
}

pub struct WgpuTensorDevice {
    pub(crate) opts: WgpuTensorDeviceOptions,
    pub(crate) inner: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) staging_buf: wgpu::Buffer,
    pub(crate) modules: HashMap<&'static str, wgpu::ShaderModule>,

    /// used for test only
    pub debug_tensors: RefCell<HashMap<String, Vec<f32>>>,
}

pub type WgpuTensorDeviceRef = Rc<WgpuTensorDevice>;

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

    pub(crate) fn load_modules(&mut self) {
        let module_sources = vec![
            ("add_inplace", include_str!("shaders/add.wgsl")),
            ("mul_inplace", include_str!("shaders/mul.wgsl")),
            ("div_inplace", include_str!("shaders/div.wgsl")),
            ("rms_norm_inplace", include_str!("shaders/rms_norm.wgsl")),
            ("sgemv", include_str!("shaders/sgemv.wgsl")),
            ("rope_inplace", include_str!("shaders/rope.wgsl")),
            ("softmax_inplace", include_str!("shaders/softmax.wgsl")),
            ("silu_inplace", include_str!("shaders/silu.wgsl")),
            ("batch_matmul", include_str!("shaders/batch_matmul.wgsl")),
            (
                "concatenate_inplace",
                include_str!("shaders/concatenate.wgsl"),
            ),
            ("contiguous", include_str!("shaders/contiguous.wgsl")),
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

    pub(crate) fn make_storage_buffer(&self, name: &'static str, content: &[u8]) -> wgpu::Buffer {
        self.inner
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: content,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub(crate) fn pipeline_for(&self, key: &'static str) -> wgpu::ComputePipeline {
        let module = self.modules.get(key).unwrap();
        self.inner
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
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
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();
        (device, queue)
    }

    pub fn encode_pipeline_commnad(
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
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(work_group_size.0, work_group_size.1, work_group_size.2);
        }
        encoder
    }

    pub fn record_debug_tensor(&self, name: String, tensor: &impl Tensor) {
        let mut dst = vec![0.0; tensor.strider().len()];
        tensor.export(&mut dst).unwrap();
        self.debug_tensors.borrow_mut().insert(name, dst);
    }

    pub fn dump_debug_tensor(&self, name: &str) -> Option<Vec<f32>> {
        self.debug_tensors.borrow().get(name).cloned()
    }
}
