use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::NoUninit;
use vulkano::buffer::Buffer;
use vulkano::buffer::BufferContents;
use vulkano::buffer::BufferCreateInfo;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::CopyBufferInfo;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::Device;
use vulkano::device::DeviceCreateInfo;
use vulkano::device::DeviceExtensions;
use vulkano::device::Queue;
use vulkano::device::QueueCreateInfo;
use vulkano::device::QueueFlags;
use vulkano::instance::Instance;
use vulkano::instance::InstanceCreateFlags;
use vulkano::instance::InstanceCreateInfo;
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::shader;
use vulkano::sync::GpuFuture;
use vulkano::sync::{self};
use vulkano::VulkanLibrary;

pub struct VulkanTensorDeviceOptions {
    pub staging_buf_bytes: usize,

    pub debug_named_tensor: bool,
}

impl VulkanTensorDeviceOptions {
    pub fn new() -> Self {
        Self {
            staging_buf_bytes: 1024 * 1024 * 2,
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

impl Default for VulkanTensorDeviceOptions {
    fn default() -> Self {
        Self::new()
    }
}

pub struct VulkanTensorDevice {
    pub(crate) opts: VulkanTensorDeviceOptions,
    pub(crate) inner: VulkanTensorDeviceInner,
}

pub type VulkanTensorDeviceRef = Arc<VulkanTensorDevice>;

macro_rules! load_shader_entry_point {
    ($shader_mod:ident, $device:expr, $entry_point:expr) => {
        $shader_mod::load($device)
            .unwrap()
            .entry_point($entry_point)
            .unwrap()
    };
}

impl VulkanTensorDevice {
    pub fn new(opts: VulkanTensorDeviceOptions) -> VulkanTensorDeviceRef {
        let inner = VulkanTensorDeviceInner::new(opts.staging_buf_bytes);
        let mut device = Self { opts, inner };
        device.load_shaders();
        device.into()
    }

    fn load_shaders(&mut self) {
        mod arithmetic_shader {
            vulkano_shaders::shader! { ty: "compute", path: "./src/shaders/arithmetic.comp" }
        }

        let device = self.inner.device.clone();
        let entry_points = [(
            "arithmetic",
            load_shader_entry_point!(arithmetic_shader, device.clone(), "main"),
        )];

        for (name, entry_point) in entry_points.into_iter() {
            self.inner.load_compute_pipeline(name, entry_point);
        }
    }
}

// Vulkan has a lot of boilerplate code, put it them a stand-alone struct to deal with it.
// I believe the complexity of Vulkan mostly comes from the fact that it is graphics-oriented.
//
// For compute, all we need is:
//
// 1. Create some buffers
// 2. Pass the buffers to the compute pipeline
// 3. Await the result, and read the buffer back to CPU with staging buffer.
//
// This struct is expected to be able to be used elsewhere, so it is not expected to tied to the
// Tensor struct.
pub(crate) struct VulkanTensorDeviceInner {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pipelines: HashMap<String, Arc<ComputePipeline>>,
    output_buffer: Subbuffer<[u8]>,
}

impl VulkanTensorDeviceInner {
    pub fn new(output_buffer_bytes: usize) -> Self {
        let (device, queue) = Self::init_device();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // We start by creating the buffer that will store the data.
        let output_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            output_buffer_bytes as u64,
        )
        .unwrap();

        let pipelines = HashMap::new();
        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            pipelines,
            output_buffer,
        }
    }

    pub fn make_device_buffer_from<T: NoUninit>(&self, data: &[T]) -> Subbuffer<[u8]> {
        let buf = bytemuck::cast_slice(data);
        self.make_device_buffer_from_bytes(buf)
    }

    /// create a buffer and copy the data from CPU to this buffer
    pub fn make_device_buffer_from_bytes(&self, data: &[u8]) -> Subbuffer<[u8]> {
        // this buffer is expected to be recycled after this function
        let staging_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .unwrap();

        // the newly created buffer
        let device_buffer = Buffer::new_slice(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            data.len() as u64,
        )
        .unwrap();

        // copy the data from the staging buffer to the device buffer and wait.
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging_buffer.clone(),
                device_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().expect("Failed to build command buffer");
        let finished = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .expect("Failed to execute command buffer")
            .then_signal_fence_and_flush()
            .expect("Failed to signal fence and flush");
        finished.wait(None).expect("Failed to wait for fence");

        device_buffer
    }

    // copy the buffer to staging buffer, then read the data from staging buffer to CPU.
    // this method is used to read the result of the compute operation.
    pub fn copy_device_buffer_to_cpu(&self, src: Subbuffer<[u8]>, dst: &mut [u8]) {
        let command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .copy_buffer(CopyBufferInfo::buffers(src, self.output_buffer.clone()))
                .unwrap();
            builder.build().expect("Failed to build command buffer")
        };

        // await the command buffer to finish
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .expect("Failed to execute command buffer")
            // This line instructs the GPU to signal a *fence* once the command buffer has finished
            // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
            // reached a certain point. We need to signal a fence here because below we want to block
            // the CPU until the GPU has reached that point in the execution.
            .then_signal_fence_and_flush()
            .expect("Failed to signal fence and flush")
            // Blocks execution until the GPU has finished the operation. This method only exists on the
            // future that corresponds to a signalled fence. In other words, this method wouldn't be
            // available if we didn't call `.then_signal_fence_and_flush()` earlier. The `None` parameter
            // is an optional timeout.
            .wait(None)
            .expect("Failed to wait for fence");

        // copy the data from staging buffer to CPU
        self.output_buffer
            .read()
            .unwrap()
            .iter()
            .zip(dst.iter_mut())
            .for_each(|(s, d)| *d = *s);
    }

    pub fn dispatch_compute<Pc: BufferContents>(
        &self,
        pipeline_name: &str,
        buffers: Vec<Subbuffer<[u8]>>,
        push_constants: Pc,
        dispatch_group: [u32; 3],
    ) {
        let pipeline = self.pipelines.get(pipeline_name).unwrap();

        let write_descriptor_set = buffers
            .into_iter()
            .enumerate()
            .map(|(i, buffer)| WriteDescriptorSet::buffer(i as u32, buffer))
            .collect::<Vec<_>>();

        let layout = pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            write_descriptor_set,
            [],
        )
        .unwrap();

        // build the command buffer for the compute operation
        let command_buffer = {
            // In order to execute our operation, we have to build a command buffer.
            let mut builder = AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .bind_pipeline_compute(pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap();
            builder
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .unwrap();
            builder.dispatch(dispatch_group).unwrap();
            builder.build().unwrap()
        };

        // flush the command buffer to GPU queue
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .flush()
            .unwrap();
    }

    pub fn init_device() -> (Arc<Device>, Arc<Queue>) {
        // As with other examples, the first step is to create an instance.
        let library = VulkanLibrary::new().unwrap();
        let instance = Instance::new(library, InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        })
        .unwrap();

        // Choose which physical device to use.
        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                // The Vulkan specs guarantee that a compliant implementation must provide at least one
                // queue that supports compute operations.
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // Now initializing the device.
        let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        })
        .unwrap();

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
        // example we use only one queue, so we just retrieve the first and only element of the
        // iterator and throw it away.
        let queue = queues.next().unwrap();
        (device, queue)
    }

    pub fn load_compute_pipeline(
        &mut self,
        pipeline_name: &str,
        shader_entry_point: shader::EntryPoint,
    ) {
        let pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(shader_entry_point);
            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                self.device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        self.pipelines.insert(pipeline_name.to_string(), pipeline);
    }
}
