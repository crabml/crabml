use std::rc::Rc;

use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGMLType;
use crabml::tensor::RopeMode;
use crabml::tensor::Tensor;
use crabml::tensor::TensorStrider;
use wgpu::util::DeviceExt;

use super::meta::ConcatenateMeta;
use super::meta::MatmulMeta;
use super::meta::RmsNormMeta;
use super::WgpuTensorDeviceRef;
use crate::meta::BatchMatmulMeta;
use crate::meta::ContiguousMeta;
use crate::meta::RopeMeta;

#[derive(Clone)]
pub struct WgpuTensor {
    buf: Rc<wgpu::Buffer>,
    dtype: GGMLType,
    capacity: usize, // max element count
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
            return Err((ErrorKind::TensorError, "new: buffer size mismatch").into());
        };
        Ok(Self {
            buf: Rc::new(buf),
            capacity: src.len(),
            dtype: GGMLType::F32,
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
    type DeviceRef = WgpuTensorDeviceRef;

    fn from_cpu(
        buf: &[u8],
        shape: &[usize],
        dtype: GGMLType,
        device: Self::DeviceRef,
    ) -> Result<Self> {
        let buf = device
            .inner
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor weights buffer"),
                contents: buf,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let strider = TensorStrider::new(shape.to_vec());
        Ok(Self {
            buf: Rc::new(buf),
            dtype,
            capacity: strider.len(),
            strider,
            device,
            name: None,
        })
    }

    fn alloc(shape: &[usize], dtype: GGMLType, device: Self::DeviceRef) -> Result<Self> {
        assert!(dtype == GGMLType::F32, "wgpu tensor only support F32 yet");
        let n_elms = shape.iter().product::<usize>();

        let buf_bytes = n_elms * std::mem::size_of::<f32>();
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
            dtype: GGMLType::F32,
            capacity: n_elms,
            strider,
            device,
            name: None,
        })
    }

    fn resize(self, axis: usize, n: usize) -> Result<Self> {
        if axis >= self.shape().len() {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "resize: axis {} is larger than the current shape {:?}",
                    axis,
                    self.shape()
                ),
            )
                .into());
        }

        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = n;

        let new_len: usize = new_shape.iter().product();
        if new_len > self.capacity {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "resize: new shape {:?} is larger than the current shape {:?}",
                    new_shape,
                    self.shape()
                ),
            )
                .into());
        }

        let new_strider = self.strider.resize(&new_shape)?;
        Ok(Self {
            buf: self.buf,
            capacity: self.capacity,
            dtype: self.dtype,
            strider: new_strider,
            device: self.device.clone(),
            name: None,
        })
    }

    fn dtype(&self) -> GGMLType {
        self.dtype
    }

    fn with_strider(self, strider: TensorStrider) -> Result<Self> {
        Ok(Self {
            buf: self.buf,
            capacity: self.capacity,
            dtype: self.dtype,
            strider,
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

    fn transpose(self, dims: &[usize]) -> Result<Self> {
        let strider = self.strider.transpose(dims)?;
        self.with_strider(strider)
    }

    fn strider(&self) -> &TensorStrider {
        &self.strider
    }

    fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    fn concatenate(&mut self, rhs: &Self, axis: usize) -> Result<()> {
        if self.shape().len() != 3 {
            return Err((
                ErrorKind::TensorError,
                "only support 3D tensor concatenation yet",
            )
                .into());
        }
        if self.dtype() != GGMLType::F32 || rhs.dtype() != GGMLType::F32 {
            return Err((ErrorKind::TensorError, "concatenate: only support f32 yet").into());
        }

        let meta = ConcatenateMeta {
            shape1: [
                self.strider.shape()[0] as u32,
                self.strider.shape()[1] as u32,
                self.strider.shape()[2] as u32,
                0,
            ],
            shape2: [
                rhs.strider.shape()[0] as u32,
                rhs.strider.shape()[1] as u32,
                rhs.strider.shape()[2] as u32,
                0,
            ],
            strides1: [
                self.strider.strides()[0] as u32,
                self.strider.strides()[1] as u32,
                self.strider.strides()[2] as u32,
                0,
            ],
            strides2: [
                rhs.strider.strides()[0] as u32,
                rhs.strider.strides()[1] as u32,
                rhs.strider.strides()[2] as u32,
                0,
            ],
            axis: axis as u32,
            dims: 3,
            _padding: [0; 2],
        };

        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::bytes_of(&meta));
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
            "concatenate_inplace",
            entries,
            (rhs.strider.len() as u32 / 16, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));

        let mut new_shape = self.strider.shape().to_vec();
        new_shape[axis] += rhs.strider.shape()[axis];
        self.strider = self.strider.resize(&new_shape)?;
        Ok(())
    }

    fn copy_rows_from(&mut self, src: &Self, src_rows: &[usize]) -> Result<()> {
        // TODO: check is_owned
        if !self.is_contiguous() {
            return Err((ErrorKind::TensorError, "not contiguous").into());
        }
        assert!(src.strider.dims() == 2);

        let n_dims = src.shape().last().unwrap();
        let f32_bytes = std::mem::size_of::<f32>();

        for (dst_row, src_row) in src_rows.iter().enumerate() {
            let dst_offset = dst_row * n_dims * f32_bytes;
            let src_offset = src_row * n_dims * f32_bytes;
            let row_bytes = n_dims * f32_bytes;

            // enqueue copy from rhs to self's buffer
            let mut encoder = self
                .device
                .inner
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(
                &src.buf,
                src_offset as u64,
                &self.buf,
                dst_offset as u64,
                row_bytes as u64,
            );
            self.device.queue.submit(Some(encoder.finish()));
        }

        Ok(())
    }

    fn export(&self, dst: &mut [f32]) -> Result<()> {
        let buf_size = std::mem::size_of_val(dst);
        if buf_size > self.device.opts.staging_buf_bytes {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "buffer size exceeded staging buffer limit: {}, got: {}",
                    self.device.opts.staging_buf_bytes, buf_size,
                ),
            )
                .into());
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
            dst.copy_from_slice(&bytemuck::cast_slice(&data)[0..dst.len()]);

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
        let new_tensor = Self::alloc(self.strider.shape(), self.dtype, self.device.clone())?;

        let mut encoder = self
            .device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buf, 0, &new_tensor.buf, 0, self.buf.size());
        self.device.queue.submit(Some(encoder.finish()));
        Ok(new_tensor)
    }

    fn rope_inplace(self, mode: RopeMode, pos: usize, rope_dims: usize) -> Result<Self> {
        assert!(self.shape().len() == 3 || self.shape().len() == 2);
        assert!(self.is_contiguous());
        assert!(mode == RopeMode::Llama, "TODO: only support Llama mode yet");

        let (rows, n_head, m) = if self.strider.dims() == 3 {
            (
                self.shape()[0],
                self.shape()[1],
                self.shape()[1] * self.shape()[2],
            )
        } else {
            (1, self.shape()[0], self.shape()[0] * self.shape()[1])
        };
        let meta = RopeMeta {
            n_batch: rows as u32,
            n_dims: m as u32,
            pos: pos as u32,
            n_heads: n_head as u32,
            n_rope_dims: rope_dims as u32,
            _padding: [0; 7],
        };

        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::bytes_of(&meta));
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
            "rope_inplace",
            entries,
            (rows as u32 / 32 + 1, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));

        Ok(self)
    }

    fn rms_norm_inplace(self, eps: f32) -> Result<Self> {
        assert!(self.strider.dims() == 2 || self.strider.dims() == 1);
        let (n_batch, n_dims) = if self.strider.dims() == 2 {
            (self.shape()[0], self.shape()[1])
        } else {
            (1, self.shape()[0])
        };
        let meta = &RmsNormMeta {
            n_batch: n_batch as u32,
            n_dims: n_dims as u32,
            eps,
            _padding: 0,
        };
        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::bytes_of(meta));
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
        let encoder =
            self.device
                .encode_pipeline_commnad("rms_norm_inplace", entries, (meta.n_batch, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn softmax_inplace(self, axis: usize) -> Result<Self> {
        assert!(axis == self.strider.dims() - 1);
        assert!(self.is_contiguous());
        assert!(self.shape().len() == 3 || self.shape().len() == 2);

        let (m, n) = if self.strider.dims() == 3 {
            (
                (self.shape()[0] * self.shape()[1]) as u32,
                self.shape()[2] as u32,
            )
        } else {
            (self.shape()[0] as u32, self.shape()[1] as u32)
        };
        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::cast_slice(&[m, n]));
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
        let encoder =
            self.device
                .encode_pipeline_commnad("softmax_inplace", entries, (m * n / 16 + 1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn silu_inplace(self) -> Result<Self> {
        assert!(self.is_contiguous());

        let elms = self.strider().len();
        let entries = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: self.buf.as_entire_binding(),
        }];
        let encoder = self.device.encode_pipeline_commnad(
            "silu_inplace",
            entries,
            ((elms / 32 + 1) as u32, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn gelu_inplace(self) -> Result<Self> {
        assert!(self.is_contiguous());

        let elms = self.strider().len();
        let entries = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: self.buf.as_entire_binding(),
        }];
        let encoder = self.device.encode_pipeline_commnad(
            "gelu_inplace",
            entries,
            ((elms / 32 + 1) as u32, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn mul_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.is_contiguous());
        assert!(rhs.is_contiguous());
        let n_elms = self.strider.len();
        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::cast_slice(&[n_elms as u32]));
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
            (n_elms as u32 / 32 + 1, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn add_inplace(self, rhs: &Self) -> Result<Self> {
        assert!(self.is_contiguous());
        assert!(rhs.is_contiguous());
        let n_elms = self.strider.len();
        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::cast_slice(&[n_elms as u32]));
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
            "add_inplace",
            entries,
            (n_elms as u32 / 32 + 1, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn scale_inplace(self, rhs: f32) -> Result<Self> {
        // assert!(self.strider().len() % 32 == 0);
        assert!(self.is_contiguous());
        let n_elms = self.strider.len();
        let rhs_buf = self
            .device
            .make_storage_buffer("rhs", bytemuck::cast_slice(&[rhs]));
        // TODO: make uniform buffer for meta
        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::cast_slice(&[n_elms as u32]));
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
            "mul_inplace",
            entries,
            (n_elms as u32 / 32 + 1, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    // (m, k) @ (b, k) => (b, m)
    fn matmul_vec(&self, rhs: &Self) -> Result<Self> {
        assert!(self.shape().len() == 2);
        assert!(self.shape().last() == rhs.shape().last());
        assert!(self.is_contiguous());
        assert!(rhs.is_contiguous());

        let output = Self::alloc(
            &[rhs.strider.shape()[0], self.strider.shape()[0]],
            GGMLType::F32,
            self.device.clone(),
        )?;
        let meta = MatmulMeta {
            b: rhs.strider.shape()[0] as u32,
            m: self.strider.shape()[0] as u32,
            k: self.strider.shape()[1] as u32,
            _padding: 0,
        };

        let meta_buf = self
            .device
            .make_storage_buffer("meta", bytemuck::bytes_of(&meta));
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buf.as_entire_binding(),
            },
        ];
        assert!(meta.m / 32 < 65535); // vulkan limit each dimension to 65535
        let encoder =
            self.device
                .encode_pipeline_commnad("sgemv", entries, (meta.b, meta.m / 32, 1));
        self.device.queue.submit(Some(encoder.finish()));

        Ok(output)
    }

    /// (b, m, k) @ (b, k, n) => (b, m, n)
    /// the A matrix is dense and the B matrix is allowed to be strided
    fn batch_matmul(&self, y: &Self) -> Result<Self> {
        assert!(self.shape().len() == 3);
        assert!(y.shape().len() == 3);
        assert!(self.shape()[0] == y.shape()[0]);
        assert!(self.shape()[2] == y.shape()[1]);
        assert!(self.is_contiguous());

        // (b, m, k) @ (b, k, n) => (b, m, n)
        let output = Self::alloc(
            &[y.shape()[0], self.shape()[1], y.shape()[2]],
            GGMLType::F32,
            self.device.clone(),
        )?;

        let meta = BatchMatmulMeta {
            b: y.shape()[0] as u32,
            m: self.shape()[1] as u32,
            k: self.shape()[2] as u32,
            n: y.shape()[2] as u32,
            strides_b: [
                y.strider.strides()[0] as u32,
                y.strider.strides()[1] as u32,
                y.strider.strides()[2] as u32,
            ],
            ..Default::default()
        };
        let meta_bytes = bytemuck::bytes_of(&meta);

        let meta_buf = self.device.make_storage_buffer("meta", meta_bytes);
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
        let encoder = self.device.encode_pipeline_commnad(
            "batch_matmul",
            entries,
            (meta.b * meta.m * meta.n / 32 + 1, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));

        Ok(output)
    }

    fn contiguous(self) -> Result<Self> {
        assert!(self.strider.dims() == 3 || self.strider.dims() == 2);
        if self.is_contiguous() {
            return Ok(self);
        }

        let n_elms = self.strider.len();
        let output = Self::alloc(self.strider.shape(), self.dtype, self.device.clone())?;
        let mut meta = ContiguousMeta {
            shape: [0; 4],
            strides: [0; 4],
            n_dims: self.strider.dims() as u32,
            n_elms: n_elms as u32,
            _padding: [0; 2],
        };
        for i in 0..self.strider.dims() {
            meta.shape[i] = self.strider.shape()[i] as u32;
            meta.strides[i] = self.strider.strides()[i] as u32;
        }
        let meta_bytes = bytemuck::bytes_of(&meta);
        let meta_buf = self.device.make_storage_buffer("meta", meta_bytes);

        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: meta_buf.as_entire_binding(),
            },
        ];
        let encoder = self.device.encode_pipeline_commnad(
            "contiguous",
            entries,
            (n_elms as u32 / 32 + 1, 1, 1),
        );
        self.device.queue.submit(Some(encoder.finish()));

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use approx::assert_relative_eq;
    use crabml::error::Result;
    use crabml::gguf::GGMLType;
    use crabml::tensor::RopeMode;
    use crabml::tensor::Tensor;

    use super::WgpuTensor;
    use crate::WgpuTensorDevice;
    use crate::WgpuTensorDeviceOptions;
    use crate::WgpuTensorDeviceRef;

    #[thread_local]
    static DEVICE: LazyLock<WgpuTensorDeviceRef> = LazyLock::new(|| {
        WgpuTensorDevice::new(WgpuTensorDeviceOptions::new().with_debug_named_tensor(true))
    });

    #[test]
    fn test_wgpu_tensor_new_and_export() -> Result<()> {
        // let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
        let t1 = WgpuTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DEVICE.clone())?;
        let mut dst = vec![0.0; 6];

        t1.export(&mut dst)?;

        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_add() -> Result<()> {
        let t1 = WgpuTensor::new(&[2.0; 64], &[16, 4], DEVICE.clone())?;
        let t2 = WgpuTensor::new(&[3.0; 64], &[16, 4], DEVICE.clone())?;
        let t1 = t1.add_inplace(&t2)?;

        let mut dst = vec![0.0; 64];
        t1.export(&mut dst)?;

        assert_eq!(dst, vec![5.0; 64]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_mul() -> Result<()> {
        let t1 = WgpuTensor::new(&[3.0; 1024], &[512, 2], DEVICE.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 1024], &[512, 2], DEVICE.clone())?;
        let t1 = t1.mul_inplace(&t2)?;
        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;
        assert_eq!(&dst[0..6], [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
        assert!(dst.iter().all(|v| *v == 6.0));

        let t1 = WgpuTensor::new(&[3.0; 6], &[3, 2], DEVICE.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 6], &[3, 2], DEVICE.clone())?;
        let t1 = t1.mul_inplace(&t2)?;

        let mut dst = vec![0.0; 6];
        t1.export(&mut dst)?;
        assert_eq!(dst, vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);

        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_alloc() -> Result<()> {
        let t1 = WgpuTensor::alloc(&[512, 2], GGMLType::F32, DEVICE.clone())?;
        let t2 = WgpuTensor::new(&[1.0; 1024], &[512, 2], DEVICE.clone())?;
        let t1 = t1.add_inplace(&t2)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;

        assert_eq!(&dst[0..3], [1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_with_name() -> Result<()> {
        let t1 = WgpuTensor::new(&[0.0; 1024], &[512, 2], DEVICE.clone())?;
        let t2 = WgpuTensor::new(&[1.0; 1024], &[512, 2], DEVICE.clone())?;
        let t1 = t1.add_inplace(&t2)?;
        let _ = t1.with_name("t1".to_string());

        let dst = DEVICE.dump_debug_tensor("t1").unwrap();
        assert_eq!(dst, vec![1.0; 1024]);
        Ok(())
    }

    #[test]
    fn test_wgpu_copy_from() -> Result<()> {
        let mut t1 = WgpuTensor::alloc(&[256, 4], GGMLType::F32, DEVICE.clone())?;
        let t2 = WgpuTensor::new(
            &(0..1024).map(|d| d as f32).collect::<Vec<f32>>(),
            &[256, 4],
            DEVICE.clone(),
        )?;

        assert_eq!(t2.strider.at(&[1, 0])?, 4);
        t1.copy_rows_from(&t2, &[1])?;

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
            for i in x {
                *i *= scale;
            }
        }

        let v1 = (1..129).map(|i| i as f32).collect::<Vec<_>>();

        let t1 = WgpuTensor::new(&v1.clone(), &[128], DEVICE.clone())?;
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
        let v1 = (0..256).map(|i| i as f32).collect::<Vec<_>>();

        let t1 = WgpuTensor::new(&v1, &[32, 8], DEVICE.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 8], &[8], DEVICE.clone())?;
        let t3 = t1.matmul_vec(&t2)?;
        let mut dst1 = vec![0.0; 32];
        t3.export(&mut dst1)?;
        assert_eq!(dst1[0..32], vec![
            56.0, 184.0, 312.0, 440.0, 568.0, 696.0, 824.0, 952.0, 1080.0, 1208.0, 1336.0, 1464.0,
            1592.0, 1720.0, 1848.0, 1976.0, 2104.0, 2232.0, 2360.0, 2488.0, 2616.0, 2744.0, 2872.0,
            3000.0, 3128.0, 3256.0, 3384.0, 3512.0, 3640.0, 3768.0, 3896.0, 4024.0
        ]);
        Ok(())
    }

    #[test]
    fn test_wgpu_batch_matmul() -> Result<()> {
        let v1 = (0..6).map(|i| i as f32).collect::<Vec<_>>();
        // 0.0, 1.0,
        // 2.0, 3.0,
        // 4.0, 5.0
        // @
        // 2.0, 2.0
        // => 2.0, 10.0, 18.0

        let t1 = WgpuTensor::new(&v1, &[1, 3, 2], DEVICE.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 2], &[1, 2, 1], DEVICE.clone())?;
        let t3 = t1.batch_matmul(&t2)?;
        let mut dst1 = vec![0.0; 3]; // 1 x 3 x 1
        t3.export(&mut dst1)?;
        assert_eq!(t1.strider().strides(), &[6, 2, 1]);
        assert_eq!(dst1, vec![2.0, 10.0, 18.0]);

        Ok(())
    }

    #[test]
    fn test_wgpu_rope() -> Result<()> {
        let v1 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t1 = WgpuTensor::new(&v1, &[2, 16], DEVICE.clone())?;
        let t1 = t1.rope_inplace(RopeMode::Llama, 1, 2)?;

        let mut dst1 = vec![0.0; 32];
        t1.export(&mut dst1)?;

        assert_relative_eq!(
            &dst1[..],
            &[
                -0.841471, 0.54030234, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, -5.6601696, 22.648676, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0
            ][..],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_wgpu_concatenate() -> Result<()> {
        // TODO: fix this test later
        let mut t1 = WgpuTensor::alloc(&[2, 2, 16], GGMLType::F32, DEVICE.clone())?.resize(0, 0)?;

        let v2 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t2 = WgpuTensor::new(&v2, &[1, 2, 16], DEVICE.clone())?;

        let v3 = (32..64).map(|i| i as f32).collect::<Vec<_>>();
        let t3 = WgpuTensor::new(&v3, &[1, 2, 16], DEVICE.clone())?;

        t1.concatenate(&t2, 0)?;
        t1.concatenate(&t3, 0)?;

        let mut dst1 = vec![0.0; 64];
        t1.export(&mut dst1)?;

        assert_eq!(t1.shape(), &[2, 2, 16]);
        assert_relative_eq!(
            &dst1[..],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
                29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0,
                43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
                57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0
            ][..],
            epsilon = 1e-5
        );
        Ok(())
    }

    #[test]
    fn test_wgpu_concatenate2() -> Result<()> {
        // TODO: fix this test later
        let mut t1 = WgpuTensor::alloc(&[2, 2, 16], GGMLType::F32, DEVICE.clone())?.resize(1, 0)?;

        let v2 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t2 = WgpuTensor::new(&v2, &[2, 1, 16], DEVICE.clone())?;

        let v3 = (32..64).map(|i| i as f32).collect::<Vec<_>>();
        let t3 = WgpuTensor::new(&v3, &[2, 1, 16], DEVICE.clone())?;

        t1.concatenate(&t2, 1)?;
        t1.concatenate(&t3, 1)?;

        let mut dst1 = vec![0.0; 64];
        t1.export(&mut dst1)?;

        assert_eq!(t1.shape(), &[2, 2, 16]);
        assert_relative_eq!(
            &dst1[..],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
                45.0, 46.0, 47.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
                27.0, 28.0, 29.0, 30.0, 31.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
                57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0
            ][..],
            epsilon = 1e-5
        );
        Ok(())
    }

    #[test]
    fn test_wgpu_softmax() -> Result<()> {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = WgpuTensor::new(&v1, &[2, 3], DEVICE.clone())?;
        let t1 = t1.softmax_inplace(1)?;

        let mut dst1 = vec![0.0; 6];
        t1.export(&mut dst1)?;

        assert_relative_eq!(
            &dst1[..],
            &[
                0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094
            ][..],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_wgpu_silu() -> Result<()> {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = WgpuTensor::new(&v1, &[6], DEVICE.clone())?;
        let t1 = t1.silu_inplace()?;

        let mut dst1 = vec![0.0; 6];
        t1.export(&mut dst1)?;

        assert_relative_eq!(
            &dst1[..],
            &[
                0.7310586, 1.761594, 2.8577225, 3.928055, 4.9665356, 5.9851646
            ][..],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_dup() -> Result<()> {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = WgpuTensor::new(&v1, &[2, 3], DEVICE.clone())?;
        let t2 = t1.dup()?;

        let mut dst1 = vec![0.0; 6];
        t2.export(&mut dst1)?;

        assert_relative_eq!(
            &dst1[..],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_congiguous() -> Result<()> {
        // 1, 2, 3
        // 4, 5, 6
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = WgpuTensor::new(&v1, &[2, 3], DEVICE.clone())?;
        let t1 = t1.transpose(&[1, 0])?;
        let t2 = t1.contiguous()?;
        // 1, 4
        // 2, 5
        // 3, 6

        let mut dst1 = vec![0.0; 6];
        t2.export(&mut dst1)?;

        assert_eq!(t2.strider.shape(), &[3, 2]);
        assert_eq!(t2.strider.dims(), 2);
        assert_relative_eq!(
            &dst1[..],
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0][..],
            epsilon = 1e-5
        );

        Ok(())
    }
}
