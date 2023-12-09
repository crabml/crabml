use std::rc::Rc;

use wgpu;
use wgpu::util::DeviceExt;

use super::meta::MatmulMeta;
use super::meta::RmsNormMeta;
use super::WgpuTensorDeviceRef;
use crate::backends::wgpu::meta::BatchMatmulMeta;
use crate::backends::wgpu::meta::RopeMeta;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::TensorArithmetics;
use crate::tensor::TensorStrider;

#[derive(Clone)]
pub struct WgpuTensor {
    buf: Rc<wgpu::Buffer>,
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
            return Err((ErrorKind::TensorError, "buffer size mismatch").into());
        };
        Ok(Self {
            buf: Rc::new(buf),
            capacity: src.len(),
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

    fn alloc(shape: &[usize], capacity: Option<usize>, device: Self::Device) -> Result<Self> {
        let n_elms = shape.iter().product::<usize>();
        let capacity = capacity.unwrap_or(n_elms);
        assert!(capacity >= n_elms);

        let buf_bytes = capacity * std::mem::size_of::<f32>();
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
            capacity,
            strider,
            device,
            name: None,
        })
    }

    fn with_strider(self, strider: TensorStrider) -> Result<Self> {
        Ok(Self {
            buf: self.buf,
            capacity: self.capacity,
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

    // extend the tensor with the rhs tensor's data.
    fn extend(&mut self, rhs: &Self) -> Result<()> {
        let new_len = self.strider.len() + rhs.strider.len();
        if new_len > self.capacity {
            return Err((
                ErrorKind::TensorError,
                format!("exceeded capacity at {}", self.capacity),
            )
                .into());
        }
        if !rhs.shape().eq(&self.shape()[1..]) {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "shape mismatch on extend, want {:?} but got {:?}",
                    &self.shape()[1..],
                    &rhs.shape()
                ),
            )
                .into());
        }

        let f32_size = std::mem::size_of::<f32>();
        let copy_offset = self.strider.len() * f32_size;
        let copy_bytes_len = rhs.strider().len() * f32_size;

        // enqueue copy from rhs to self's buffer
        let mut encoder = self
            .device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &rhs.buf,
            0 as u64,
            &self.buf,
            copy_offset as u64,
            copy_bytes_len as u64,
        );
        self.device.queue.submit(Some(encoder.finish()));

        // update strider
        let new_shape = {
            let mut shape = self.shape().to_vec();
            shape[0] += 1;
            shape
        };
        self.strider = TensorStrider::new(new_shape);
        Ok(())
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
        if buf_size > self.device.opts.staging_buf_bytes {
            return Err((
                ErrorKind::TensorError,
                format!(
                    "buffer size exceeded staging buffer limit: {}",
                    self.device.opts.staging_buf_bytes
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
            dst.copy_from_slice(&bytemuck::cast_slice(&data)[0..self.strider.len()]);

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
        let mut new_tensor = Self::alloc(self.strider.shape(), None, self.device.clone())?;
        new_tensor
            .copy_from(&self, &vec![0; self.shape().len()], self.strider.len())
            .unwrap();
        Ok(new_tensor)
    }
}

impl TensorArithmetics for WgpuTensor {
    fn rope_inplace(self, pos: usize, rope_dims: usize) -> Result<Self> {
        assert!(self.shape().len() == 2);
        assert!(self.is_contiguous());

        let n_heads = self.shape()[0];
        let meta = RopeMeta {
            M: 1,
            N: self.strider.len() as u32,
            pos: pos as u32,
            n_heads: n_heads as u32,
            rope_dims: rope_dims as u32,
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
        let encoder = self
            .device
            .encode_pipeline_commnad("rope_inplace", entries, (1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));

        Ok(self)
    }

    fn rms_norm_inplace(self, eps: f32) -> Result<Self> {
        let meta_buf = self.device.make_storage_buffer(
            "meta",
            bytemuck::bytes_of(&RmsNormMeta {
                M: 1,
                N: self.strider.len() as u32,
                eps: eps,
                _padding: 0.0,
            }),
        );
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
        assert!(axis == 1);
        assert!(self.is_contiguous());
        assert!(self.shape().len() == 2);

        let m = self.shape()[0] as u32;
        let n = self.shape()[1] as u32;
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
                .encode_pipeline_commnad("softmax_inplace", entries, (m / 32 + 1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn silu_inplace(self) -> Result<Self> {
        return Err((ErrorKind::NotImplemented, "not implemented").into());
    }

    fn mul_inplace(self, rhs: &Self) -> Result<Self> {
        let meta_buf = self.device.make_storage_buffer(
            "meta",
            bytemuck::cast_slice(&[1u32, self.strider.len() as u32]),
        );
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
        let meta_buf = self.device.make_storage_buffer(
            "meta",
            bytemuck::cast_slice(&[1u32, self.strider.len() as u32]),
        );
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
        // assert!(self.strider().len() % 32 == 0);
        let meta_buf = self.device.make_storage_buffer(
            "meta",
            bytemuck::cast_slice(&[1u32, self.strider.len() as u32]),
        );
        let rhs_buf = self
            .device
            .make_storage_buffer("rhs", bytemuck::cast_slice(&[rhs]));
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
        let encoder = self
            .device
            .encode_pipeline_commnad("div_inplace", entries, (1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));
        Ok(self)
    }

    fn matmul(&self, y: &Self) -> Result<Self> {
        assert!(self.shape().len() == 2);
        assert!(self.shape()[1] == y.shape()[0]);
        assert!(y.shape().len() == 1);
        assert!(self.is_contiguous());
        assert!(y.is_contiguous());

        let output = Self::alloc(&[self.strider.shape()[0]], None, self.device.clone())?;
        let meta = MatmulMeta {
            M: self.strider.shape()[0] as u32,
            N: self.strider.shape()[1] as u32,
            K: 1,
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
        assert!(self.shape().len() == 3);
        assert!(y.shape().len() == 2);
        assert!(self.shape()[0] == y.shape()[0]);
        assert!(self.shape()[2] == y.shape()[1]);
        assert!(y.is_contiguous());

        // (m, n, k) @ (m, k) => (m, n)
        let output = Self::alloc(
            &[self.shape()[0], self.shape()[1]],
            None,
            self.device.clone(),
        )?;

        let meta = BatchMatmulMeta {
            M: self.strider.shape()[0] as u32,
            N: self.strider.shape()[1] as u32,
            K: self.strider.shape()[2] as u32,
            strides_0: [
                self.strider.strides()[0] as u32,
                self.strider.strides()[1] as u32,
                self.strider.strides()[2] as u32,
            ],
            repeats_0: [
                self.strider.repeats()[0] as u32,
                self.strider.repeats()[1] as u32,
                self.strider.repeats()[2] as u32,
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
        let encoder =
            self.device
                .encode_pipeline_commnad("batch_matmul", entries, (meta.M / 32 + 1, 1, 1));
        self.device.queue.submit(Some(encoder.finish()));

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use half::vec;

    use super::WgpuTensor;
    use crate::backends::wgpu::WgpuTensorDevice;
    use crate::backends::wgpu::WgpuTensorDeviceOptions;
    use crate::error::Result;
    use crate::tensor::Tensor;
    use crate::tensor::TensorArithmetics;
    use crate::tensor::TensorStrider;

    #[test]
    fn test_wgpu_tensor_new_and_export() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
        let t1 = WgpuTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)?;
        let mut dst = vec![0.0; 6];

        t1.export(&mut dst)?;

        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_add() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
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
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
        let t1 = WgpuTensor::new(&[3.0; 1024], &[512, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.mul_inplace(&t2)?;
        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;
        assert_eq!(&dst[0..6], [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
        assert!(dst.iter().all(|v| *v == 6.0));

        let t1 = WgpuTensor::new(&[3.0; 6], &[3, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 6], &[3, 2], device)?;
        let t1 = t1.mul_inplace(&t2)?;

        let mut dst = vec![0.0; 6];
        t1.export(&mut dst)?;
        assert_eq!(dst, vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);

        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_div_scalar() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
        let t1 = WgpuTensor::new(&[6.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.div_scalar_inplace(2.0)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;

        assert_eq!(&dst[0..3], [3.0, 3.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_alloc() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
        let t1 = WgpuTensor::alloc(&[512, 2], None, device.clone())?;
        let t2 = WgpuTensor::new(&[1.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.add_inplace(&t2)?;

        let mut dst = vec![0.0; 1024];
        t1.export(&mut dst)?;

        assert_eq!(&dst[0..3], [1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_tensor_with_name() -> Result<()> {
        let device_opts = WgpuTensorDeviceOptions::new().with_debug_named_tensor(true);
        let device = WgpuTensorDevice::new(device_opts);

        let t1 = WgpuTensor::alloc(&[512, 2], None, device.clone())?;
        let t2 = WgpuTensor::new(&[1.0; 1024], &[512, 2], device.clone())?;
        let t1 = t1.add_inplace(&t2)?;
        let _ = t1.with_name("t1".to_string());

        let dst = device.dump_debug_tensor("t1").unwrap();
        assert_eq!(dst, vec![1.0; 1024]);
        Ok(())
    }

    #[test]
    fn test_wgpu_copy_from() -> Result<()> {
        let device_opts = WgpuTensorDeviceOptions::new().with_debug_named_tensor(true);
        let device = WgpuTensorDevice::new(device_opts);

        let mut t1 = WgpuTensor::alloc(&[256, 4], None, device.clone())?;
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

        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
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
    fn test_wgpu_batch_matmul_plain_code() -> Result<()> {
        fn batch_matmul_plain_code(
            m: usize,
            n: usize,
            k: usize,
            st: TensorStrider,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
        ) {
            for mi in 0..m {
                for ni in 0..n {
                    let mut sum = 0.0;
                    for ki in 0..k {
                        let ai = mi / st.repeats()[0] * st.strides()[0]
                            + ni * st.strides()[1] / st.repeats()[1]
                            + ki * st.strides()[2] / st.repeats()[2];
                        println!(
                            "mi: {} ni: {} ai: {} st.at(): {:?}",
                            mi,
                            ni,
                            ai,
                            st.at(&[mi, ni, ki]).unwrap()
                        );
                        let av = a[ai];
                        let bv = b[k * mi + ki];
                        sum += av * bv;
                    }
                    c[mi * n + ni] = sum;
                }
            }
        }

        // m: 2, n: 3, k: 2
        let (m, n, k) = (2, 3, 2);
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // m, n, k
        let b = vec![2.0, 2.0, 2.0, 2.0]; // m, k
        let mut c = vec![0.0; 6]; // m x n
        let st = TensorStrider::new(vec![1, 3, 2]).repeat(vec![2, 1, 1])?;
        batch_matmul_plain_code(m, n, k, st, &a, &b, &mut c);

        assert_eq!(c, vec![2.0, 10.0, 18.0, 2.0, 10.0, 18.0]);
        Ok(())
    }

    #[test]
    fn test_wgpu_matmul() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
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

    #[test]
    fn test_wgpu_batch_matmul() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());

        let v1 = (0..6).map(|i| i as f32).collect::<Vec<_>>();
        // 0.0, 1.0,
        // 2.0, 3.0,
        // 4.0, 5.0
        // @
        // 2.0, 2.0
        // => 2.0, 10.0, 18.0

        let t1 = WgpuTensor::new(&v1, &[1, 3, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 2], &[1, 2], device.clone())?;
        let t3 = t1.batch_matmul(&t2)?;
        let mut dst1 = vec![0.0; 3]; // 1 x 3
        t3.export(&mut dst1)?;
        // assert_eq!(t1.strider().strides(), &[6, 2, 1]);
        // assert_eq!(dst1, vec![2.0, 10.0, 18.0]);

        let v1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let t1 = WgpuTensor::new(&v1, &[2, 3, 2], device.clone())?;
        let t2 = WgpuTensor::new(&[2.0; 4], &[2, 2], device.clone())?;
        let t3 = t1.batch_matmul(&t2)?;
        let mut dst1 = vec![0.0; 6]; // 2 x 3
        t3.export(&mut dst1)?;
        assert_eq!(t1.strider().shape(), &[2, 3, 2]);
        assert_eq!(t1.strider().strides(), &[6, 2, 1]);
        // assert_eq!(dst1, vec![2.0, 10.0, 18.0, 2.0, 10.0, 18.0]);

        let v1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let t1 = WgpuTensor::new(&v1, &[1, 3, 2], device.clone())?.repeat(&[2, 1, 1])?;
        let t2 = WgpuTensor::new(&[2.0; 4], &[2, 2], device.clone())?;
        let t3 = t1.batch_matmul(&t2)?;
        let mut dst1 = vec![0.0; 6]; // 2 x 3
        t3.export(&mut dst1)?;
        assert_eq!(t1.strider().shape(), &[2, 3, 2]);
        assert_eq!(t1.strider().strides(), &[6, 2, 1]);
        assert_eq!(dst1, vec![2.0, 10.0, 18.0, 2.0, 10.0, 18.0]);

        Ok(())
    }

    #[test]
    fn test_wgpu_rope() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
        let v1 = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let t1 = WgpuTensor::new(&v1, &[2, 16], device.clone())?;
        let t1 = t1.rope_inplace(1, 2)?;

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
    fn test_wgpu_extend() -> Result<()> {
        let device = WgpuTensorDevice::new(WgpuTensorDeviceOptions::new());
        let mut t1 = WgpuTensor::alloc(&[0, 16], Some(1024), device.clone())?;

        let v2 = (0..16).map(|i| i as f32).collect::<Vec<_>>();
        let t2 = WgpuTensor::new(&v2, &[16], device.clone())?;

        let v3 = (100..116).map(|i| i as f32).collect::<Vec<_>>();
        let t3 = WgpuTensor::new(&v3, &[16], device.clone())?;

        t1.extend(&t2)?;
        t1.extend(&t3)?;

        let mut dst1 = vec![0.0; 32];
        t1.export(&mut dst1)?;

        assert_eq!(t1.shape(), &[2, 16]);
        assert_relative_eq!(
            &dst1[..],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                111.0, 112.0, 113.0, 114.0, 115.0
            ][..],
            epsilon = 1e-5
        );
        Ok(())
    }
}
