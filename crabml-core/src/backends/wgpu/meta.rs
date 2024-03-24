#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct RmsNormMeta {
    pub n_batch: u32,
    pub n_dims: u32,
    pub eps: f32,
    pub _padding: u32,
}

// (M, N) x (N, K) = (M, K), now we only support K = 1
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct MatmulMeta {
    pub b: u32,
    pub m: u32,
    pub k: u32,
    pub _padding: u32,
}

// (M, N, K) x (N, K) = (M, N)
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C, align(16))]
pub struct BatchMatmulMeta {
    pub b: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub strides_b: [u32; 3],
    pub _padding_1: u32,
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct RopeMeta {
    pub n_batch: u32,
    pub n_dims: u32,
    pub pos: u32,
    pub n_heads: u32,
    pub n_rope_dims: u32,
    pub _padding: [u32; 7],
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C, align(16))]
pub struct ConcatenateMeta {
    pub shape1: [u32; 4],
    pub shape2: [u32; 4],
    pub strides1: [u32; 4],
    pub strides2: [u32; 4],
    pub axis: u32,
    pub dims: u32,
    pub _padding: [u32; 2],
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C, align(16))]
pub struct ContiguousMeta {
    pub shape: [u32; 4],
    pub strides: [u32; 4],
    pub n_dims: u32,
    pub n_elms: u32,
    pub _padding: [u32; 2],
}
