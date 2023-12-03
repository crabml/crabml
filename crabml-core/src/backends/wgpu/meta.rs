use bytemuck;

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct RmsNormMeta {
    pub M: u32,
    pub N: u32,
    pub eps: f32,
    pub _padding: f32,
}

// (M, N) x (N, K) = (M, K), now we only support K = 1
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct MatmulMeta {
    pub M: u32,
    pub N: u32,
    pub K: u32,
    pub _padding: u32,
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct RopeMeta {
    pub M: u32,
    pub N: u32,
    pub pos: u32,
    pub n_heads: u32,
    pub rope_dims: u32,
    pub _padding: [u32; 7],
}
