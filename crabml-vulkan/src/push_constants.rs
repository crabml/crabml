use vulkano::buffer::BufferContents;

#[derive(BufferContents)]
#[repr(C)]
pub struct ArithmeticPushConstants {
    pub n_elms: u32,
    pub op: u32,
    pub use_scalar_rhs: u32,
    pub scalar_rhs: f32,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct SoftmaxPushConstants {
    pub n_rows: u32,
    pub n_cols: u32,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct RmsNormPushConstants {
    pub n_rows: u32,
    pub n_cols: u32,
    pub eps: f32,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct RopePushConstants {
    pub n_batch: u32,
    pub n_dims: u32,
    pub pos: u32,
    pub n_heads: u32,
    pub n_rope_dims: u32,
}

#[derive(BufferContents, Default)]
#[repr(C)]
pub struct ContiguousPushConstants {
    pub shape: [u32; 4],
    pub strides: [u32; 4],
    pub n_dims: u32,
    pub n_elms: u32,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct ConcatenatePushConstants {
    pub shape1: [u32; 4],
    pub shape2: [u32; 4],
    pub strides1: [u32; 4],
    pub strides2: [u32; 4],
    pub axis: u32,
    pub dims: u32,
    pub n_elms: u32,
}

// (M, N) x (N, K) = (M, K), now we only support K = 1
#[derive(BufferContents)]
#[repr(C)]
pub struct MatmulPushConstants {
    pub b: u32,
    pub m: u32,
    pub k: u32,
}

// (M, N, K) x (N, K) = (M, N)
#[derive(BufferContents)]
#[repr(C)]
pub struct BatchMatmuPushConstants {
    pub b: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub strides_b: [u32; 3],
}
