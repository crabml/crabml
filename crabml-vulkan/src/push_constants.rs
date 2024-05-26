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
