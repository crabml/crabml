use vulkano::buffer::BufferContents;

#[derive(BufferContents)]
#[repr(C)]
pub struct ArithmeticPushConstants {
    pub n_elms: u32,
    pub op: u32,
}
