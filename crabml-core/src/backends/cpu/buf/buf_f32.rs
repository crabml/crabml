use std::borrow::Cow;
use std::simd::f32x32;
use std::simd::prelude::SimdFloat;
use std::slice;

pub fn f32_buf_from_bytes<'a>(buf: &[u8]) -> Cow<'a, [f32]> {
    let len = buf.len();
    assert_eq!(
        len % std::mem::size_of::<f32>(),
        0,
        "Length of slice must be multiple of f32 size"
    );
    let new_len = len / std::mem::size_of::<f32>();
    let ptr = buf.as_ptr() as *const f32;
    let f32_buf = unsafe { slice::from_raw_parts(ptr, new_len) };
    f32_buf.into()
}

pub fn vec_dot_f32_f32(a: &[f32], a_offset: usize, b: &[f32], b_offset: usize, len: usize) -> f32 {
    let ac = &a[a_offset..a_offset + len];
    let bc = &b[b_offset..b_offset + len];
    let mut sum = 0.0;
    for i in 0..len {
        sum += ac[i] * bc[i];
    }
    sum
}
