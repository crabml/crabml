use std::borrow::Cow;
use std::simd::f32x32;
use std::simd::prelude::SimdFloat;
use std::slice;

use super::buf::CpuTensorBufVecDot;

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

impl<'a> CpuTensorBufVecDot for Cow<'a, [f32]> {
    fn vec_dot_f32(&self, offset: usize, x: &[f32]) -> f32 {
        let chunks = self[offset..offset + x.len()].chunks(32);
        let mut acc = f32x32::splat(0.0);
        for (chunk_idx, chunk) in chunks.enumerate() {
            let block = f32x32::from_slice(chunk);
            let x = f32x32::from_slice(&x[chunk_idx * 32..(chunk_idx + 1) * 32]);
            acc += block * x;
        }
        acc.reduce_sum()
    }
}
