use std::borrow::Cow;
use std::simd::f32x32;
use std::simd::prelude::SimdFloat;

use super::buf::VecDotF32;

impl<'a> VecDotF32 for Cow<'a, [f32]> {
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
