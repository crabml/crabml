use std::{
    borrow::Cow,
    simd::{f32x32, SimdFloat},
};

use super::BufVecDotF32;

impl BufVecDotF32 for Cow<'_, [f32]> {
    fn vec_dot_f32(&self, offset: usize, x: &[f32]) -> f32 {
        let blocks = self[offset..offset + x.len()].as_chunks::<32>().0;
        let mut acc = f32x32::splat(0.0);
        for (block_idx, block) in blocks.iter().enumerate() {
            let block = f32x32::from_slice(block);
            let x = f32x32::from_slice(&x[block_idx * 32..(block_idx + 1) * 32]);
            acc += block * x;
        }
        acc.reduce_sum()
    }
}