use std::borrow::Cow;
use std::simd::f32x32;
use std::simd::SimdFloat;

use super::buf::BlockVecCompute;

impl<'a> BlockVecCompute for Cow<'a, [f32]> {
    type BlockType = [f32; 32];

    fn block_elms(&self) -> usize {
        32
    }

    fn blocks_between(&self, start: usize, end: usize) -> &[Self::BlockType] {
        let start = start * self.block_elms();
        let end = end * self.block_elms();
        self[start..end].as_chunks().0
    }

    fn vec_dot_f32(&self, row: &[Self::BlockType], x: &[f32]) -> f32 {
        let mut acc = f32x32::splat(0.0);
        for (block_idx, block) in row.iter().enumerate() {
            let block = f32x32::from_slice(block);
            let x = f32x32::from_slice(&x[block_idx * 32..(block_idx + 1) * 32]);
            acc += block * x;
        }
        acc.reduce_sum()
    }
}
