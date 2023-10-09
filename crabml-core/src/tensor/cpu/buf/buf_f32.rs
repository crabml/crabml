use std::{simd::{SimdFloat, f32x32}, borrow::Cow, slice};

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
        for (i, block) in row.iter().enumerate() {
            let block = f32x32::from_slice(block);
            let x = f32x32::splat(x[i]);
            acc += block * x;
        }
        acc.reduce_sum()
    }
}