use std::borrow::Cow;
use std::simd::f32x8;
use std::simd::i32x4;
use std::simd::prelude::SimdFloat;
use std::simd::SimdInt;

use half::f16;

#[derive(Debug, Clone)]
pub struct QuantBufQ8_0<'a> {
    blocks: Cow<'a, [BlockQ8_0]>,
}

impl<'a> QuantBufQ8_0<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ8_0>();
        assert!(
            data.len() % blk_size == 0,
            "data length must be a multiple of QuantBlockQ8_0 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8_0, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let mut bs: Vec<BlockQ8_0> = vec![];
        let chunks = data.chunks(32);
        for chunk in chunks {
            let mut max = f32::MIN;
            for i in 0..32 {
                if chunk[i].abs() > max {
                    max = chunk[i].abs();
                }
            }
            let d = f16::from_f32(max / 127.0);
            let mut qs = [0_i8; 32];
            for i in 0..32 {
                qs[i] = (chunk[i] / d.to_f32()).round() as i8;
            }
            bs.push(BlockQ8_0 { d, qs })
        }
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ8_0] {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * 32
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert!(start % 32 == 0);

        let block_start = start / 32;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0.0; 32];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot_f32(&self, offset: usize, x: &[f32]) -> f32 {
        let row = &self.blocks[offset / 32..(offset + x.len()) / 32];
        assert!(row.len() * 32 == x.len());
        let mut sum = 0.0;
        for i in 0..row.len() {
            let block = &row[i];
            let d = block.d.to_f32();
            let mut sum_block = 0.0;
            for j in 0..4 {
                let qs = &block.qs[j * 8..(j + 1) * 8];
                let qv = f32x8::from_array([
                    qs[0] as f32,
                    qs[1] as f32,
                    qs[2] as f32,
                    qs[3] as f32,
                    qs[4] as f32,
                    qs[5] as f32,
                    qs[6] as f32,
                    qs[7] as f32,
                ]);
                let xv = f32x8::from_slice(&x[i * 32 + j * 8..i * 32 + (j + 1) * 8]);
                sum_block += (qv * xv).reduce_sum();
            }
            sum += sum_block * d;
        }
        sum
    }

    #[cfg(not(all(target_feature = "neon")))]
    pub fn vec_dot(&self, offset: usize, b: &Self) -> f32 {
        let abs = &self.blocks[offset / 32..((offset + b.len()) / 32)];
        assert!(abs.len() == b.blocks().len());

        let bbs = b.blocks();
        let mut sumf: f32 = 0.0;
        for i in 0..bbs.len() {
            let mut sumi: i32 = 0;
            for j in 0..8 {
                let ax = i32x4::from_array([
                    abs[i].qs[j * 4] as i32,
                    abs[i].qs[j * 4 + 1] as i32,
                    abs[i].qs[j * 4 + 2] as i32,
                    abs[i].qs[j * 4 + 3] as i32,
                ]);
                let bx = i32x4::from_array([
                    bbs[i].qs[j * 4] as i32,
                    bbs[i].qs[j * 4 + 1] as i32,
                    bbs[i].qs[j * 4 + 2] as i32,
                    bbs[i].qs[j * 4 + 3] as i32,
                ]);
                sumi += (ax * bx).reduce_sum();
            }
            sumf += sumi as f32 * abs[i].d.to_f32() * bbs[i].d.to_f32();
        }

        sumf
    }

    #[cfg(target_feature = "neon")]
    pub fn vec_dot(&self, offset: usize, b: &Self) -> f32 {
        let abs = &self.blocks[offset / 32..((offset + b.len()) / 32)];
        assert!(abs.len() == b.blocks().len());
        assert!(
            abs.len() % 2 == 0,
            "abs len must be a multiple of 2, got: {}",
            abs.len()
        );
        let bbs = b.blocks();

        unsafe {
            use std::arch::aarch64;
            let mut sumv0 = aarch64::vdupq_n_f32(0.0);
            let mut sumv1 = aarch64::vdupq_n_f32(0.0);
            let zerov = aarch64::vdupq_n_s32(0);

            for i in (0..bbs.len()).step_by(2) {
                let ab0 = abs.get_unchecked(i);
                let ab1 = abs.get_unchecked(i + 1);
                let bb0 = bbs.get_unchecked(i);
                let bb1 = bbs.get_unchecked(i + 1);

                let av00 = aarch64::vld1q_s8(ab0.qs.as_ptr());
                let av01 = aarch64::vld1q_s8(ab0.qs.as_ptr().add(16));
                let av10 = aarch64::vld1q_s8(ab1.qs.as_ptr());
                let av11 = aarch64::vld1q_s8(ab1.qs.as_ptr().add(16));

                let bv00 = aarch64::vld1q_s8(bb0.qs.as_ptr());
                let bv01 = aarch64::vld1q_s8(bb0.qs.as_ptr().add(16));
                let bv10 = aarch64::vld1q_s8(bb1.qs.as_ptr());
                let bv11 = aarch64::vld1q_s8(bb1.qs.as_ptr().add(16));

                sumv0 = aarch64::vmlaq_n_f32(
                    sumv0,
                    aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                        aarch64::vdotq_s32(zerov, av00, bv00),
                        aarch64::vdotq_s32(zerov, av01, bv01),
                    )),
                    f16::to_f32(ab0.d) * f16::to_f32(bb0.d),
                );

                sumv1 = aarch64::vmlaq_n_f32(
                    sumv1,
                    aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                        aarch64::vdotq_s32(zerov, av10, bv10),
                        aarch64::vdotq_s32(zerov, av11, bv11),
                    )),
                    f16::to_f32(ab1.d) * f16::to_f32(bb1.d),
                );
            }

            aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
        }
    }

    fn vec_dot_naive(&self, offset: usize, b: &Self) -> f32 {
        let abs = &self.blocks[offset / 32..((offset + b.len()) / 32)];
        assert!(abs.len() == b.blocks().len());

        let bbs = b.blocks();
        let mut sumf: f32 = 0.0;
        for i in 0..bbs.len() {
            let mut sumi: i32 = 0;
            for j in 0..32 {
                sumi += (abs[i].qs[j] as i32) * (bbs[i].qs[j] as i32);
            }
            sumf += sumi as f32 * abs[i].d.to_f32() * bbs[i].d.to_f32();
        }

        sumf
    }
}

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}

impl BlockQ8_0 {
    pub const BLOCK_ELEMS: usize = 32;

    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d.to_f32();
        for i in 0..32 {
            let q = self.qs[i];
            buf[i] = q as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q80_block() {
        let mut buf: [u8; 68] = [0x1; 68];
        let d = f16::from_f32(3.0).to_bits().to_le_bytes();
        buf[0] = d[0];
        buf[1] = d[1];
        buf[2] = 2;
        buf[3] = 3;
        buf[4] = 4;
        buf[2 + 31] = 7;
        buf[34] = d[0];
        buf[35] = d[1];
        buf[66] = 9;
        buf[67] = 9;

        let blocks = QuantBufQ8_0::from_bytes(&buf[0..34]).blocks;
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(blocks[0].qs, [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 7
        ]);

        let bf = QuantBufQ8_0::from_bytes(&buf);
        assert_eq!(bf.len(), 64);
        assert_eq!(bf.dequantize(0).collect::<Vec<_>>(), vec![
            6.0, 9.0, 12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 27.0, 27.0
        ]);
    }

    #[test]
    fn test_vec_dot_q() {
        let mut buf: [u8; 64] = [0x1; 64];
    }
}
