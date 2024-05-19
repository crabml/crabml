use std::borrow::Cow;

use bytemuck::Pod;
use bytemuck::Zeroable;
use half::f16;

use super::QuantBufQ8_0;
use crate::cpu::buf::buf_q8_0::BlockQ8_0;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_0 {
    d: f16,       // delta
    qs: [u8; 16], // quants
}

impl BlockQ4_0 {
    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d.to_f32();
        for i in 0..16 {
            let x0 = (self.qs[i] & 0x0F) as i16 - 8;
            let x1 = (self.qs[i] >> 4) as i16 - 8;

            buf[i] = (x0 as f32) * d;
            buf[i + 16] = (x1 as f32) * d;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ4_0<'a> {
    pub blocks: Cow<'a, [BlockQ4_0]>,
}

impl<'a> QuantBufQ4_0<'_> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ4_0>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ4_0 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4_0, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }
    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q4_0(data);
        Self { blocks: bs.into() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.blocks)
    }

    fn blocks(&self) -> &[BlockQ4_0] {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * 32
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert!(start % 32 == 0);

        let block_start = start / 32;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0f32; 32];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &QuantBufQ8_0, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks[b_offset / 32..(b_offset + len) / 32];

        vec_dot_q4_0_q8_0(abs, bbs)
    }
}

pub fn quantize_f32_q4_0(data: &[f32]) -> Vec<BlockQ4_0> {
    let mut bs = Vec::with_capacity(data.len() / 32);

    for chunk in data.chunks(32) {
        let mut max_abs_value = 0.0;

        // Find the maximum absolute value in the chunk
        for &value in chunk {
            let abs_value = value.abs();
            if abs_value > max_abs_value {
                max_abs_value = abs_value;
            }
        }

        let d = max_abs_value / -8.0; // Compute the scaling factor
        let id = if d != 0f32 { 1. / d } else { 0. };
        let mut qs = [0_u8; 16]; // Initialize the quantized values array

        // Quantize the chunk
        for (i, value) in qs.iter_mut().enumerate() {
            let x0 = chunk[i] * id;
            let x1 = chunk[16 + i] * id;
            let xi0 = u8::min(15, (x0 + 8.5) as u8);
            let xi1 = u8::min(15, (x1 + 8.5) as u8);
            *value = xi0 | (xi1 << 4)
        }
        // Store the block with the scaling factor, quantized values
        bs.push(BlockQ4_0 {
            d: f16::from_f32(d),
            qs,
        });
    }

    bs
}

pub fn vec_dot_q4_0_q8_0(abs: &[BlockQ4_0], bbs: &[BlockQ8_0]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        vec_dot_q4_0_q8_0_neon(abs, bbs)
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        vec_dot_q4_0_q8_0_fallback(abs, bbs)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn vec_dot_q4_0_q8_0_neon(abs: &[BlockQ4_0], bbs: &[BlockQ8_0]) -> f32 {
    use std::arch::aarch64::*;
    let n_blocks = abs.len();
    let n_blocks_rounded = n_blocks - n_blocks % 2;

    let mut sumf = unsafe {
        let mut sumv0 = vdupq_n_f32(0.0);
        let mut sumv1 = vdupq_n_f32(0.0);
        let zerov = vdupq_n_s32(0);

        for i in (0..n_blocks_rounded).step_by(2) {
            let ab0 = abs.get_unchecked(i);
            let ab1 = abs.get_unchecked(i + 1);
            let bb0 = bbs.get_unchecked(i);
            let bb1 = bbs.get_unchecked(i + 1);

            // one q4_0 block can be loaded in a single v1d1q_u8 as a u8x16 register
            let av0_orig = vld1q_u8(ab0.qs.as_ptr());
            let av1_orig = vld1q_u8(ab1.qs.as_ptr());

            // convert each u8x16 to two i16x8
            // in each q4_0 block, the lower 4 bits of the 16 elems makes the first 16 values,
            // and the higher 4 bits makes the second 16 values.
            let mask4bl = vdupq_n_u8(0x0F);
            let eightv = vdupq_n_s8(8);
            let av00 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(av0_orig, mask4bl)), eightv);
            let av01 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(av0_orig, 4)), eightv);
            let av10 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(av1_orig, mask4bl)), eightv);
            let av11 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(av1_orig, 4)), eightv);

            // load bv
            let bv00 = vld1q_s8(bb0.qs.as_ptr());
            let bv01 = vld1q_s8(bb0.qs.as_ptr().add(16));
            let bv10 = vld1q_s8(bb1.qs.as_ptr());
            let bv11 = vld1q_s8(bb1.qs.as_ptr().add(16));

            // same as q8_0
            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(
                    vdotq_s32(zerov, av00, bv00),
                    vdotq_s32(zerov, av01, bv01),
                )),
                f16::to_f32(ab0.d) * f16::to_f32(bb0.d),
            );
            sumv1 = vmlaq_n_f32(
                sumv1,
                vcvtq_f32_s32(vaddq_s32(
                    vdotq_s32(zerov, av10, bv10),
                    vdotq_s32(zerov, av11, bv11),
                )),
                f16::to_f32(ab1.d) * f16::to_f32(bb1.d),
            );
        }
        vaddvq_f32(sumv0) + vaddvq_f32(sumv1)
    };

    // handle the remaining blocks, it seems that only tinyllamas has the case where n_blocks % 2 != 0
    if n_blocks > n_blocks_rounded {
        sumf += vec_dot_q4_0_q8_0_fallback(
            &abs[n_blocks_rounded..n_blocks],
            &bbs[n_blocks_rounded..n_blocks],
        );
    }

    sumf
}

pub fn vec_dot_q4_0_q8_0_fallback(abs: &[BlockQ4_0], bbs: &[BlockQ8_0]) -> f32 {
    let mut sumf: f32 = 0f32;
    for i in 0..bbs.len() {
        let mut sumi: i32 = 0;
        for j in 0..16 {
            let v0 = (abs[i].qs[j] & 0x0F) as i32 - 8;
            let v1 = (abs[i].qs[j] >> 4) as i32 - 8;
            sumi += v0 * bbs[i].qs[j] as i32 + v1 * bbs[i].qs[j + 16] as i32
        }
        sumf += sumi as f32 * f16::to_f32(abs[i].d) * f16::to_f32(bbs[i].d)
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_block() {
        assert_eq!(
            std::mem::size_of::<BlockQ4_0>(),
            std::mem::size_of::<f16>() + 16,
            "wrong q4_0 block size/padding"
        );

        let mut buf: [u8; 36] = [0x1; 36];

        let d = f16::from_f32(3.0).to_bits().to_le_bytes();
        buf[0] = d[0];
        buf[1] = d[1];
        buf[2] = 2;
        buf[3] = 3;
        buf[4] = 4;
        buf[18] = d[0];
        buf[19] = d[1];
        buf[20] = 2;
        buf[21] = 3;
        buf[22] = 4;

        let blocks = QuantBufQ4_0::from_bytes(&buf[0..18]).blocks;
        assert_eq!(blocks[0].qs, [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]);

        let bf = QuantBufQ4_0::from_bytes(&buf);

        assert_eq!(bf.len(), 64);

        assert_eq!(bf.dequantize(0).collect::<Vec<_>>(), vec![
            -18.0, -15.0, -12.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0,
            -21.0, -21.0, -21.0, -21.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0,
            -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -18.0, -15.0, -12.0, -21.0,
            -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0,
            -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0,
            -24.0, -24.0, -24.0, -24.0
        ]);
    }
}
