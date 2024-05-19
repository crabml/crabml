use std::borrow::Cow;
use std::simd::num::SimdFloat;

use bytemuck::Pod;
use bytemuck::Zeroable;
use half::f16;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct BlockQ8_0 {
    pub d: f16,       // delta
    pub qs: [i8; 32], // quants
}

impl BlockQ8_0 {
    pub const BLOCK_ELEMS: usize = 32;

    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d.to_f32();
        for (i, v) in buf.iter_mut().enumerate().take(32) {
            *v = self.qs[i] as f32 * d;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ8_0<'a> {
    pub blocks: Cow<'a, [BlockQ8_0]>,
}

impl<'a> QuantBufQ8_0<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ8_0>();
        assert_eq!(
            data.len() % blk_size,
            0,
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
        let bs = quantize_f32_q8_0(data);
        Self { blocks: bs.into() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.blocks)
    }

    fn blocks(&self) -> &[BlockQ8_0] {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * 32
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert_eq!(start % 32, 0);

        let block_start = start / 32;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0.0; 32];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &Self, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks()[b_offset / 32..(b_offset + len) / 32];

        vec_dot_q8_0_q8_0(abs, bbs)
    }
}

pub fn quantize_f32_q8_0(data: &[f32]) -> Vec<BlockQ8_0> {
    use std::simd::f32x4;
    assert!(data.len() % 32 == 0);

    let mut bs = Vec::with_capacity(data.len() / 32);

    for i in (0..data.len()).step_by(32) {
        let mut vsrc = [f32x4::splat(0.0); 8];
        let mut vasrc = [f32x4::splat(0.0); 8];
        let mut vmax = [f32x4::splat(0.0); 8];

        for j in 0..8 {
            vsrc[j] = f32x4::from_slice(&data[i + j * 4..]);
            vasrc[j] = vsrc[j].abs();
        }

        for j in 0..4 {
            vmax[2 * j] = vasrc[2 * j].simd_max(vasrc[2 * j + 1]);
        }
        for j in 0..2 {
            vmax[4 * j] = vmax[4 * j].simd_max(vmax[4 * j + 2]);
        }
        for j in 0..1 {
            vmax[8 * j] = vmax[8 * j].simd_max(vmax[8 * j + 4]);
        }
        let max = vmax[0].reduce_max();

        let d = max / 127.0;
        let vd = f32x4::splat(d);
        let mut qs = [0_i8; 32];

        for j in 0..8 {
            let v = vsrc[j] / vd;
            let vi: std::simd::i32x4 = v.cast();
            qs[4 * j] = vi[0] as i8;
            qs[4 * j + 1] = vi[1] as i8;
            qs[4 * j + 2] = vi[2] as i8;
            qs[4 * j + 3] = vi[3] as i8;
        }

        bs.push(BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        });
    }

    bs
}

fn vec_dot_q8_0_q8_0(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        vec_dot_q8_0_q8_0_neon(abs, bbs)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        vec_dot_q8_0_q8_0_avx2(abs, bbs)
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    vec_dot_q8_0_q8_0_fallback(abs, bbs)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn vec_dot_q8_0_q8_0_neon(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    use std::arch::aarch64;

    let blocks_rounded = bbs.len() - bbs.len() % 2;
    let mut result = 0.0;

    unsafe {
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        let zerov = aarch64::vdupq_n_s32(0);

        for i in (0..blocks_rounded).step_by(2) {
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
        result += aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1);

        let mut sumv = aarch64::vdupq_n_f32(0.0);
        for i in blocks_rounded..bbs.len() {
            let ab = abs.get_unchecked(i);
            let bb = bbs.get_unchecked(i);

            let av0 = aarch64::vld1q_s8(ab.qs.as_ptr());
            let av1 = aarch64::vld1q_s8(ab.qs.as_ptr().add(16));
            let bv0 = aarch64::vld1q_s8(bb.qs.as_ptr());
            let bv1 = aarch64::vld1q_s8(bb.qs.as_ptr().add(16));

            sumv = aarch64::vmlaq_n_f32(
                sumv,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av0, bv0),
                    aarch64::vdotq_s32(zerov, av1, bv1),
                )),
                f16::to_f32(ab.d) * f16::to_f32(bb.d),
            );
            result += aarch64::vaddvq_f32(sumv);
        }
    };

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn vec_dot_q8_0_q8_0_avx2(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;

    use crate::backends::cpu::archutil::x86_64::*;
    debug_assert_eq!(abs.len(), bbs.len());

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for [(abs0, bbs0), (abs1, bbs1)] in abs.iter().zip(bbs).array_chunks::<2>() {
            let d0 = _mm256_set1_ps(abs0.d.to_f32() * bbs0.d.to_f32());
            let d1 = _mm256_set1_ps(abs1.d.to_f32() * bbs1.d.to_f32());

            let qa0 = _mm256_loadu_si256(abs0.qs.as_ptr() as *const __m256i);
            let qb0 = _mm256_loadu_si256(bbs0.qs.as_ptr() as *const __m256i);

            let qa1 = _mm256_loadu_si256(abs1.qs.as_ptr() as *const __m256i);
            let qb1 = _mm256_loadu_si256(bbs1.qs.as_ptr() as *const __m256i);

            let q0 = mul_sum_i8_pairs_float(qa0, qb0);
            let q1 = mul_sum_i8_pairs_float(qa1, qb1);

            acc0 = _mm256_fmadd_ps(d0, q0, acc0);
            acc1 = _mm256_fmadd_ps(d1, q1, acc1);
        }

        if abs.len() % 2 == 1 {
            let a = abs.last().unwrap_unchecked();
            let b = abs.last().unwrap_unchecked();

            let d = _mm256_set1_ps(a.d.to_f32() * b.d.to_f32());

            let qa = _mm256_loadu_si256(a.qs.as_ptr() as *const __m256i);
            let qb = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);

            let q = mul_sum_i8_pairs_float(qa, qb);

            acc0 = _mm256_fmadd_ps(d, q, acc0);
        }

        hsum_float_8(_mm256_add_ps(acc0, acc1))
    }
}

#[allow(unused)]
pub fn vec_dot_q8_0_q8_0_fallback(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
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
    fn test_vec_dot_q8_0_q8_0() {
        let tests = vec![
            (
                "2*2",
                vec![
                    BlockQ8_0 {
                        qs: [
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        ],
                        d: f16::from_f32(0.4),
                    },
                    BlockQ8_0 {
                        qs: [
                            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
                            -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30,
                            -31, -32,
                        ],
                        d: f16::from_f32(0.7),
                    },
                ],
                vec![
                    BlockQ8_0 {
                        qs: [
                            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
                            14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                        ],
                        d: f16::from_f32(1.3),
                    },
                    BlockQ8_0 {
                        qs: [
                            -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19,
                            -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4,
                            -3, -2, -1,
                        ],
                        d: f16::from_f32(1.4),
                    },
                ],
                8978.046,
            ),
            (
                "1*1",
                vec![BlockQ8_0 {
                    qs: [
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    ],
                    d: f16::from_f32(0.4),
                }],
                vec![BlockQ8_0 {
                    qs: [
                        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
                        13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                    ],
                    d: f16::from_f32(1.3),
                }],
                3110.453,
            ),
        ];

        for (name, abs, bbs, expect) in tests {
            let result = vec_dot_q8_0_q8_0(&abs, &bbs);
            assert_eq!(result, expect, "test: {}", name);
        }
    }
}
