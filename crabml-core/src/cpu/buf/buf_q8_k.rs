use std::borrow::Cow;

use bytemuck::Pod;
use bytemuck::Zeroable;

#[repr(C)]
#[derive(Debug, Clone, Zeroable, Pod, Copy)]
pub struct BlockQ8K {
    pub d: f32,        // delta
    pub qs: [i8; 256], // quants
    pub bsums: [i16; 16],
}

impl BlockQ8K {
    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d;
        for (i, &q) in self.qs.iter().enumerate() {
            buf[i] = d * q as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ8K<'a> {
    pub blocks: Cow<'a, [BlockQ8K]>,
}

impl<'a> QuantBufQ8K<'_> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ8K>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ8_K size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8K, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q8_k(data);
        Self { blocks: bs.into() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.blocks)
    }

    fn blocks(&self) -> &[BlockQ8K] {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * 256
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert!(start % 256 == 0);

        let block_start = start / 256;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0f32; 256];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &QuantBufQ8K, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 256..(a_offset + len) / 256];
        let bbs = &b.blocks[b_offset / 256..(b_offset + len) / 256];

        vec_dot_q8_k_q8_k(abs, bbs)
    }
}

pub fn quantize_f32_q8_k(data: &[f32]) -> Vec<BlockQ8K> {
    let mut bs = Vec::with_capacity(data.len() / 32);

    for chunk in data.chunks(256) {
        let mut max_abs_value = 0.0;
        let mut max_value = 0.0;

        // Find the maximum absolute value in the chunk
        for &value in chunk {
            let abs_value = value.abs();
            if abs_value > max_abs_value {
                max_abs_value = abs_value;
                max_value = value;
            }
        }

        let scale = -128f32 / max_value;
        let mut d = 1.0 / scale; // Compute the scaling factor
        let mut qs = [0_i8; 256]; // Initialize the quantized values array
        let mut bsums = [0_i16; 16];

        // Quantize the chunk

        if max_abs_value == 0f32 {
            d = 0f32;
            qs.fill(0)
        } else {
            for (i, q) in qs.iter_mut().enumerate() {
                // ggml uses nearest_int with bit magic here, maybe we want the same
                // but we would have to test and benchmark it.
                let v = (scale * chunk[i]).round();
                *q = v.min(127.) as i8
            }
            for i in 0..16 {
                let mut sum = 0i32;
                for j in 0..16 {
                    sum += qs[i * 16 + j] as i32
                }
                bsums[i] = sum as i16
            }
        }

        // Store the block with the scaling factor, quantized values
        bs.push(BlockQ8K { d, qs, bsums });
    }

    bs
}

pub fn vec_dot_q8_k_q8_k(abs: &[BlockQ8K], bbs: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        vec_dot_q8_k_q8_k_neon(abs, bbs)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        vec_dot_q8_k_q8_k_avx2(abs, bbs)
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    vec_dot_q8_k_q8_k_fallback(abs, bbs)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn vec_dot_q8_k_q8_k_neon(abs: &[BlockQ8K], bbs: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    use crate::cpu::archutil::aarch64::vdotq_s32;

    debug_assert_eq!(abs.len(), bbs.len());

    let mut sumf = 0f32;
    for (xs, ys) in abs.iter().zip(bbs.iter()) {
        unsafe {
            let mut sum_i = vdupq_n_s32(0);
            let scale = xs.d * ys.d;
            let xs = xs.qs.as_ptr();
            let ys = ys.qs.as_ptr();
            for i in (0..256).step_by(16) {
                let xs = vld1q_s8(xs.add(i));
                let ys = vld1q_s8(ys.add(i));
                let xy = vdotq_s32(xs, ys);
                sum_i = vaddq_s32(sum_i, xy)
            }
            sumf += vaddvq_s32(sum_i) as f32 * scale
        }
    }
    sumf
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn vec_dot_q8_k_q8_k_avx2(abs: &[BlockQ8K], bbs: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    use crate::cpu::archutil::x86_64::*;

    debug_assert_eq!(abs.len(), bbs.len());

    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (xs, ys) in abs.iter().zip(bbs.iter()) {
            let mut sumi = _mm256_setzero_si256();
            let x_qs = xs.qs.as_ptr();
            let y_qs = ys.qs.as_ptr();
            for j in (0..256).step_by(32) {
                let xs = _mm256_loadu_si256(x_qs.add(j) as *const __m256i);
                let ys = _mm256_loadu_si256(y_qs.add(j) as *const __m256i);

                let xs0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xs, 0));
                let ys0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ys, 0));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(xs0, ys0));

                let xs1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xs, 1));
                let ys1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ys, 1));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(xs1, ys1));
            }
            let d = _mm256_set1_ps(xs.d * ys.d);
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}

pub fn vec_dot_q8_k_q8_k_fallback(abs: &[BlockQ8K], bbs: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (abs, bbs) in abs.iter().zip(bbs.iter()) {
        let sum_i = abs
            .qs
            .iter()
            .zip(bbs.qs.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum::<i32>();
        sumf += sum_i as f32 * abs.d * bbs.d
    }

    sumf
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;

    #[test]
    fn test_q8_k_block() {
        assert_eq!(
            std::mem::size_of::<BlockQ8K>(),
            std::mem::size_of::<f32>() + 256 + 16 * 2,
            "wrong q8_k block size/padding"
        );

        let mut buf: [u8; 292] = [0x1; 292];
        let delta = f16::from_f32(3.0).to_bits().to_le_bytes();
        let min_val = f16::from_f32(1.0).to_bits().to_le_bytes();

        buf[0] = delta[0];
        buf[1] = delta[1];
        buf[2] = min_val[0];
        buf[3] = min_val[1];
        buf[4] = 2;
        buf[5] = 3;
        buf[6] = 4;
        buf[4 + 15] = 7;
        buf[283] = 10;

        let blocks = QuantBufQ8K::from_bytes(&buf[0..292]).blocks;
        assert_eq!(blocks[0].d, 0.007828236);
        assert_eq!(blocks[0].qs[0..16], [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7
        ]);
        assert_eq!(blocks[0].bsums[0..16], [
            257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 2561, 257, 257, 257, 257
        ]);
    }

    #[test]
    fn test_q8_k_quantize() {
        let data = vec![
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        let bs = QuantBufQ8K::quantize(&data);

        assert_eq!(bs.blocks.len(), 1);
        assert_eq!(bs.blocks[0].d, 0.0625);

        let mut dequantize = [0.0f32; 256];
        bs.blocks[0].dequantize(&mut dequantize);
        assert_eq!(dequantize, *data);
    }
}
