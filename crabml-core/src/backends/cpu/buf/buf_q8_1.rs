use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct QuantBufQ8_1<'a> {
    pub blocks: Cow<'a, [BlockQ8_1]>,
}

impl<'a> QuantBufQ8_1<'_> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ8_1>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ8_1 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8_1, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }
    pub fn quantize(data: &[f32]) -> Self {
        let bs = unsafe { quantize_f32_q8_1(data) };
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ8_1] {
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
            let mut buf = [0.0; 32];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    #[cfg(not(all(target_feature = "neon")))]
    pub fn vec_dot(&self, a_offset: usize, b: &Self, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks()[b_offset / 32..(b_offset + len) / 32];

        vec_dot_q8_1_q8_1_naive(abs, bbs)
    }

    #[cfg(target_feature = "neon")]
    pub fn vec_dot(&self, a_offset: usize, b: &Self, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks()[b_offset / 32..(b_offset + len) / 32];

        if bbs.len() % 2 == 0 {
            return vec_dot_q8_1_q8_1_neon_unrolled(abs, bbs);
        }
        vec_dot_q8_1_q8_1_neon(abs, bbs)
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct BlockQ8_1 {
    d: f32,       // delta
    s: f32,       // d * sum(qs[i])
    qs: [i8; 32], // quants
}

impl BlockQ8_1 {
    pub fn dequantize(&self, buf: &mut [f32]) {
        for (i, v) in buf.iter_mut().enumerate().take(32) {
            *v = self.qs[i] as f32 * self.d;
        }
    }
}

unsafe fn quantize_f32_q8_1(data: &[f32]) -> Vec<BlockQ8_1> {
    #[cfg(target_arch = "aarch64")]
    {
        quantize_f32_q8_1_neon(data)
    }
    #[cfg(target_arch = "x86_64")]
    #[cfg(target_feature = "avx2")]
    {
        quantize_f32_q8_0_avx2(data)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        quantize_f32_q8_1_fallback(data)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn quantize_f32_q8_1_neon(data: &[f32]) -> Vec<BlockQ8_1> {
    use std::arch::aarch64;

    let mut bs = Vec::with_capacity(data.len() / 32);
    for i in (0..data.len()).step_by(32) {
        let mut vsrc = [aarch64::vdupq_n_f32(0.0); 8];
        let mut vasrc = [aarch64::vdupq_n_f32(0.0); 8];
        let mut vmax = [aarch64::vdupq_n_f32(0.0); 8];

        for j in 0..8 {
            vsrc[j] = aarch64::vld1q_f32(data.as_ptr().add(i + j * 4));
            vasrc[j] = aarch64::vabsq_f32(vsrc[j]);
        }

        for j in 0..4 {
            vmax[2 * j] = aarch64::vmaxq_f32(vasrc[2 * j], vasrc[2 * j + 1]);
        }
        for j in 0..2 {
            vmax[4 * j] = aarch64::vmaxq_f32(vmax[4 * j], vmax[4 * j + 2]);
        }
        for j in 0..1 {
            vmax[8 * j] = aarch64::vmaxq_f32(vmax[8 * j], vmax[8 * j + 4]);
        }
        let max = aarch64::vmaxvq_f32(vmax[0]);

        let d = max / 127.0;
        let mut qs = [0_i8; 32];
        let mut sum_qs = 0_f32;

        for j in 0..8 {
            let v = aarch64::vdivq_f32(vsrc[j], aarch64::vdupq_n_f32(d));
            let vi = aarch64::vcvtq_s32_f32(v);
            let q = aarch64::vgetq_lane_s32(vi, 0) as i8;
            qs[4 * j] = q;
            sum_qs += q as f32;
            let q = aarch64::vgetq_lane_s32(vi, 1) as i8;
            qs[4 * j + 1] = q;
            sum_qs += q as f32;
            let q = aarch64::vgetq_lane_s32(vi, 2) as i8;
            qs[4 * j + 2] = q;
            sum_qs += q as f32;
            let q = aarch64::vgetq_lane_s32(vi, 3) as i8;
            qs[4 * j + 3] = q;
            sum_qs += q as f32;
        }

        bs.push(BlockQ8_1 {
            d,
            s: d * sum_qs,
            qs,
        });
    }

    bs
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
unsafe fn quantize_f32_q8_0_avx2(data: &[f32]) -> Vec<BlockQ8_1> {
    use std::arch::x86_64::*;

    let mut bs = Vec::with_capacity(data.len() / 32);

    for chunk in data.chunks(32) {
        let mut max_abs_values = _mm256_setzero_ps();

        for &value in chunk {
            let val_vec = _mm256_set1_ps(value);
            max_abs_values = _mm256_max_ps(
                max_abs_values,
                _mm256_andnot_ps(_mm256_set1_ps(-0.0), val_vec),
            );
        }

        let max_abs_value = {
            let mut max_vals = [0.0; 8];
            _mm256_storeu_ps(max_vals.as_mut_ptr(), max_abs_values);
            *max_vals
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
        };

        let d = max_abs_value / 127.0;
        let d_vec = _mm256_set1_ps(d);
        let mut qs = [0_i8; 32];
        let mut temp = [0i32; 8]; // Temporary array to hold intermediate results
        let mut s = 0f32; // Initialize s

        for (chunk_index, values) in chunk.chunks(8).enumerate() {
            let values_vec = _mm256_loadu_ps(values.as_ptr());
            let scaled_vec = _mm256_div_ps(values_vec, d_vec);
            let clamped_vec = _mm256_max_ps(
                _mm256_set1_ps(i8::MIN as f32),
                _mm256_min_ps(_mm256_set1_ps(i8::MAX as f32), scaled_vec),
            );
            let converted_vec = _mm256_cvtps_epi32(clamped_vec);
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, converted_vec);

            for (i, &value) in temp.iter().enumerate() {
                qs[chunk_index * 8 + i] = value as i8;
                s += value as f32; // Accumulate the sum of scaled values
            }
        }

        s *= d; // Multiply the sum by d to get the final value of s

        bs.push(BlockQ8_1 { d, s, qs });
    }

    bs
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
fn quantize_f32_q8_1_fallback(data: &[f32]) -> Vec<BlockQ8_1> {
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

        let d = max_abs_value / 127.0; // Compute the scaling factor
        let mut qs = [0_i8; 32]; // Initialize the quantized values array
        let mut s = 0f32; // Initialize the sum of scaled values

        // Quantize the chunk
        for (i, &value) in chunk.iter().enumerate() {
            let scaled_value = value / d; // Scale the value
            // Convert the scaled value to i8, clamping it to the i8 range
            let quantized_value = scaled_value.max(i8::MIN as f32).min(i8::MAX as f32) as i8;
            qs[i] = quantized_value;
            s += quantized_value as f32; // Accumulate the sum of quantized values
        }

        s *= d; // Multiply the sum by d to get the final value of s

        // Store the block with the scaling factor, quantized values, and the sum
        bs.push(BlockQ8_1 { d, s, qs });
    }

    bs
}

pub fn vec_dot_q8_1_q8_1_naive(abs: &[BlockQ8_1], bbs: &[BlockQ8_1]) -> f32 {
    let mut sumf: f32 = 0.0;
    for i in 0..bbs.len() {
        let mut sumi: i32 = 0;
        for j in 0..32 {
            sumi += (abs[i].qs[j] as i32) * (bbs[i].qs[j] as i32);
        }
        sumf += sumi as f32 * abs[i].d * bbs[i].d;
    }

    sumf
}

#[cfg(target_feature = "neon")]
pub fn vec_dot_q8_1_q8_1_neon(abs: &[BlockQ8_1], bbs: &[BlockQ8_1]) -> f32 {
    assert!(abs.len() == bbs.len());

    unsafe {
        use std::arch::aarch64;
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let zerov = aarch64::vdupq_n_s32(0);

        for i in 0..bbs.len() {
            let ab0 = abs.get_unchecked(i);
            let bb0 = bbs.get_unchecked(i);

            let av00 = aarch64::vld1q_s8(ab0.qs.as_ptr());
            let av01 = aarch64::vld1q_s8(ab0.qs.as_ptr().add(16));

            let bv00 = aarch64::vld1q_s8(bb0.qs.as_ptr());
            let bv01 = aarch64::vld1q_s8(bb0.qs.as_ptr().add(16));

            sumv0 = aarch64::vmlaq_n_f32(
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av00, bv00),
                    aarch64::vdotq_s32(zerov, av01, bv01),
                )),
                ab0.d * bb0.d,
            );
        }

        aarch64::vaddvq_f32(sumv0)
    }
}

#[cfg(target_feature = "neon")]
pub fn vec_dot_q8_1_q8_1_neon_unrolled(abs: &[BlockQ8_1], bbs: &[BlockQ8_1]) -> f32 {
    assert!(abs.len() == bbs.len());
    assert!(
        bbs.len() % 2 == 0,
        "bbs.len() must be a multiple of 64, got: {}",
        bbs.len()
    );

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
                ab0.d * bb0.d,
            );

            sumv1 = aarch64::vmlaq_n_f32(
                sumv1,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av10, bv10),
                    aarch64::vdotq_s32(zerov, av11, bv11),
                )),
                ab1.d * bb1.d,
            );
        }

        aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_1_block() {
        const _: () = assert_eq!(
            std::mem::size_of::<BlockQ8_1>(),
            2 * std::mem::size_of::<f32>() + 32,
            "wrong q8_1 block size/padding"
        );

        let mut buf: [u8; 80] = [0x1; 80];

        let d_bytes = f32::to_le_bytes(3.0);
        let s_bytes = f32::to_le_bytes(96.0);
        buf[0..4].copy_from_slice(&d_bytes);
        buf[4..8].copy_from_slice(&s_bytes);

        buf[8] = 2;
        buf[9] = 3;
        buf[10] = 4;
        buf[39] = 7;

        buf[40..44].copy_from_slice(&d_bytes);
        buf[44..48].copy_from_slice(&s_bytes);

        buf[48] = 2;
        buf[49] = 3;
        buf[50] = 4;
        buf[79] = 7;

        let blocks = QuantBufQ8_1::from_bytes(&buf[0..40]).blocks;
        assert_eq!(blocks[0].d, 3.0);
        assert_eq!(blocks[0].s, 96.0);
        assert_eq!(blocks[0].qs, [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 7
        ]);

        let bf = QuantBufQ8_1::from_bytes(&buf);

        assert_eq!(bf.len(), 64);

        assert_eq!(bf.dequantize(0).collect::<Vec<_>>(), vec![
            6.0, 9.0, 12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0, 6.0, 9.0,
            12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0
        ]);
    }
}
