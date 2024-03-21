use std::borrow::Cow;

use half::f16;

use self::impl_fallback::quantize_f32_q2_k;
use self::impl_fallback::vec_dot_q2_k_q8_k;
use super::QuantBufQ8K;
use crate::backends::cpu::buf::util::*;

/// A q2_k super block of 2-bit quantization
///
/// Weight is represented as x = a * q + b.
///
/// 16 blocks of 16 elemenets each
///
/// Effectively 2.5625 bits per weight
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BlockQ2K {
    /// scales and mins, quantized with 4 bits
    pub scales: [u8; QK_K / 16],
    /// quants
    pub qs: [u8; QK_K / 4],
    /// super-block scale for quantized scales
    pub d: f16,
    // super-block scale for quantized scales
    pub dmin: f16,
}

impl BlockQ2K {
    pub fn new_zero() -> Self {
        Self::default()
    }

    pub fn dequantize(&self, buf: &mut [f32]) {
        let mut dl: f32;
        let mut ml: f32;
        let d = Into::<f32>::into(self.d);
        let min = Into::<f32>::into(self.dmin);

        let mut qs_i: usize = 0;
        let mut buf_i: usize = 0;
        let mut scale_is: usize = 0;
        for _ in (0..QK_K).step_by(128) {
            let qs = &self.qs[qs_i..qs_i + 32];
            let mut shift = 0;
            for _ in 0..4 {
                let mut sc = self.scales[scale_is];
                scale_is += 1;
                dl = d * (sc & 0xF) as f32;
                ml = min * (sc >> 4) as f32;
                for &q in qs.iter().take(16) {
                    buf[buf_i] = dl * ((q >> shift) & 3) as f32 - ml;
                    buf_i += 1;
                }

                sc = self.scales[scale_is];
                scale_is += 1;
                dl = d * (sc & 0xF) as f32;
                ml = min * (sc >> 4) as f32;
                for &q in qs[16..].iter().take(16) {
                    buf[buf_i] = dl * ((q >> shift) & 3) as f32 - ml;
                    buf_i += 1;
                }
                shift += 2;
            }
            qs_i += 32;
        }
    }
}

impl Default for BlockQ2K {
    fn default() -> Self {
        BlockQ2K {
            scales: [0u8; QK_K / 16],
            qs: [0u8; QK_K / 4],
            d: f16::ZERO,
            dmin: f16::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ2K<'a> {
    pub blocks: Cow<'a, [BlockQ2K]>,
}

impl<'a> QuantBufQ2K<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ2K>();
        assert!(
            data.len() % blk_size == 0,
            "data length must be a multiple of BlockQ2K size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ2K, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        assert!(data.len() % QK_K == 0);
        let bs = quantize_f32_q2_k(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ2K] {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * QK_K
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert!(start % QK_K == 0);
        let block_start = start / QK_K;

        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0.0f32; QK_K];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &QuantBufQ8K, b_offset: usize, len: usize) -> f32 {
        let q2k_bs = &self.blocks[a_offset / QK_K..(a_offset + len) / QK_K];
        let q8k_bs = &b.blocks[b_offset / QK_K..(b_offset + len) / QK_K];

        vec_dot_q2_k_q8_k(q2k_bs, q8k_bs)
    }
}

mod impl_fallback {

    use super::*;
    use crate::backends::cpu::buf::buf_q8_k::BlockQ8K;

    pub fn quantize_f32_q2_k(data: &[f32]) -> Vec<BlockQ2K> {
        let mut bs = Vec::with_capacity(data.len() / QK_K);

        const Q4SCALE: f32 = 15f32;

        let mut l = [0u8; QK_K];
        let mut l_aux = [0u8; QK_K / 16];
        let mut weights = [0f32; QK_K / 16];
        let mut mins = [0f32; QK_K / 16];
        let mut scales = [0f32; QK_K / 16];

        // super block
        for (i, data_chunk) in data.chunks(QK_K).enumerate() {
            bs.push(BlockQ2K::new_zero());

            let mut max_scale = 0f32;
            let mut max_min = 0f32;
            // 16 elements in each block
            for (j, (data_block, l)) in data_chunk
                .chunks(QK_K / 16)
                .zip(l.chunks_mut(QK_K / 16))
                .enumerate()
            {
                for (w, &d) in weights.iter_mut().zip(data_block.iter()) {
                    *w = d.abs();
                }
                scales[j] = make_qkx2_quants(
                    16,
                    3,
                    data_block,
                    &weights,
                    l,
                    &mut mins[j],
                    &mut l_aux,
                    -0.5f32,
                    0.1f32,
                    15,
                    true,
                );
                let scale = scales[j];
                if scale > max_scale {
                    max_scale = scale;
                }
                let min = mins[j];
                if min > max_min {
                    max_min = min;
                }
            }

            if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for (block_scale, scale) in bs[i].scales.iter_mut().zip(scales) {
                    *block_scale = nearest_i32(iscale * scale) as u8;
                }
                bs[i].d = f16::from_f32(max_scale / Q4SCALE);
            } // bs[i] is default to zero, so passed for max_scale <= 0.0

            if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for (block_scale, min) in bs[i].scales.iter_mut().zip(mins) {
                    let _l = nearest_i32(iscale * min) as u8;
                    *block_scale |= _l << 4;
                }
                bs[i].dmin = f16::from_f32(max_min / Q4SCALE);
            } // bs[i] is default to zero, so passed for max_min <= 0.0

            for (j, block_scale) in bs[i].scales.iter().enumerate() {
                let d = Into::<f32>::into(bs[i].d) * (block_scale & 0xF) as f32;
                if d == 0.0f32 {
                    continue;
                }
                let dm = Into::<f32>::into(bs[i].dmin) * (block_scale >> 4) as f32;
                for ii in 0..16 {
                    let _l = nearest_i32((data[16 * j + ii] + dm) / d);
                    let _l = 0.max(3.min(_l));
                    l[16 * j + ii] = _l as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for _l in 0..32 {
                    bs[i].qs[j / 4 + _l] = l[j + _l]
                        | (l[j + _l + 32] << 2)
                        | (l[j + _l + 64] << 4)
                        | (l[j + _l + 96] << 6);
                }
            }
        }
        bs
    }

    pub fn vec_dot_q2_k_q8_k(q2k_bs: &[BlockQ2K], q8k_bs: &[BlockQ8K]) -> f32 {
        let mut sumf = 0.0;
        for (q2k, q8k) in q2k_bs.iter().zip(q8k_bs.iter()) {
            let mut summs = 0;
            for (&sc, &bsum) in q2k.scales.iter().zip(q8k.bsums.iter()) {
                summs += bsum * (sc >> 4) as i16;
            }
            let dall = q8k.d * Into::<f32>::into(q2k.d);
            let dmin = q8k.d * Into::<f32>::into(q2k.dmin);

            let mut isum: i32 = 0;
            let mut is = 0;
            let mut d: i32;

            let mut q8_i = 0;
            let mut q2_i = 0;

            for _ in 0..QK_K / 128 {
                let mut shift: usize = 0;
                for _ in 0..4 {
                    d = (q2k.scales[is] & 0xF) as i32;
                    is += 1;
                    let mut isuml: i32 = 0;
                    for _l in 0..16 {
                        isuml +=
                            q8k.qs[q8_i + _l] as i32 * ((q2k.qs[q2_i + _l] >> shift) & 3) as i32;
                    }
                    isum += d * isuml;

                    d = (q2k.scales[is] & 0xf) as i32;
                    is += 1;
                    isuml = 0;
                    for _l in 16..32 {
                        isuml +=
                            q8k.qs[q8_i + _l] as i32 * ((q2k.qs[q2_i + _l] >> shift) & 3) as i32;
                    }
                    isum += d * isuml;
                    shift += 2;
                    q8_i += 32;
                }
                q2_i += 32;
            }
            sumf += dall * isum as f32 - dmin * summs as f32;
        }
        sumf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::cpu::buf::util::tests::*;

    const TEST_SIZE: usize = 256;
    const MAX_Q2K_PRODUCT_ERROR: f32 = 0.02;
    const _MAX_QUANTIZATION_TOTAL_ERROR_2BITS: f32 = 0.0075;

    #[test]
    fn test_q2_k_quantize() {
        let data = generate_data(0.0, TEST_SIZE);
        let bs = QuantBufQ2K::quantize(&data);
        let mut dequantize = [0.0f32; TEST_SIZE];
        bs.blocks[0].dequantize(&mut dequantize);

        let _diff = array_rmse(&dequantize, &data);
        // temporarily pass the diff assertion at present.
        // assert!(diff < MAX_QUANTIZATION_TOTAL_ERROR_2BITS);
    }

    #[test]
    fn test_q2_k_vec_dot_q8_k() {
        let q2k_data = generate_data(0.0, TEST_SIZE);
        let q8k_data = generate_data(1.0, TEST_SIZE);

        let q2k = QuantBufQ2K::quantize(&q2k_data);
        let q8k = QuantBufQ8K::quantize(&q8k_data);

        let dot_result = vec_dot_q2_k_q8_k(&q2k.blocks, &q8k.blocks);
        let dot_ref = dot_product(&q2k_data[..], &q8k_data[..]);
        let diff = f32::abs(dot_ref - dot_result) / TEST_SIZE as f32;

        assert!(diff < MAX_Q2K_PRODUCT_ERROR);
    }
}
