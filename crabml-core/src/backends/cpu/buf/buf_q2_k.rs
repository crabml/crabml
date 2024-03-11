use std::borrow::Cow;

use half::f16;

use self::impl_fallback::quantize_f32_q2_k;
use crate::backends::cpu::buf::qkk::*;

/// A q2_k super block of 2-bit quantization
///
/// Weight is represented as x = a * q + b.
///
/// 16 blocks of 16 elemenets each
///
/// Effectively 2.5625 bits per weight
#[repr(C)]
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct BlockQ2_K {
    /// scales and mins, quantized with 4 bits
    pub scales: [u8; QK_K / 16],
    /// quants
    pub qs: [u8; QK_K / 4],
    /// super-block scale for quantized scales
    pub d: f16,
    // super-block scale for quantized scales
    pub dmin: f16,
}

impl BlockQ2_K {
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
                for q in qs.iter().take(16) {
                    buf[buf_i] = dl * ((*q >> shift) & 3) as f32 - ml;
                    buf_i += 1;
                }

                sc = self.scales[scale_is];
                scale_is += 1;
                dl = d * (sc & 0xF) as f32;
                ml = min * (sc >> 4) as f32;
                for q in qs[16..].iter().take(16) {
                    buf[buf_i] = dl * ((q >> shift) & 3) as f32 - ml;
                    buf_i += 1;
                }
                shift += 2;
            }
            qs_i += 32;
        }
    }
}

impl Default for BlockQ2_K {
    fn default() -> Self {
        BlockQ2_K {
            scales: [0u8; QK_K / 16],
            qs: [0u8; QK_K / 4],
            d: f16::ZERO,
            dmin: f16::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct QuantBufQ2_K<'a> {
    pub blocks: Cow<'a, [BlockQ2_K]>,
}

impl<'a> QuantBufQ2_K<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ2_K>();
        assert!(
            data.len() % blk_size == 0,
            "data length must be a multiple of BlockQ2_K size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ2_K, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q2_k(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ2_K] {
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
            let mut buf = [0.0f32; 256];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &Self, b_offset: usize, len: usize) -> f32 {
        todo!();
    }
}

mod impl_fallback {
    use half::f16;

    use super::*;

    pub fn quantize_f32_q2_k(data: &[f32]) -> Vec<BlockQ2_K> {
        assert!(data.len() % QK_K == 0);

        const Q4SCALE: f32 = 15f32;

        let mut _L = [0u8; QK_K];
        let mut mins = [0f32; QK_K / 16];
        let mut scales = [0f32; QK_K / 16];

        let nb = data.len() / QK_K; // super blocks vec length
        let mut bs = Vec::with_capacity(nb);

        // super block
        for (i, data_chunk) in data.chunks(QK_K).enumerate() {
            bs.push(BlockQ2_K::new_zero());

            let mut max_scale = 0f32;
            let mut max_min = 0f32;
            // 16 elements in each block
            for (j, (data_block, _L)) in data_chunk.chunks(16).zip(_L.chunks_mut(16)).enumerate() {
                scales[j] = make_qkx1_quants(16, 3, data_block, _L, &mut mins[j], 5);
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
                    let l = nearest_i32(iscale * min) as u8;
                    *block_scale |= l << 4;
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
                    let l = nearest_i32((data[16 * j + ii] + dm) / d);
                    let l = 0.max(3.min(l));
                    _L[16 * j + ii] = l as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l in 0..32 {
                    bs[i].qs[j / 4 + l] = _L[j + l]
                        | (_L[j + l + 32] << 2)
                        | (_L[j + l + 64] << 4)
                        | (_L[j + l + 96] << 6);
                }
            }
        }
        bs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::cpu::buf::qkk::tests::*;

    const MAX_QUANTIZATION_TOTAL_ERROR_2BITS: f32 = 0.0075;
    const TEST_SIZE: usize = 256;

    #[test]
    fn test_q2_k_quantize() {
        let data = generate_data(0.0, TEST_SIZE);
        let bs = QuantBufQ2_K::quantize(&data);
        let mut dequantize = [0.0f32; TEST_SIZE];
        bs.blocks[0].dequantize(&mut dequantize);

        let diff = array_rmse(&dequantize, &data);
        // println!("{diff}");
        // println!("dequantize: {:?}", dequantize);
        // println!("data: {:?}", data);
        assert!(diff < MAX_QUANTIZATION_TOTAL_ERROR_2BITS);
    }
}
