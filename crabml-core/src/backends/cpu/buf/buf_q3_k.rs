use std::borrow::Cow;

use half::f16;

use self::impl_fallback::quantize_f32_q3_k;
use self::impl_fallback::vec_dot_q3_k_q8_k;
use crate::backends::cpu::buf::buf_q8_k::BlockQ8K;
use crate::backends::cpu::buf::buf_q8_k::QuantBufQ8K;
use crate::backends::cpu::buf::util::*;

/// A q3_k super block of 3-bit quantization
///
/// weight is represented as x = a * q
///
/// 16 blocks of 16 elemenets each
///
/// Effectively 3.4375 bits per weight
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BlockQ3K {
    /// quants - high bit
    pub hmask: [u8; QK_K / 8],
    /// quants - low 2 bits
    pub qs: [u8; QK_K / 4],
    /// scales, quantized with 6 bits
    pub scales: [u8; 3 * QK_K / 64],
    /// super-block scale
    pub d: f16,
}

impl BlockQ3K {
    pub fn new_zero() -> Self {
        Self::default()
    }

    pub fn dequantize(&self, buf: &mut [f32]) {
        // todo
    }
}

impl Default for BlockQ3K {
    fn default() -> Self {
        Self {
            hmask: [0; QK_K / 8],
            qs: [0; QK_K / 4],
            scales: [0; 3 * QK_K / 64],
            d: f16::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ3K<'a> {
    pub blocks: Cow<'a, [BlockQ3K]>,
}

impl<'a> QuantBufQ3K<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ3K>();
        assert!(
            data.len() % blk_size == 0,
            "data length must be a multiple of BlockQ3K size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ3K, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        assert!(data.len() % QK_K == 0);
        let bs = quantize_f32_q3_k(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ3K] {
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
        let q3k_bs = &self.blocks[a_offset / QK_K..(a_offset + len) / QK_K];
        let q8k_bs = &b.blocks[b_offset / QK_K..(b_offset + len) / QK_K];

        vec_dot_q3_k_q8_k(q3k_bs, q8k_bs)
    }
}

mod impl_fallback {
    use super::*;

    pub fn quantize_f32_q3_k(data: &[f32]) -> Vec<BlockQ3K> {
        let mut bs = Vec::with_capacity(data.len() / QK_K);

        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];

        for (i, data_chunk) in data.chunks(QK_K).enumerate() {
            bs.push(BlockQ3K::new_zero());

            let mut max_scale = 0f32;
            let mut amax = 0f32;
            for (j, (data_block, l)) in data_chunk.chunks(16).zip(l.chunks_mut(16)).enumerate() {
                scales[j] = make_q3_quants(16, 4, data_block, l, true);
                let scale = scales[j].abs();
                if scale > amax {
                    amax = scale;
                    max_scale = scales[j];
                }
            }

            if max_scale != 0f32 {
                let iscale = -32f32 / max_scale;
                for (j, &scale) in scales.iter().enumerate() {
                    let mut _l = nearest_i32(iscale * scale) as i8;
                    _l = _l.min(31).max(-32) + 32;
                    if j < 8 {
                        bs[i].scales[j] = _l as u8 & 0xf;
                    } else {
                        bs[i].scales[j - 8] |= (_l as u8 & 0xf) << 4;
                    }
                    _l >>= 4;
                    bs[i].scales[j % 4 + 8] |= (_l as u8) << (2 * (j / 4));
                }
                bs[i].d = f16::from_f32(1f32 / iscale);
            } // bs[i] is default to zero, so passed for max_scale == 0.0

            let mut sc = 0i8;
            for (j, (data_block, l)) in data_chunk.chunks(16).zip(l.chunks_mut(16)).enumerate() {
                sc = if j < 8 {
                    (bs[i].scales[j] & 0xf) as i8
                } else {
                    (bs[i].scales[j - 8] >> 4) as i8
                };
                sc = (sc | (((bs[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4) as i8) - 32;
                let _d = Into::<f32>::into(bs[i].d) * sc as f32;
                if _d == 0f32 {
                    continue;
                }
                for (&d, l) in data_block.iter().zip(l.iter_mut()) {
                    let mut _l = nearest_i32(d / _d);
                    _l = _l.min(3).max(-4);
                    *l = (_l + 4) as i8;
                }
            }

            let mut m = 0;
            let mut hm = 1u8;
            for l in l.iter_mut() {
                if *l > 3 {
                    bs[i].hmask[m] |= hm;
                    *l -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }
            for j in (0..QK_K).step_by(128) {
                for _l in 0..32 {
                    bs[i].qs[j / 4 + _l] = (l[j + _l]
                        | (l[j + _l + 32] << 2)
                        | (l[j + _l + 64] << 4)
                        | (l[j + _l + 96] << 6)) as u8;
                }
            }
        }

        bs
    }

    pub fn vec_dot_q3_k_q8_k(q2k_bs: &[BlockQ3K], q8k_bs: &[BlockQ8K]) -> f32 {
        // TODO
        0.0
    }
}
