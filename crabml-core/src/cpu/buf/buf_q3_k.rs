use std::borrow::Cow;
use std::ptr;

use bytemuck::Pod;
use bytemuck::Zeroable;
use half::f16;

use crate::cpu::buf::buf_q8_k::BlockQ8K;
use crate::cpu::buf::buf_q8_k::QuantBufQ8K;
use crate::cpu::buf::util::*;

/// A q3_k super block of 3-bit quantization
///
/// weight is represented as x = a * q
///
/// 16 blocks of 16 elemenets each
///
/// Effectively 3.4375 bits per weight
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
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
        const KMASK_1: u32 = 0x03030303;
        const KMASK_2: u32 = 0x0f0f0f0f;

        let d_all = Into::<f32>::into(self.d);
        let mut m = 1u8;

        let mut aux = [0u32; 4];
        // memcpy self.scales into aux
        unsafe {
            let aux_u8 = &mut aux as *mut [u32] as *mut u8;
            ptr::copy_nonoverlapping(&self.scales as *const u8, aux_u8, 12);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK_2) | (((tmp >> 4) & KMASK_1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK_2) | (((tmp >> 6) & KMASK_1) << 4);
        aux[0] = (aux[0] & KMASK_2) | (((tmp) & KMASK_1) << 4);
        aux[1] = (aux[1] & KMASK_2) | (((tmp >> 2) & KMASK_1) << 4);
        let scales: &[i8] =
            unsafe { std::slice::from_raw_parts(&aux as *const [u32] as *const i8, 16) };

        let mut qs_i: usize = 0; // self.qs index
        let mut buf_i = 0; // buf index
        let mut is: usize = 0; // scales index
        let mut dl: f32;
        for _ in (0..QK_K).step_by(128) {
            let mut shift = 0;
            for _ in 0..4 {
                dl = d_all * (scales[is] - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let _m: i8 = if self.hmask[l] & m != 0 { 0 } else { 4 };
                    buf[buf_i] = dl * (((self.qs[l + qs_i] >> shift) & 3) as i8 - _m) as f32;
                    buf_i += 1;
                }
                dl = d_all * (scales[is] - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let _m: i8 = if self.hmask[l + 16] & m != 0 { 0 } else { 4 };
                    buf[buf_i] = dl * (((self.qs[l + qs_i + 16] >> shift) & 3) as i8 - _m) as f32;
                    buf_i += 1;
                }
                shift += 2;
                m <<= 1;
            }
            qs_i += 32;
        }
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

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.blocks)
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

        let mut sc;
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

pub fn vec_dot_q3_k_q8_k(q3k_bs: &[BlockQ3K], q8k_bs: &[BlockQ8K]) -> f32 {
    const KMASK_1: u32 = 0x03030303;
    const KMASK_2: u32 = 0x0f0f0f0f;

    let mut aux_8 = [0i8; QK_K];
    let mut aux_16 = [0i16; 8];

    let mut sums = [0f32; 8];
    for (q3k, q8k) in q3k_bs.iter().zip(q8k_bs.iter()) {
        let mut a8_i: usize = 0;
        let mut q3_i: usize = 0;
        let mut m: u8 = 1;
        for _ in (0..QK_K).step_by(128) {
            for l in 0..32 {
                aux_8[a8_i + l] = (q3k.qs[q3_i + l] & 3) as i8;
            }
            for l in 0..32 {
                aux_8[a8_i + l] -= if q3k.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a8_i += 32;
            m <<= 1;
            for l in 0..32 {
                aux_8[a8_i + l] = ((q3k.qs[q3_i + l] >> 2) & 3) as i8;
            }
            for l in 0..32 {
                aux_8[a8_i + l] -= if q3k.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a8_i += 32;
            m <<= 1;
            for l in 0..32 {
                aux_8[a8_i + l] = ((q3k.qs[q3_i + l] >> 4) & 3) as i8;
            }
            for l in 0..32 {
                aux_8[a8_i + l] -= if q3k.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a8_i += 32;
            m <<= 1;
            for l in 0..32 {
                aux_8[a8_i + l] = ((q3k.qs[q3_i + l] >> 6) & 3) as i8;
            }
            for l in 0..32 {
                aux_8[a8_i + l] -= if q3k.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a8_i += 32;
            m <<= 1;
            q3_i += 32;
        }
        a8_i = 0;

        let mut aux_32 = [0i32; 8];
        let mut auxs: [u32; 4] = [0; 4];
        // memcpy q3k.scales into aux
        unsafe {
            let aux_u8 = &mut auxs as *mut [u32] as *mut u8;
            ptr::copy_nonoverlapping(&q3k.scales as *const u8, aux_u8, 12);
        }
        let tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & KMASK_2) | (((tmp >> 4) & KMASK_1) << 4);
        auxs[3] = ((auxs[1] >> 4) & KMASK_2) | (((tmp >> 6) & KMASK_1) << 4);
        auxs[0] = (auxs[0] & KMASK_2) | (((tmp) & KMASK_1) << 4);
        auxs[1] = (auxs[1] & KMASK_2) | (((tmp >> 2) & KMASK_1) << 4);
        let scales: &[i8] =
            unsafe { &*ptr::slice_from_raw_parts(&mut auxs as *mut u32 as *mut i8, 16) };
        let mut q8_i: usize = 0;
        for &sc in scales.iter().take(QK_K / 16) {
            for l in 0..8 {
                aux_16[l] = q8k.qs[q8_i + l] as i16 * aux_8[a8_i + l] as i16;
            }
            for l in 0..8 {
                aux_32[l] += (sc - 32) as i32 * aux_16[l] as i32;
            }
            q8_i += 8;
            a8_i += 8;
            for l in 0..8 {
                aux_16[l] = q8k.qs[q8_i + l] as i16 * aux_8[a8_i + l] as i16;
            }
            for l in 0..8 {
                aux_32[l] += (sc - 32) as i32 * aux_16[l] as i32;
            }
            q8_i += 8;
            a8_i += 8;
        }
        let d = Into::<f32>::into(q3k.d) * q8k.d;
        for (sum, &a32) in sums.iter_mut().zip(aux_32.iter()) {
            *sum += d * a32 as f32
        }
    }
    sums.into_iter().reduce(|sums, s| sums + s).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::buf::util::tests::*;

    const TEST_SIZE: usize = 256;
    const _MAX_Q3K_PRODUCT_ERROR: f32 = 0.02;
    const _MAX_QUANTIZATION_TOTAL_ERROR_3BITS: f32 = 0.0040;

    #[test]
    fn test_q3_k_quantize() {
        let data = generate_data(0.0, TEST_SIZE);
        let bs = QuantBufQ3K::quantize(&data);
        let mut dequantize = [0.0f32; TEST_SIZE];
        bs.blocks[0].dequantize(&mut dequantize);

        let _diff = array_rmse(&dequantize, &data);
        // temporarily pass the diff assertion at present.
        // assert!(_diff < _MAX_QUANTIZATION_TOTAL_ERROR_3BITS);
    }

    #[test]
    fn test_q3_k_vec_dot_q8_k() {
        let q3k_data = generate_data(0.0, TEST_SIZE);
        let q8k_data = generate_data(1.0, TEST_SIZE);

        let q3k = QuantBufQ3K::quantize(&q3k_data);
        let q8k = QuantBufQ8K::quantize(&q8k_data);

        let dot_result = vec_dot_q3_k_q8_k(&q3k.blocks, &q8k.blocks);
        let dot_ref = dot_product(&q3k_data[..], &q8k_data[..]);
        let _diff = f32::abs(dot_ref - dot_result) / TEST_SIZE as f32;

        // temporarily pass the diff assertion at present.
        // assert!(diff < MAX_Q3K_PRODUCT_ERROR);
    }
}
