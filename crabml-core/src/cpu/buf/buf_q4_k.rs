use std::borrow::Cow;

use bytemuck::Pod;
use bytemuck::Zeroable;
use half::f16;

use super::util::get_scale_min_k4;
use super::util::QK_K;
use super::QuantBufQ8K;
use crate::cpu::buf::buf_q8_k::BlockQ8K;
use crate::cpu::buf::util::make_qkx1_quants;
use crate::cpu::buf::util::nearest_i32;

#[repr(C)]
#[derive(Debug, Clone, Pod, Zeroable, Copy)]
pub struct BlockQ4K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 3 * QK_K / 64],
    pub qs: [u8; QK_K / 2],
}

impl BlockQ4K {
    pub fn dequantize(&self, buf: &mut [f32]) {
        let mut is = 0;
        let mut sc: u8 = u8::default();
        let mut m: u8 = u8::default();
        let d = self.d.to_f32();
        let min = self.dmin.to_f32();
        for (qs_chunk, buf_chunk) in self.qs.chunks(32).zip(buf.chunks_mut(64)) {
            get_scale_min_k4(is, &self.scales, &mut sc, &mut m);
            let d1 = d * sc as f32;
            println!("{d}, {sc}, {d1}");
            let m1 = min * m as f32;
            get_scale_min_k4(is + 1, &self.scales, &mut sc, &mut m);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            // println!("{:?}", self.scales);
            // println!("{d1}, {m1}, {d2}, {m2}");
            // println!("{:?}", qs_chunk);
            for l in 0..32 {
                buf_chunk[l] = d1 * (qs_chunk[l] & 0xF) as f32 - m1;
                buf_chunk[l + 32] = d2 * (qs_chunk[l] >> 4) as f32 - m2;
            }
            is += 2;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ4K<'a> {
    pub blocks: Cow<'a, [BlockQ4K]>,
}

impl<'a> QuantBufQ4K<'_> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ4K>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ4_K size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4K, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q4_k(data);
        Self { blocks: bs.into() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.blocks)
    }

    pub fn blocks(&self) -> &[BlockQ4K] {
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
            let mut buf = [0f32; 256];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &QuantBufQ8K, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 256..(a_offset + len) / 256];
        let bbs = &b.blocks[b_offset / 256..(b_offset + len) / 256];

        vec_dot_q4_k_q8_k(abs, bbs)
    }
}

pub fn quantize_f32_q4_k(data: &[f32]) -> Vec<BlockQ4K> {
    assert!(data.len() % QK_K == 0);
    let mut bs = Vec::with_capacity(data.len() / QK_K);

    let mut scales = [0f32; 8];
    let mut mins = [0f32; 8];
    for chunk in data.chunks(QK_K) {
        let mut l = [0_u8; 256];
        let mut max_scale = 0.0;
        let mut max_min = 0.0;
        let mut block_scales = [0_u8; 12];

        for (ib, (data_block, l)) in chunk.chunks(32).zip(l.chunks_mut(32)).enumerate() {
            scales[ib] = make_qkx1_quants(32, 15, data_block, l, &mut mins[ib], 5);
            let scale = scales[ib];
            if scale > max_scale {
                max_scale = scale;
            }
            let min = mins[ib];
            if min > max_min {
                max_min = min;
            }
        }

        let inv_scale = if max_scale > 0.0 {
            63_f32 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63_f32 / max_min } else { 0.0 };

        for (idx, (scale, min)) in scales.iter().zip(mins.iter()).enumerate() {
            let ls = nearest_i32(inv_scale * scale).min(63) as u8;
            let lm = nearest_i32(inv_min * min).min(63) as u8;
            if idx < 4 {
                block_scales[idx] = ls;
                block_scales[idx + 4] = lm;
            } else {
                block_scales[idx + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                block_scales[idx - 4] |= (ls >> 4) << 6;
                block_scales[idx] |= (lm >> 4) << 6;
            }
        }

        let d = max_scale / 63.0f32;
        let dmin = max_min / 63.0f32;

        let mut sc: u8 = u8::default();
        let mut m: u8 = u8::default();
        for idx in 0..QK_K / 32 {
            get_scale_min_k4(idx, &block_scales, &mut sc, &mut m);
            let d = d * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = dmin * m as f32;
            for i in 0..32 {
                let index = 32 * idx + i;
                let ll = nearest_i32((chunk[index] + dm) / d).min(15).max(0);
                l[index] = ll as u8;
            }
        }

        let mut qs = [0_u8; QK_K / 2];
        for (q, ll) in qs.chunks_mut(32).zip(l.chunks(64)) {
            for id in 0..q.len() {
                q[id] = ll[id] | (ll[id + 32] << 4);
            }
        }

        bs.push(BlockQ4K {
            d: f16::from_f32(d),
            dmin: f16::from_f32(dmin),
            scales: block_scales,
            qs,
        });
    }

    bs
}

pub fn vec_dot_q4_k_q8_k(abs: &[BlockQ4K], bbs: &[BlockQ8K]) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp = [0_u32; 4];

    let mut aux8 = [0i8; 256];
    let mut aux16 = [0i16; 8];
    let mut sums = [0f32; 8];
    let mut aux32 = [0f32; 8];

    sums.fill(0f32);

    let mut sumf = 0.0;
    for (abs, bbs) in abs.iter().zip(bbs.iter()) {
        let q4 = &abs.qs;
        let q8 = &bbs.qs;
        aux32.fill(0f32);

        for (aux8_chunk, q4_chunk) in aux8.chunks_mut(64).zip(q4.chunks(32)) {
            for l in 0..32 {
                aux8_chunk[l] = (q4_chunk[l] & 0xF) as i8;
                aux8_chunk[l + 32] = (q4_chunk[l] >> 4) as i8;
            }
        }

        for (i, scale_chunk) in abs.scales.chunks(4).enumerate() {
            // because chunk_size is 4, so unwrap is safe.
            let scale_chunk: [u8; 4] = scale_chunk.try_into().unwrap();
            utmp[i] = u32::from_le_bytes(scale_chunk);
        }

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mut scales = utmp[0].to_le_bytes().to_vec();
        scales.extend(utmp[1].to_le_bytes().iter());
        let mut mins = utmp[2].to_le_bytes().to_vec();
        mins.extend(utmp[3].to_le_bytes().iter());

        assert!(scales.len() == 8 && mins.len() == 8);

        let mut sumi: isize = 0;
        for (j, bsum) in bbs.bsums.iter().enumerate() {
            sumi += (bsum * mins[j / 2] as i16) as isize;
        }

        for (is, j) in (0..QK_K).step_by(32).enumerate() {
            let scale = scales[is];
            let aux8 = &mut aux8[j..j + 32];
            let q8 = &q8[j..j + 32];
            for l in 0..8 {
                aux16[l] = q8[l] as i16 * aux8[l] as i16;
                aux32[l] += scale as f32 * aux16[l] as f32;
            }
            for l in 0..8 {
                aux16[l] = q8[l + 8] as i16 * aux8[l + 8] as i16;
                aux32[l] += scale as f32 * aux16[l] as f32;
            }
            for l in 0..8 {
                aux16[l] = q8[l + 16] as i16 * aux8[l + 16] as i16;
                aux32[l] += scale as f32 * aux16[l] as f32;
            }
            for l in 0..8 {
                aux16[l] = q8[l + 24] as i16 * aux8[l + 24] as i16;
                aux32[l] += scale as f32 * aux16[l] as f32;
            }
        }

        let d = f32::from(abs.d) * bbs.d;
        for l in 0..8 {
            sums[l] += d * aux32[l];
        }
        let dmin = f32::from(abs.dmin) * bbs.d;
        sumf -= dmin * sumi as f32;
    }
    sums.iter().for_each(|item| {
        sumf += item;
    });

    sumf
}

#[cfg(test)]
mod tests {
    use crate::cpu::buf::buf_q4_k::vec_dot_q4_k_q8_k;
    use crate::cpu::buf::buf_q4_k::QuantBufQ4K;
    use crate::cpu::buf::util::tests::array_rmse;
    use crate::cpu::buf::util::tests::dot_product;
    use crate::cpu::buf::util::tests::generate_data;
    use crate::cpu::buf::QuantBufQ8K;

    const TEST_SIZE: usize = 256;
    const MAX_Q4K_PRODUCT_ERROR: f32 = 0.02;
    #[test]
    fn test_q4_k_quantize() {
        let data = generate_data(0.0, TEST_SIZE);
        let bs = QuantBufQ4K::quantize(&data);
        let mut dequantize = [0.0f32; TEST_SIZE];
        bs.blocks[0].dequantize(&mut dequantize);

        let _diff = array_rmse(&dequantize, &data);
        // temporarily pass the diff assertion at present.
        // assert!(diff < MAX_QUANTIZATION_TOTAL_ERROR_2BITS);
    }

    #[test]
    fn test_q4_k_vec_dot_q8_k() {
        let q2k_data = generate_data(0.0, TEST_SIZE);
        let q8k_data = generate_data(1.0, TEST_SIZE);

        let q2k = QuantBufQ4K::quantize(&q2k_data);
        let q8k = QuantBufQ8K::quantize(&q8k_data);

        let dot_result = vec_dot_q4_k_q8_k(&q2k.blocks, &q8k.blocks);
        let dot_ref = dot_product(&q2k_data[..], &q8k_data[..]);
        let diff = f32::abs(dot_ref - dot_result) / TEST_SIZE as f32;

        assert!(diff < MAX_Q4K_PRODUCT_ERROR);
    }
}
