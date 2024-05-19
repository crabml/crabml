use std::borrow::Cow;
use std::cmp;

use bytemuck::Pod;
use bytemuck::Zeroable;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use half::f16;

use super::QuantBufQ8_0;
use crate::cpu::buf::buf_q8_0::BlockQ8_0;

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_0 {
    d: f16,       // delta
    qh: [u8; 4],  // 5-bit quants
    qs: [u8; 16], // quants
}

impl BlockQ5_0 {
    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d.to_f32();
        let qh: u32 = LittleEndian::read_u32(&self.qh);

        for i in 0..16 {
            let xh_0: u8 = (((qh >> i) << 4) & 0x10) as u8;
            let xh_1: u8 = ((qh >> (i + 12)) & 0x10) as u8;

            let x0 = ((self.qs[i] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((self.qs[i] >> 4) | xh_1) as i32 - 16;

            buf[i] = (x0 as f32) * d;
            buf[i + 16] = (x1 as f32) * d;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ5_0<'a> {
    pub blocks: Cow<'a, [BlockQ5_0]>,
}

impl<'a> QuantBufQ5_0<'_> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ5_0>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ5_0 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ5_0, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q5_0(data);
        Self { blocks: bs.into() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.blocks)
    }

    fn blocks(&self) -> &[BlockQ5_0] {
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

        vec_dot_q5_0_q8_0(abs, bbs)
    }
}

pub fn quantize_f32_q5_0(data: &[f32]) -> Vec<BlockQ5_0> {
    let mut bs = Vec::with_capacity(data.len() / 32);
    for chunk in data.chunks(32) {
        let mut max_val = 0.0;
        let mut max_abs_val = 0.0;

        for &v in chunk {
            let abs_value = v.abs();
            if max_abs_val < abs_value {
                max_abs_val = abs_value;
                max_val = v;
            }
        }

        let d = max_val / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qh = [0u8; 4];
        let mut iqh = 0u32;

        let mut qs = [0u8; 16];

        for (i, q) in qs.iter_mut().take(16).enumerate() {
            // Scale the value and convert to u8
            let x0 = chunk[i] * id;
            let x1 = (chunk[i + 16]) * id;

            let xi0 = cmp::min((x0 + 16.5) as i8, 31) as u8;
            let xi1 = cmp::min((x1 + 16.5) as i8, 31) as u8;

            *q = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // get the 5-th bit and store it in qh at the right position
            iqh |= ((xi0 as u32 & 0x10) >> 4) << i;
            iqh |= ((xi1 as u32 & 0x10) >> 4) << (i + 16);
        }
        LittleEndian::write_u32(&mut qh, iqh);
        bs.push(BlockQ5_0 {
            d: f16::from_f32(d),
            qh,
            qs,
        })
    }
    bs
}

pub fn vec_dot_q5_0_q8_0(abs: &[BlockQ5_0], bbs: &[BlockQ8_0]) -> f32 {
    let mut sumf: f32 = 0f32;
    for i in 0..bbs.len() {
        let qh: u32 = LittleEndian::read_u32(&abs[i].qh);
        let mut sumi: i32 = 0;

        for j in 0..16 {
            let xh_0 = ((qh & (1 << j)) >> j) << 4;
            let xh_1 = (qh & (1 << (j + 16))) >> (j + 12);

            let x0: i32 = ((abs[i].qs[j] as i32 & 0x0F) | xh_0 as i32) - 16;
            let x1: i32 = ((abs[i].qs[j] as i32 >> 4) | xh_1 as i32) - 16;
            sumi += (x0 * bbs[i].qs[j] as i32) + (x1 * bbs[i].qs[j + 16] as i32);
        }
        sumf += sumi as f32 * f16::to_f32(abs[i].d) * f16::to_f32(bbs[i].d);
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_vector(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|x| (1. * x).round() / 1.)
            .collect::<Vec<_>>()
    }

    #[test]
    fn test_q5_0_block() {
        assert_eq!(
            std::mem::size_of::<BlockQ5_0>(),
            std::mem::size_of::<f16>() + 4 + 16,
            "wrong q5_0 block size/padding"
        );

        let mut buf: [u8; 22] = [0x1; 22];
        let delta = f16::from_f32(3.0).to_bits().to_le_bytes();

        buf[0] = delta[0];
        buf[1] = delta[1];
        buf[2] = 2;
        buf[3] = 3;
        buf[4] = 4;
        buf[2 + 15] = 7;

        let blocks = QuantBufQ5_0::from_bytes(&buf[0..22]).blocks;
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(blocks[0].qh, [2, 3, 4, 1]);
        assert_eq!(blocks[0].qs, [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1
        ])
    }

    #[test]
    fn test_q5_0_quantize() {
        let data = vec![
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        let bs = QuantBufQ5_0::quantize(&data);

        assert_eq!(bs.blocks.len(), 1);
        assert_eq!(bs.blocks[0].d.to_f32(), 0.5);
        assert_eq!(bs.blocks[0].qs, [
            0, 34, 68, 102, 136, 170, 204, 238, 0, 34, 68, 102, 136, 170, 204, 238
        ]);

        let mut dequantize = [0.0f32; 32];
        bs.blocks[0].dequantize(&mut dequantize);
        assert_eq!(round_vector(&dequantize), *data);
    }
}
