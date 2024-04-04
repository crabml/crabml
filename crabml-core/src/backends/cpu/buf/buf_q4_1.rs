use std::borrow::Cow;

use half::f16;

use super::QuantBufQ8_1;
use crate::backends::cpu::buf::buf_q8_1::BlockQ8_1;
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BlockQ4_1 {
    pub d: f16,       // delta
    pub m: f16,       // min
    pub qs: [u8; 16], // nibbles / quants
}

impl BlockQ4_1 {
    pub fn dequantize(&self, buf: &mut [f32]) {
        let delta = self.d.to_f32();
        let min_val = self.m.to_f32();

        for (i, &quantized) in self.qs.iter().enumerate() {
            let x0 = (quantized & 0x0F) as f32;
            let x1 = ((quantized >> 4) & 0x0F) as f32;

            buf[i * 2] = x0 * delta + min_val;
            buf[i * 2 + 1] = x1 * delta + min_val;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ4_1<'a> {
    pub blocks: Cow<'a, [BlockQ4_1]>,
}

impl<'a> QuantBufQ4_1<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ4_1>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ8_0 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4_1, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q4_1(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ4_1] {
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

    pub fn vec_dot(&self, a_offset: usize, b: &QuantBufQ8_1, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks[b_offset / 32..(b_offset + len) / 32];

        vec_dot_q4_1_q8_1(abs, bbs)
    }
}

pub fn quantize_f32_q4_1(data: &[f32]) -> Vec<BlockQ4_1> {
    let mut bs = Vec::with_capacity(data.len() / 32);
    for chunk in data.chunks(32) {
        // Find the maximum and minimum value in the chunk
        let (min_val, max_val) = chunk
            .iter()
            .fold((f32::MAX, f32::MIN), |(min_val, max_val), &v| {
                (v.min(min_val), v.max(max_val))
            });

        let d = (max_val - min_val) / 15.0; // Compute the scaling factor
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16]; // Initialize the quantized values array

        for i in (0..32).step_by(2) {
            // Scale the value and convert to u8
            let scale_val0 = ((chunk[i] - min_val) * id).round().min(15.0) as u8;
            let scale_val1 = ((chunk[i + 1] - min_val) * id).round().min(15.0) as u8;

            qs[i / 2] = scale_val0 | (scale_val1 << 4);
        }

        bs.push(BlockQ4_1 {
            d: f16::from_f32(d),
            m: f16::from_f32(min_val),
            qs,
        })
    }
    bs
}

pub fn vec_dot_q4_1_q8_1(abs: &[BlockQ4_1], bbs: &[BlockQ8_1]) -> f32 {
    let mut sumf = 0f32;
    for i in 0..abs.len() {
        let mut sumi: i32 = 0;
        for j in 0..16 {
            let v0 = (abs[i].qs[j] & 0x0F) as i32;
            let v1 = ((abs[i].qs[j] >> 4) & 0x0F) as i32;

            sumi += v0 * bbs[i].qs[j] as i32 + v1 * bbs[i].qs[j + 16] as i32;
        }
        sumf += (abs[i].d.to_f32() * bbs[i].d) * sumi as f32 + abs[i].m.to_f32() * bbs[i].s;
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_1_block() {
        assert_eq!(
            std::mem::size_of::<BlockQ4_1>(),
            2 * std::mem::size_of::<f16>() + 16,
            "wrong q4_1 block size/padding"
        );

        let mut buf: [u8; 20] = [0x1; 20];
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

        let blocks = QuantBufQ4_1::from_bytes(&buf[0..20]).blocks;
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(blocks[0].m.to_f32(), 1.0);
        assert_eq!(blocks[0].qs, [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7
        ])
    }

    #[test]
    fn test_q4_1_quantize() {
        let data = vec![
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        let bs = QuantBufQ4_1::quantize(&data);

        assert_eq!(bs.blocks.len(), 1);
        assert_eq!(bs.blocks[0].d.to_f32(), 1.0);
        assert_eq!(bs.blocks[0].m.to_f32(), -8.0);
        assert_eq!(bs.blocks[0].qs, [
            16, 50, 84, 118, 152, 186, 220, 254, 16, 50, 84, 118, 152, 186, 220, 254
        ]);

        let mut dequantize = [0.0f32; 32];
        bs.blocks[0].dequantize(&mut dequantize);
        assert_eq!(dequantize, *data);
    }
}
