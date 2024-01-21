use std::simd::f32x8;
use std::simd::prelude::SimdFloat;

use half::f16;

use super::buf::QuantizedBuf;

#[derive(Debug, Clone)]
pub struct QuantBufQ8_0<'a> {
    raw: &'a [u8],
    num_blocks: usize,
}

impl<'a> QuantBufQ8_0<'a> {
    pub fn from_bytes(buf: &'a [u8]) -> Self {
        let block_mem = std::mem::size_of::<BlockQ8_0>();
        // assert!(buf.len() % block_mem == 0);
        let num_blocks = buf.len() / block_mem;
        Self {
            raw: buf,
            num_blocks,
        }
    }

    fn blocks(&self) -> &[BlockQ8_0] {
        BlockQ8_0::from_bytes(self.raw)
    }

    pub fn len(&self) -> usize {
        self.num_blocks * 32
    }

    pub fn dequantize_from(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert!(start % 32 == 0);

        let block_start = start / 32;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0.0; 32];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }
}

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}

impl BlockQ8_0 {
    pub const BLOCK_ELEMS: usize = 32;

    pub fn from_bytes(data: &[u8]) -> &[BlockQ8_0] {
        let size = std::mem::size_of::<BlockQ8_0>();
        assert!(
            data.len() % size == 0,
            "data length must be a multiple of QuantBlockQ8_0 size"
        );
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8_0, data.len() / size) }
    }

    pub fn quantize(data: &[f32]) -> Vec<BlockQ8_0> {
        let mut bs: Vec<BlockQ8_0> = vec![];
        let chunks = data.chunks(32);
        for chunk in chunks {
            let mut max = f32::MIN;
            for i in 0..32 {
                if chunk[i] > max {
                    max = chunk[i];
                }
            }
            let d = f16::from_f32(max / 127.0);
            let mut qs = [0_i8; 32];
            for i in 0..32 {
                qs[i] = (chunk[i] / d.to_f32()).round() as i8;
            }
            bs.push(BlockQ8_0 { d, qs })
        }
        bs
    }

    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d.to_f32();
        for i in 0..32 {
            let q = self.qs[i];
            buf[i] = q as f32 * d;
        }
    }
}

impl<'a> QuantizedBuf for QuantBufQ8_0<'a> {
    fn vec_dot_f32(&self, offset: usize, x: &[f32]) -> f32 {
        let blocks = BlockQ8_0::from_bytes(self.raw);
        let row = &blocks[offset / 32..(offset + x.len()) / 32];
        assert!(row.len() * 32 == x.len());
        let mut sum = 0.0;
        for i in 0..row.len() {
            let block = &row[i];
            let d = block.d.to_f32();
            let mut sum_block = 0.0;
            for j in 0..4 {
                let qs = &block.qs[j * 8..(j + 1) * 8];
                let qv = f32x8::from_array([
                    qs[0] as f32,
                    qs[1] as f32,
                    qs[2] as f32,
                    qs[3] as f32,
                    qs[4] as f32,
                    qs[5] as f32,
                    qs[6] as f32,
                    qs[7] as f32,
                ]);
                let xv = f32x8::from_slice(&x[i * 32 + j * 8..i * 32 + (j + 1) * 8]);
                sum_block += (qv * xv).reduce_sum();
            }
            sum += sum_block * d;
        }
        sum
    }
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

        let blocks = BlockQ8_0::from_bytes(&buf[0..34]);
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(blocks[0].qs, [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 7
        ]);

        let bf = QuantBufQ8_0::from_bytes(&buf);
        assert_eq!(bf.len(), 64);
        assert_eq!(bf.dequantize_from(0).collect::<Vec<_>>(), vec![
            6.0, 9.0, 12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 27.0, 27.0
        ]);
    }
}
