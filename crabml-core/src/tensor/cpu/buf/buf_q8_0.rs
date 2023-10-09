use half::f16;
use std::simd::{f32x8, SimdFloat};

use super::buf::BlockVecCompute;

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct QuantBlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}

impl QuantBlockQ8_0 {
    pub const BLOCK_ELEMS: usize = 32;

    pub fn from_bytes(data: &[u8]) -> &[QuantBlockQ8_0] {
        let size = std::mem::size_of::<QuantBlockQ8_0>();
        assert!(
            data.len() % size == 0,
            "data length must be a multiple of QuantBlockQ8_0 size"
        );
        unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const QuantBlockQ8_0, data.len() / size)
        }
    }

    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d.to_f32();
        for i in 0..32 {
            let q = self.qs[i];
            buf[i] = q as f32 * d;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBuf8_0<'a> {
    raw: &'a [u8],
    num_blocks: usize,
}

impl<'a> QuantBuf8_0<'a> {
    pub fn from_bytes(buf: &'a [u8]) -> Self {
        let block_mem = std::mem::size_of::<QuantBlockQ8_0>();
        // assert!(buf.len() % block_mem == 0);
        let num_blocks = buf.len() / block_mem;
        Self {
            raw: buf,
            num_blocks,
        }
    }

    pub fn len(&self) -> usize {
        self.num_blocks * 32
    }

    pub fn iter_range(
        &'a self,
        start: usize,
        end: usize,
        step: usize,
    ) -> impl Iterator<Item = f32> + 'a {
        BlockBufIterQ8_0 {
            buf: &self,
            pos: start,
            end: end,
            step: step,
            current_f32_buf: [0.0; 32],
            current_block: usize::MAX,
        }
    }
}

impl<'a> BlockVecCompute for QuantBuf8_0<'a> {
    type BlockType = QuantBlockQ8_0;

    fn block_elms(&self) -> usize {
        QuantBlockQ8_0::BLOCK_ELEMS
    }

    fn blocks(&self) -> &[Self::BlockType] {
        QuantBlockQ8_0::from_bytes(self.raw)
    }

    fn vec_dot_f32(&self, row: &[Self::BlockType], x: &[f32]) -> f32 {
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

pub struct BlockBufIterQ8_0<'a> {
    buf: &'a QuantBuf8_0<'a>,
    current_f32_buf: [f32; 32],
    current_block: usize,
    pos: usize,
    end: usize,
    step: usize,
}

impl<'a> Iterator for BlockBufIterQ8_0<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }

        let block_idx = self.pos / QuantBlockQ8_0::BLOCK_ELEMS;
        if block_idx != self.current_block {
            let block = &self.buf.blocks()[block_idx];
            block.dequantize(&mut self.current_f32_buf);
            self.current_block = block_idx;
        }

        let val = self.current_f32_buf[self.pos % 32];
        self.pos += self.step;
        Some(val)
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

        let blocks = QuantBlockQ8_0::from_bytes(&buf[0..34]);
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(
            blocks[0].qs,
            [
                2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 7
            ]
        );

        let bf = QuantBuf8_0::from_bytes(&buf);
        assert_eq!(bf.len(), 64);
        assert_eq!(
            bf.iter_range(0, bf.len(), 1).collect::<Vec<_>>(),
            vec![
                6.0, 9.0, 12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0,
                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 27.0, 27.0
            ]
        );
        assert_eq!(bf.iter_range(10, bf.len(), 1).collect::<Vec<_>>().len(), 54);
    }
}
