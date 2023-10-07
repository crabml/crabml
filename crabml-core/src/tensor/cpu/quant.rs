use half::f16;

#[repr(C, packed)]
#[derive(Debug)]
pub struct BlockQ8_0 {
    d: f16,      // delta
    qs: [i8; 32], // quants
}

impl BlockQ8_0 {
    pub const BLOCK_ELEMS: usize = 32;

    pub fn from_bytes(buf: &[u8]) -> Self {
        assert_eq!(buf.len(), std::mem::size_of::<BlockQ8_0>());
        unsafe { std::ptr::read(buf.as_ptr() as *const BlockQ8_0) }
    }

    pub fn at(&self, idx: usize) -> f32 {
        let d = self.d.to_f32();
        let q = self.qs[idx];
        q as f32 * d
    }
}

#[derive(Debug, Clone)]
pub struct BlockBufQ8_0<'a> {
    raw: &'a [u8],
    num_blocks: usize,
}

impl<'a> BlockBufQ8_0<'a> {
    pub fn from_bytes(buf: &'a [u8]) -> Self {
        let block_mem = std::mem::size_of::<BlockQ8_0>();
        assert!(buf.len() % block_mem == 0);
        let num_blocks = buf.len() / block_mem;
        Self {
            raw: buf,
            num_blocks,
        }
    }

    pub fn len(&self) -> usize {
        self.num_blocks * 32
    }

    pub fn block_at(&self, idx: usize) -> BlockQ8_0 {
        let block_mem = std::mem::size_of::<BlockQ8_0>();
        let buf = &self.raw[idx * block_mem..(idx + 1) * block_mem];
        BlockQ8_0::from_bytes(buf)
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
            val: 0.0,
        }
    }
}

pub struct BlockBufIterQ8_0<'a> {
    buf: &'a BlockBufQ8_0<'a>,
    pos: usize,
    end: usize,
    step: usize,
    val: f32,
}

impl<'a> Iterator for BlockBufIterQ8_0<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let block_idx = self.pos / BlockQ8_0::BLOCK_ELEMS;

        if self.pos >= self.end {
            return None;
        }

        let block = self.buf.block_at(block_idx);
        self.val = block.at(self.pos % 32);
        self.pos += self.step;
        Some(self.val)
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
        buf[2+31] = 7;
        buf[34] = d[0];
        buf[35] = d[1];
        buf[66] = 9;
        buf[67] = 9;

        let block = BlockQ8_0::from_bytes(&buf[0..34]);
        assert_eq!(block.d.to_f32(), 3.0);
        assert_eq!(block.qs, [2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7]);

        let bf = BlockBufQ8_0::from_bytes(&buf);
        assert_eq!(bf.len(), 64);
        assert_eq!(bf.iter_range(0, bf.len(), 1).collect::<Vec<_>>(), vec![6.0, 9.0, 12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 27.0, 27.0]);
        assert_eq!(bf.iter_range(10, bf.len(), 1).collect::<Vec<_>>().len(), 54);
    }
}
