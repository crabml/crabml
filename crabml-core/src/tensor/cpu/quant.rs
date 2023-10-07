use half;
use half::bf16;

#[repr(C, packed)]
#[derive(Debug)]
pub struct Q80Block {
    d: bf16,      // delta
    qs: [u8; 32], // quants
}

impl Q80Block {
    pub fn from_bytes(buf: &[u8]) -> Self {
        assert_eq!(buf.len(), std::mem::size_of::<Q80Block>());
        unsafe { std::ptr::read(buf.as_ptr() as *const Q80Block) }
    }

    pub fn into_iter(self) -> impl Iterator<Item = f32> {
        let d = self.d.to_f32();
        self.qs.into_iter().map(move |q| q as f32 * d)
    }
}

#[derive(Debug)]
pub struct Q80BlockBuf<'a> {
    raw: &'a [u8],
    num_blocks: usize,
}

impl<'a> Q80BlockBuf<'a> {
    pub fn from_raw_bytes(buf: &'a [u8]) -> Self {
        let num_blocks = buf.len() / std::mem::size_of::<Q80Block>();
        Self {
            raw: buf,
            num_blocks,
        }
    }

    pub fn iter_range(
        &self,
        start: usize,
        end: usize,
        step: usize,
    ) -> impl Iterator<Item = f32> + '_ {
        let block_start = start / self.num_blocks;
        let block_size = std::mem::size_of::<Q80Block>();
        let count = end - start;
        self.raw[block_start * block_size..]
            .chunks(block_size)
            .flat_map(|buf| Q80Block::from_bytes(buf).into_iter())
            .step_by(step)
            .take(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q80_block() {
        let mut buf:[u8; 34] = [0x0; 34];
        let d = bf16::from_f32(3.0).to_bits().to_le_bytes();
        buf[0] = d[0];
        buf[1] = d[1];

        let block = Q80Block::from_bytes(&buf);
        assert_eq!(block.d.to_f32(), 3.0);
        assert_eq!(block.qs, [0; 32]);
    }
}