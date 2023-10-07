use half;
use half::bf16;

#[repr(C, packed)]
#[derive(Debug)]
pub struct Q8Block {
    d: bf16,      // delta
    qs: [u8; 32], // quants
}

impl Q8Block {
    pub fn from_bytes(buf: &[u8]) -> Self {
        assert_eq!(buf.len(), std::mem::size_of::<Q8Block>());
        unsafe { std::ptr::read(buf.as_ptr() as *const Q8Block) }
    }

    pub fn into_iter(self) -> impl Iterator<Item = f32> {
        let d = self.d.to_f32();
        self.qs.into_iter().map(move |q| q as f32 * d)
    }
}

#[derive(Debug)]
pub struct Q8BlockBuf<'a> {
    raw: &'a [u8],
    num_blocks: usize,
}

impl<'a> Q8BlockBuf<'a> {
    pub fn from_raw_bytes(buf: &'a [u8]) -> Self {
        let num_blocks = buf.len() / std::mem::size_of::<Q8Block>();
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
        let block_size = std::mem::size_of::<Q8Block>();
        let count = end - start;
        self.raw[block_start * block_size..]
            .chunks(block_size)
            .flat_map(|buf| Q8Block::from_bytes(buf).into_iter())
            .step_by(step)
            .take(count)
    }
}
