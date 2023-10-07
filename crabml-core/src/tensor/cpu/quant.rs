use half::bf16;

pub struct Q8Block {
    d: bf16, // delta
    qs: [u8; 32], // quants
}

pub struct Q8BlockBuf<'a> {
    blocks: &'a [Q8Block],
}