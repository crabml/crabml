use std::borrow::Cow;

use half::f16;

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    pub d: f16,       // delta
    pub qs: [i8; 32], // quants
}

impl BlockQ8_0 {
    pub const BLOCK_ELEMS: usize = 32;

    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = self.d.to_f32();
        for (i, v) in buf.iter_mut().enumerate().take(32) {
            *v = self.qs[i] as f32 * d;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ8_0<'a> {
    pub blocks: Cow<'a, [BlockQ8_0]>,
}

impl<'a> QuantBufQ8_0<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ8_0>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ8_0 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8_0, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = super::quantize_f32_q8_0(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ8_0] {
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

    pub fn vec_dot(&self, a_offset: usize, b: &Self, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks()[b_offset / 32..(b_offset + len) / 32];

        super::vec_dot_q8_0_q8_0(abs, bbs)
    }
}

#[cfg(test)]
mod tests {
    use super::super::vec_dot_q8_0_q8_0;
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

        let blocks = QuantBufQ8_0::from_bytes(&buf[0..34]).blocks;
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(blocks[0].qs, [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 7
        ]);

        let bf = QuantBufQ8_0::from_bytes(&buf);
        assert_eq!(bf.len(), 64);
        assert_eq!(bf.dequantize(0).collect::<Vec<_>>(), vec![
            6.0, 9.0, 12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 27.0, 27.0
        ]);
    }

    #[test]
    fn test_vec_dot_q8_0_q8_0() {
        let tests = vec![
            (
                "2*2",
                vec![
                    BlockQ8_0 {
                        qs: [
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        ],
                        d: f16::from_f32(0.4),
                    },
                    BlockQ8_0 {
                        qs: [
                            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
                            -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30,
                            -31, -32,
                        ],
                        d: f16::from_f32(0.7),
                    },
                ],
                vec![
                    BlockQ8_0 {
                        qs: [
                            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
                            14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                        ],
                        d: f16::from_f32(1.3),
                    },
                    BlockQ8_0 {
                        qs: [
                            -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19,
                            -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4,
                            -3, -2, -1,
                        ],
                        d: f16::from_f32(1.4),
                    },
                ],
                8978.046,
            ),
            (
                "1*1",
                vec![BlockQ8_0 {
                    qs: [
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    ],
                    d: f16::from_f32(0.4),
                }],
                vec![BlockQ8_0 {
                    qs: [
                        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
                        13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                    ],
                    d: f16::from_f32(1.3),
                }],
                3110.453,
            ),
        ];

        for (name, abs, bbs, expect) in tests {
            let result = vec_dot_q8_0_q8_0(&abs, &bbs);
            assert_eq!(result, expect, "test: {}", name);
        }
    }
}
