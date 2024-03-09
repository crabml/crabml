use std::borrow::Cow;
use std::slice;

use half::f16;

pub fn f16_buf_from_bytes<'a>(buf: &[u8]) -> Cow<'a, [f16]> {
    let len = buf.len();
    assert_eq!(
        len % std::mem::size_of::<f32>(),
        0,
        "Length of slice must be multiple of f32 size"
    );
    let new_len = len / std::mem::size_of::<f16>();
    let ptr = buf.as_ptr() as *const f16;
    let f16_buf = unsafe { slice::from_raw_parts(ptr, new_len) };
    f16_buf.into()
}

pub fn dequantize_f16_buf(buf: &[f16], start: usize) -> impl Iterator<Item = f32> + '_ {
    buf.iter().skip(start).map(|x| x.to_f32())
}

pub fn quantize_f32_f16<'a>(buf: &[f32]) -> Cow<'a, [f16]> {
    buf.iter()
        .map(|x| f16::from_f32(*x))
        .collect::<Vec<_>>()
        .into()
}

pub fn vec_dot_f16_f16(a: &[f16], a_offset: usize, b: &[f16], b_offset: usize, len: usize) -> f32 {
    let ac = &a[a_offset..a_offset + len];
    let bc = &b[b_offset..b_offset + len];
    let mut sum = 0.0;
    for i in 0..len {
        sum += ac[i].to_f32() * bc[i].to_f32();
    }
    sum
}
