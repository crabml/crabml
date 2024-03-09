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
    #[cfg(target_arch = "aarch64")]
    {
        vec_dot_f16_f16_simd(a, a_offset, b, b_offset, len)
    }

    #[cfg(not(any(target_arch = "aarch64",)))]
    {
        vec_dot_f16_f16_simd_fallback(a, a_offset, b, b_offset, len)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_f16_f16_simd(
    a: &[f16],
    a_offset: usize,
    b: &[f16],
    b_offset: usize,
    k: usize,
) -> f32 {
    use crate::backends::cpu::arch::aarch64 as myaarch64;
    unsafe {
        let mut sumv0 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let mut sumv1 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let k_rounded = k - k % 16;
        for ki in (0..k_rounded).step_by(16) {
            let av0 = myaarch64::vld1q_f16(a.as_ptr().add(a_offset + ki));
            let bv0 = myaarch64::vld1q_f16(b.as_ptr().add(b_offset + ki));
            let av1 = myaarch64::vld1q_f16(a.as_ptr().add(a_offset + ki + 8));
            let bv1 = myaarch64::vld1q_f16(b.as_ptr().add(b_offset + ki + 8));
            sumv0 = myaarch64::vfmaq_f16(sumv0, av0, bv0);
            sumv1 = myaarch64::vfmaq_f16(sumv1, av1, bv1);
        }

        let mut sum = myaarch64::vaddvq_f16(sumv0) + myaarch64::vaddvq_f16(sumv1);
        for ki in k_rounded..k {
            sum += (a.get_unchecked(a_offset + ki) * b.get_unchecked(b_offset + ki)).to_f32();
        }
        sum
    }
}

pub fn vec_dot_f16_f16_fallback(
    a: &[f16],
    a_offset: usize,
    b: &[f16],
    b_offset: usize,
    len: usize,
) -> f32 {
    let ac = &a[a_offset..a_offset + len];
    let bc = &b[b_offset..b_offset + len];
    let mut sum = 0.0;
    for i in 0..len {
        sum += ac[i].to_f32() * bc[i].to_f32();
    }
    sum
}

pub fn vec_dot_f16_f16_strided(
    a: &[f16],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f16],
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        vec_dot_f16_f16_strided_simd(a, a_base, a_stride, k, b)
    }

    #[cfg(not(any(target_arch = "aarch64",)))]
    {
        vec_dot_f16_f16_strided_fallback(a, a_base, a_stride, k, b)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_f16_f16_strided_simd(
    a: &[f16],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f16],
) -> f32 {
    use crate::backends::cpu::arch::aarch64 as myaarch64;
    unsafe {
        let a_ptr = a.as_ptr().add(a_base);

        let mut sumv0 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let mut sumv1 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let k_rounded = k - k % 16;
        for ki in (0..k_rounded).step_by(16) {
            let av_tmp = [
                *a_ptr.add(ki * a_stride),
                *a_ptr.add((ki + 1) * a_stride),
                *a_ptr.add((ki + 2) * a_stride),
                *a_ptr.add((ki + 3) * a_stride),
                *a_ptr.add((ki + 4) * a_stride),
                *a_ptr.add((ki + 5) * a_stride),
                *a_ptr.add((ki + 6) * a_stride),
                *a_ptr.add((ki + 7) * a_stride),
                *a_ptr.add((ki + 8) * a_stride),
                *a_ptr.add((ki + 9) * a_stride),
                *a_ptr.add((ki + 10) * a_stride),
                *a_ptr.add((ki + 11) * a_stride),
                *a_ptr.add((ki + 12) * a_stride),
                *a_ptr.add((ki + 13) * a_stride),
                *a_ptr.add((ki + 14) * a_stride),
                *a_ptr.add((ki + 15) * a_stride),
            ];
            let av0 = myaarch64::vld1q_f16(av_tmp.as_ptr());
            let bv0 = myaarch64::vld1q_f16(b.as_ptr().add(ki));
            let av1 = myaarch64::vld1q_f16(av_tmp.as_ptr().add(8));
            let bv1 = myaarch64::vld1q_f16(b.as_ptr().add(ki + 8));
            sumv0 = myaarch64::vfmaq_f16(sumv0, av0, bv0);
            sumv1 = myaarch64::vfmaq_f16(sumv1, av1, bv1);
        }

        let mut sum = myaarch64::vaddvq_f16(sumv0) + myaarch64::vaddvq_f16(sumv1);
        for ki in k_rounded..k {
            sum += (a.get_unchecked(a_base + ki * a_stride) * b.get_unchecked(ki)).to_f32();
        }
        sum
    }
}

pub fn vec_dot_f16_f16_strided_fallback(
    a: &[f16],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f16],
) -> f32 {
    let mut sum: f16 = f16::ZERO;
    let k_rounded = k - k % 4;
    for ki in (0..k_rounded).step_by(4) {
        sum += a[a_base + ki * a_stride] * b[ki];
        sum += a[a_base + (ki + 1) * a_stride] * b[ki + 1];
        sum += a[a_base + (ki + 2) * a_stride] * b[ki + 2];
        sum += a[a_base + (ki + 3) * a_stride] * b[ki + 3];
    }
    for ki in (k_rounded..k).step_by(1) {
        sum += a[a_base + ki * a_stride] * b[ki];
    }
    sum.to_f32()
}
