use std::borrow::Cow;
use std::slice;

use half::f16;

pub fn f32_buf_from_bytes<'a>(buf: &[u8]) -> Cow<'a, [f32]> {
    let len = buf.len();
    assert_eq!(
        len % std::mem::size_of::<f32>(),
        0,
        "Length of slice must be multiple of f32 size"
    );
    let new_len = len / std::mem::size_of::<f32>();
    let ptr = buf.as_ptr() as *const f32;
    let f32_buf = unsafe { slice::from_raw_parts(ptr, new_len) };
    f32_buf.into()
}

pub fn vec_dot_f32_f32(a: &[f32], a_offset: usize, b: &[f32], b_offset: usize, len: usize) -> f32 {
    let ac = &a[a_offset..a_offset + len];
    let bc = &b[b_offset..b_offset + len];
    let mut sum = 0.0;
    for i in 0..len {
        sum += ac[i] * bc[i];
    }
    sum
}

pub fn exp_f32_cached(x: f32, cache: &[f16]) -> f32 {
    let cache_ptr = cache.as_ptr();
    let x16 = f16::from_f32(x);
    let x16n = x16.to_bits();

    unsafe { (*cache_ptr.add(x16n as usize)).to_f32() }
}

/// vec_dot_f32_f32_strided is called in batch_matmul_vec, which is used in the computation of the
/// scaled dot product attention in the transformer model. the lhs are allowed to be not contiguous,
/// while the rhs are required to be contiguous.
pub fn vec_dot_f32_f32_strided(
    a: &[f32],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f32],
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        vec_dot_f32_f32_strided_simd(a, a_base, a_stride, k, b)
    }
    #[cfg(target_arch = "x86_64")]
    #[cfg(target_feature = "avx2")]
    {
        vec_dot_f32_f32_strided_simd(a, a_base, a_stride, k, b)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        vec_dot_f32_f32_strided_fallback(a, a_base, a_stride, k, b)
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
fn vec_dot_f32_f32_strided_fallback(
    a: &[f32],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f32],
) -> f32 {
    let mut sum = 0.0;
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
    sum
}

#[cfg(target_arch = "aarch64")]
fn vec_dot_f32_f32_strided_simd(
    a: &[f32],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f32],
) -> f32 {
    use std::arch::aarch64;

    unsafe {
        let a_ptr = a.as_ptr().add(a_base);

        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        let k_rounded = k - k % 8;
        for ki in (0..k_rounded).step_by(8) {
            let av_tmp = [
                *a_ptr.add(ki * a_stride),
                *a_ptr.add((ki + 1) * a_stride),
                *a_ptr.add((ki + 2) * a_stride),
                *a_ptr.add((ki + 3) * a_stride),
                *a_ptr.add((ki + 4) * a_stride),
                *a_ptr.add((ki + 5) * a_stride),
                *a_ptr.add((ki + 6) * a_stride),
                *a_ptr.add((ki + 7) * a_stride),
            ];
            let av0 = aarch64::vld1q_f32(av_tmp.as_ptr());
            let bv0 = aarch64::vld1q_f32(b.as_ptr().add(ki));
            let av1 = aarch64::vld1q_f32(av_tmp.as_ptr().add(4));
            let bv1 = aarch64::vld1q_f32(b.as_ptr().add(ki + 4));
            sumv0 = aarch64::vfmaq_f32(sumv0, av0, bv0);
            sumv1 = aarch64::vfmaq_f32(sumv1, av1, bv1);
        }

        let mut sum = aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1);
        for ki in k_rounded..k {
            sum += a[a_base + ki * a_stride] * b[ki];
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
fn vec_dot_f32_f32_strided_simd(
    a: &[f32],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let a_ptr = a.as_ptr().add(a_base);

        let mut sumv = _mm256_setzero_ps();
        let k_rounded_down = k - k % 8; // Round down to the nearest multiple of 8

        for ki in (0..k_rounded_down).step_by(8) {
            let mut av_tmp = [0.0_f32; 8];
            // Load elements from 'a' with stride
            for i in 0..8 {
                av_tmp[i] = *a_ptr.add(ki * a_stride + i * a_stride);
            }
            let av = _mm256_loadu_ps(av_tmp.as_ptr());
            let bv = _mm256_loadu_ps(b.as_ptr().add(ki));
            // Fused multiply-add operation: sumv += av * bv
            sumv = _mm256_fmadd_ps(av, bv, sumv);
        }

        // Horizontal sum of the vector elements
        let mut sum_arr = [0.0_f32; 8];
        _mm256_storeu_ps(sum_arr.as_mut_ptr(), sumv);
        let partial_sum = sum_arr.iter().sum::<f32>();

        // Scalar computation for the remaining elements
        let mut scalar_sum = 0.0;
        for ki in k_rounded_down..k {
            scalar_sum += a[a_base + ki * a_stride] * b[ki];
        }

        partial_sum + scalar_sum
    }
}
