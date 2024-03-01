//! Inspired a lot by [ggml](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c)

use std::arch::x86_64::*;

use half::f16;

use super::BlockQ8_0;

pub fn quantize_f32_q8_0(data: &[f32]) -> Vec<BlockQ8_0> {
    let mut bs = Vec::with_capacity(data.len() / 32);

    unsafe {
        for chunk in data.chunks(32) {
            let mut max_abs_values = _mm256_setzero_ps();

            for &value in chunk {
                let val_vec = _mm256_set1_ps(value);
                max_abs_values = _mm256_max_ps(
                    max_abs_values,
                    _mm256_andnot_ps(_mm256_set1_ps(-0.0), val_vec),
                );
            }

            let max_abs_value = {
                let mut max_vals = [0.0; 8];
                _mm256_storeu_ps(max_vals.as_mut_ptr(), max_abs_values);
                *max_vals
                    .iter()
                    .max_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap()
            };

            let d = max_abs_value / 127.0;
            let d_vec = _mm256_set1_ps(d);
            let mut qs = [0_i8; 32];
            let mut temp = [0i32; 8]; // Temporary array to hold intermediate results

            for (chunk_index, values) in chunk.chunks(8).enumerate() {
                let values_vec = _mm256_loadu_ps(values.as_ptr());
                let scaled_vec = _mm256_div_ps(values_vec, d_vec);
                let clamped_vec = _mm256_max_ps(
                    _mm256_set1_ps(i8::MIN as f32),
                    _mm256_min_ps(_mm256_set1_ps(i8::MAX as f32), scaled_vec),
                );
                let converted_vec = _mm256_cvtps_epi32(clamped_vec);
                _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, converted_vec);

                for (i, &value) in temp.iter().enumerate() {
                    qs[chunk_index * 8 + i] = value as i8;
                }
            }

            bs.push(BlockQ8_0 {
                d: f16::from_f32(d),
                qs,
            });
        }
    }

    bs
}

pub fn vec_dot_q8_0_q8_0(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();

        for (abs, bbs) in abs.iter().zip(bbs) {
            let d = _mm256_set1_ps(abs.d.to_f32() * bbs.d.to_f32());

            let qa = _mm256_loadu_si256(abs.qs.as_ptr() as *const __m256i);
            let qb = _mm256_loadu_si256(bbs.qs.as_ptr() as *const __m256i);

            let q = mul_sum_i8_pairs_float(qa, qb);

            acc = _mm256_fmadd_ps(d, q, acc);
        }

        hsum_float_8(acc)
    }
}

/// TODO: Adding AVX-VNNI support so that we can use `_mm256_dpbssd_epi32`
#[inline]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    // Get absolute values of x vectors
    let ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

#[inline]
unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let axl = _mm256_castsi256_si128(ax);
    let axh = _mm256_extractf128_si256(ax, 1);
    let syl = _mm256_castsi256_si128(sy);
    let syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    let dotl = _mm_maddubs_epi16(axl, syl);
    let doth = _mm_maddubs_epi16(axh, syh);
    sum_i16_pairs_float(doth, dotl)
}

#[inline]
unsafe fn sum_i16_pairs_float(xh: __m128i, xl: __m128i) -> __m256 {
    let ones = _mm_set1_epi16(1);
    let summed_pairsl = _mm_madd_epi16(ones, xl);
    let summed_pairsh = _mm_madd_epi16(ones, xh);
    let summed_pairs = _mm256_set_m128i(summed_pairsh, summed_pairsl);
    _mm256_cvtepi32_ps(summed_pairs)
}

/// horizontally add 8 floats
#[inline]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let res = _mm256_extractf128_ps(x, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}
