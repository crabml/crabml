use std::arch::x86_64::*;

/// TODO: Adding AVX-VNNI support so that we can use `_mm256_dpbssd_epi32`
#[inline]
#[allow(dead_code)]
pub unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    // Get absolute values of x vectors
    let ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

#[inline]
pub unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
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
pub unsafe fn sum_i16_pairs_float(xh: __m128i, xl: __m128i) -> __m256 {
    let ones = _mm_set1_epi16(1);
    let summed_pairsl = _mm_madd_epi16(ones, xl);
    let summed_pairsh = _mm_madd_epi16(ones, xh);
    let summed_pairs = _mm256_set_m128i(summed_pairsh, summed_pairsl);
    _mm256_cvtepi32_ps(summed_pairs)
}

/// horizontally add 8 floats
#[inline]
pub unsafe fn hsum_float_8(x: __m256) -> f32 {
    let res = _mm256_extractf128_ps(x, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
#[inline]
pub unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const _);
    let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}
