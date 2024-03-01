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

/// TODO: Use AVX2 instead.
pub fn vec_dot_q8_0_q8_0(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    let mut sumf: f32 = 0.0;
    for i in 0..bbs.len() {
        let mut sumi: i32 = 0;
        for j in 0..32 {
            sumi += (abs[i].qs[j] as i32) * (bbs[i].qs[j] as i32);
        }
        sumf += sumi as f32 * abs[i].d.to_f32() * bbs[i].d.to_f32();
    }

    sumf
}
