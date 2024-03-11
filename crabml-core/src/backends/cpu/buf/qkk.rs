//! The common mod for K-Quants
//!
//! Including shared constants and functions

/// Super-block size for Quants-K.
///
/// `QK_K` elements in a super block
pub const QK_K: usize = 256;

pub fn nearest_i32(fval: f32) -> i32 {
    assert!(fval <= 4194303.0f32);
    let mut val = (fval + 12582912.0f32) as i32;
    let mut i = 0;
    std::mem::swap(&mut i, &mut val);
    (i & 0x007fffff) - 0x00400000
}

pub fn make_qkx1_quants(
    n: usize,
    nmax: i32,
    data: &[f32],
    L: &mut [u8],
    the_min: &mut f32,
    ntry: i32,
) -> f32 {
    let mut min = data[0];
    let mut max = data[0];
    for d in data.iter().take(n) {
        let d = *d;
        if d < min {
            min = d;
        }
        if d > max {
            max = d;
        }
    }

    if max == min {
        for l in L.iter_mut().take(n) {
            *l = 0;
        }
        *the_min = 0.0f32;
        return 0.0f32;
    }
    if min > 0.0f32 {
        min = 0.0f32;
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0f32 / iscale;
    for _ in 0..ntry {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0i32;
        let mut did_change = false;
        for i in 0..n {
            let l = nearest_i32(iscale * (data[i] - min));
            let l = 0.max(nmax.min(l));
            if l as u8 != L[i] {
                L[i] = l as u8;
                did_change = true;
            }
            sumlx += (data[i] - min) * l as f32;
            suml2 += l * l;
        }
        scale = sumlx / suml2 as f32;
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += data[i] - scale * L[i] as f32;
        }
        min = sum / n as f32;
        if min > 0f32 {
            min = 0f32;
        }
        iscale = 1f32 / scale;
        if !did_change {
            break;
        }
    }
    *the_min = -min;
    scale
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Generate synthetic test data
    pub fn generate_data(offset: f32, n: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(0.1 + 2.0 * f32::cos(i as f32 + offset));
        }
        data
    }

    /// Calculate RMSE between two f32 slices as the difference
    pub fn array_rmse(s1: &[f32], s2: &[f32]) -> f32 {
        if s1.len() != s2.len() {
            panic!(
                "s1 and s2 should have the same length: s1 length {}, s2 length {}",
                s1.len(),
                s2.len()
            );
        }

        let n = s1.len();
        let mut sum = 0.0;
        for (s1_d, s2_d) in s1.iter().zip(s2.iter()) {
            let diff = *s1_d - *s2_d;
            sum += diff * diff;
        }
        f32::sqrt(sum) / n as f32
    }

    #[test]
    fn test_nearest_i32() {
        let data_against = [
            (3_256_291.8_f32, 3256292_i32),
            (234_730.28_f32, 234730_i32),
            (3_271_636.3_f32, 3271636_i32),
            (143_427.25_f32, 143427_i32),
            (624_284.7_f32, 624285_i32),
            (601459.0_f32, 601459_i32),
            (929_129.4_f32, 929129_i32),
            (196_503.23_f32, 196503_i32),
            (906_489.75_f32, 906490_i32),
            (1_711_053.4_f32, 1711053_i32),
        ];
        for (i, (input, expected)) in data_against.into_iter().enumerate() {
            let result = nearest_i32(input);
            assert_eq!(
                expected, result,
                "{}th result incorrect: expect {}_i32 for {}_f32, get {}",
                i, expected, input, result
            );
        }
    }
}
