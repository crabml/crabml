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

#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8], d: &mut u8, m: &mut u8) {
    if j < 4 {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

pub fn make_qx_quants(n: usize, nmax: i32, data: &[f32], ls: &mut [i8], rmse_type: i32) -> f32 {
    let mut max = 0.0;
    let mut abs_max = 0.0;
    for &x in data.iter().take(n) {
        let abs_x = x.abs();
        if abs_x > abs_max {
            abs_max = abs_x;
            max = x;
        }
    }
    if abs_max == 0. {
        // all zero
        for l in ls.iter_mut().take(n) {
            *l = 0;
        }
        return 0.;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for (i, &xi) in data.iter().take(n).enumerate() {
            let l = nearest_i32(iscale * xi);
            ls[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let weight_type = rmse_type % 2;
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for (i, &xi) in data.iter().take(n).enumerate() {
        let l = nearest_i32(iscale * xi);
        let l = l.clamp(-nmax, nmax - 1);
        ls[i] = (l + nmax) as i8;
        let w = if weight_type == 1 { xi * xi } else { 1.0 };
        let l = l as f32;
        sumlx += w * xi * l;
        suml2 += w * l * l;
    }
    let mut scale = sumlx / suml2;
    let mut best = scale * sumlx;
    for _itry in 0..3 {
        let iscale = 1.0 / scale;
        let mut slx = 0f32;
        let mut sl2 = 0f32;
        let mut changed = false;
        for (i, &xi) in data.iter().take(n).enumerate() {
            let l = nearest_i32(iscale * xi);
            let l = l.clamp(-nmax, nmax - 1);
            if l + nmax != ls[i] as i32 {
                changed = true;
            }
            let w = if weight_type == 1 { xi * xi } else { 1f32 };
            let l = l as f32;
            slx += w * xi * l;
            sl2 += w * l * l;
        }
        if !changed || sl2 == 0.0 || slx * slx <= best * sl2 {
            break;
        }
        for (i, &xi) in data.iter().take(n).enumerate() {
            let l = nearest_i32(iscale * xi);
            ls[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        sumlx = slx;
        suml2 = sl2;
        scale = sumlx / suml2;
        best = scale * sumlx;
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for (i, &xi) in data.iter().take(n).enumerate() {
            let w = if weight_type == 1 { xi * xi } else { 1. };
            let l = ls[i] as i32 - nmax;
            let mut slx = sumlx - w * xi * l as f32;
            if slx > 0. {
                let mut sl2 = suml2 - w * l as f32 * l as f32;
                let new_l = nearest_i32(xi * sl2 / slx);
                let new_l = new_l.clamp(-nmax, nmax - 1);
                if new_l != l {
                    slx += w * xi * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0. && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        ls[i] = (nmax + new_l) as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        scale = sumlx / suml2;
                        best = scale * sumlx;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if rmse_type < 3 {
        return scale;
    }
    for is in -4..4 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1f32 * is as f32) / max;
        let mut sumlx = 0.;
        let mut suml2 = 0.;
        for &xi in data.iter().take(n) {
            let l = nearest_i32(iscale * xi);
            let l = l.clamp(-nmax, nmax - 1);
            let w = if weight_type == 1 { xi * xi } else { 1. };
            let l = l as f32;
            sumlx += w * xi * l;
            suml2 += w * l * l;
        }
        if suml2 > 0. && sumlx * sumlx > best * suml2 {
            for (i, &xi) in data.iter().take(n).enumerate() {
                let l = nearest_i32(iscale * xi);
                ls[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

pub fn make_qkx1_quants(
    n: usize,
    nmax: i32,
    data: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    ntry: i32,
) -> f32 {
    let mut min = data[0];
    let mut max = data[0];
    for &d in data.iter().take(n) {
        if d < min {
            min = d;
        }
        if d > max {
            max = d;
        }
    }

    if max == min {
        for _l in l.iter_mut().take(n) {
            *_l = 0;
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
            let _l = nearest_i32(iscale * (data[i] - min));
            let _l = 0.max(nmax.min(_l));
            if _l as u8 != l[i] {
                l[i] = _l as u8;
                did_change = true;
            }
            sumlx += (data[i] - min) * _l as f32;
            suml2 += _l * _l;
        }
        scale = sumlx / suml2 as f32;
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += data[i] - scale * l[i] as f32;
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

pub fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    data: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    l_aux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = data[0];
    let mut max = data[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * data[0];

    for (&d, &w) in data[1..n].iter().zip(weights[1..n].iter()) {
        if d < min {
            min = d;
        }
        if d > max {
            max = d
        }
        sum_w += w;
        sum_x += w * d
    }
    if min > 0f32 {
        min = 0f32;
    }
    if max == min {
        for _l in l.iter_mut() {
            *_l = 0;
        }
        *the_min = -min;
        return 0f32;
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1f32 / iscale;
    let mut best_mad = 0f32;

    for (&d, (_l, &w)) in data.iter().zip(l.iter_mut().zip(weights.iter())).take(n) {
        let l = nearest_i32(iscale * (d - min));
        *_l = (l.min(nmax) as u8).max(0u8);
        let mut diff = scale * *_l as f32 + min - d;
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_mad += w * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0f32;
        let mut sum_l2 = 0f32;
        let mut sum_xl = 0f32;
        for (&d, (l_aux, &w)) in data
            .iter()
            .zip(l_aux.iter_mut().zip(weights.iter()))
            .take(n)
        {
            let mut l = nearest_i32(iscale * (d - min));
            l = l.min(nmax).max(0);
            *l_aux = l as u8;
            let l = l as f32;
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * d;
        }
        let d = sum_w * sum_xl - sum_l * sum_l;
        if d > 0f32 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if (this_min > 0f32) {
                this_min = 0f32;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0f32;
            for (&d, (&l_aux, &w)) in data.iter().zip(l_aux.iter().zip(weights.iter())).take(n) {
                let mut diff = this_scale * l_aux as f32 + this_min - d;
                diff = if use_mad { diff.abs() } else { diff * diff };
                mad += w * diff;
            }
            if mad < best_mad {
                for (l, &l_aux) in l.iter_mut().zip(l_aux.iter()).take(n) {
                    *l = l_aux;
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

pub fn make_q3_quants(n: usize, nmax: i32, data: &[f32], l: &mut [i8], do_rmse: bool) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for &d in data.iter().take(n) {
        let ax = d.abs();
        if ax > amax {
            amax = ax;
            max = d;
        }
    }
    // all zero
    if amax == 0f32 {
        for l in l.iter_mut().take(n) {
            *l = 0;
        }
        return 0f32;
    }
    let iscale = -nmax as f32 / max;
    if do_rmse {
        let mut sumlx = 0f32;
        let mut suml2 = 0f32;
        for (&d, l) in data.iter().zip(l.iter_mut()).take(n) {
            let mut _l = nearest_i32(iscale * d);
            _l = _l.min(nmax - 1).max(-nmax);
            *l = _l as i8;
            let w = d * d;
            sumlx += w * d * _l as f32;
            suml2 += w * (_l * _l) as f32;
        }
        // try at most 5 times
        for _ in 0..5 {
            let mut n_changed = 0;
            for (&d, l) in data.iter().zip(l.iter_mut()).take(n) {
                let w = d * d;
                let mut slx = sumlx - w * d * *l as f32;
                if slx > 0f32 {
                    let mut sl2 = suml2 - w * (*l as f32) * (*l as f32);
                    let mut new_l = nearest_i32(d * sl2 / slx);
                    new_l = new_l.min(nmax - 1).max(-nmax);
                    if new_l != *l as i32 {
                        slx += w * d * new_l as f32;
                        sl2 += w * (new_l * new_l) as f32;
                        if sl2 > 0f32 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            *l = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for l in l.iter_mut().take(n) {
            *l += nmax as i8;
        }
        return sumlx / suml2;
    }
    for (&d, l) in data.iter().zip(l.iter_mut()).take(n) {
        let mut _l = nearest_i32(iscale * d);
        _l = _l.min(nmax - 1).max(-nmax);
        *l = (_l + nmax) as i8;
    }
    1f32 / iscale
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
        for (&s1_d, &s2_d) in s1.iter().zip(s2.iter()) {
            let diff = s1_d - s2_d;
            sum += diff * diff;
        }
        f32::sqrt(sum) / n as f32
    }

    /// Calculate the dot product of original inputs for test reference
    pub fn dot_product(s1: &[f32], s2: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (f1, f2) in s1.iter().zip(s2.iter()) {
            sum += *f1 * *f2;
        }
        sum
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

    #[test]
    fn test_get_scale_min_k4() {
        let data = [255, 255, 255, 255, 255];
        let mut sc: u8 = u8::default();
        let mut m: u8 = u8::default();
        get_scale_min_k4(0, &data, &mut sc, &mut m);
        assert_eq!(sc, 63);
        assert_eq!(m, 63);
    }
}
