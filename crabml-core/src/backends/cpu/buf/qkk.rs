pub fn nearest_i32(fval: f32) -> i32 {
    fval.round() as i32
}

pub fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    ls: &mut [i8],
    rmse_type: i32,
) -> f32 {
    let mut max = 0.0;
    let mut amax = 0.0;
    for &x in x.iter() {
        let ax = x.abs();
        if ax > amax {
            amax = ax;
            max = x;
        }
    }
    if amax == 0. {
        // all zero
        for l in ls.iter_mut().take(n) {
            *l = 0;
        }
        return 0.;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for (i, &xi) in x.iter().enumerate() {
            let l = nearest_i32(iscale * xi);
            ls[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let weight_type = rmse_type % 2;
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for (i, &xi) in x.iter().enumerate() {
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
        for (i, &xi) in x.iter().enumerate() {
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
        for (i, &xi) in x.iter().enumerate() {
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
        for (i, &xi) in x.iter().enumerate() {
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
        for &xi in x.iter() {
            let l = nearest_i32(iscale * xi);
            let l = l.clamp(-nmax, nmax - 1);
            let w = if weight_type == 1 { xi * xi } else { 1. };
            let l = l as f32;
            sumlx += w * xi * l;
            suml2 += w * l * l;
        }
        if suml2 > 0. && sumlx * sumlx > best * suml2 {
            for (i, &xi) in x.iter().enumerate() {
                let l = nearest_i32(iscale * xi);
                ls[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}
