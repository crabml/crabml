use std::arch::aarch64;

use half::f16;

use super::BlockQ8_0;

pub fn quantize_f32_q8_0(data: &[f32]) -> Vec<BlockQ8_0> {
    let mut bs = Vec::with_capacity(data.len() / 32);

    unsafe {
        for i in (0..data.len()).step_by(32) {
            let mut vsrc = [aarch64::vdupq_n_f32(0.0); 8];
            let mut vasrc = [aarch64::vdupq_n_f32(0.0); 8];
            let mut vmax = [aarch64::vdupq_n_f32(0.0); 8];

            for j in 0..8 {
                vsrc[j] = aarch64::vld1q_f32(data.as_ptr().add(i + j * 4));
                vasrc[j] = aarch64::vabsq_f32(vsrc[j]);
            }

            for j in 0..4 {
                vmax[2 * j] = aarch64::vmaxq_f32(vasrc[2 * j], vasrc[2 * j + 1]);
            }
            for j in 0..2 {
                vmax[4 * j] = aarch64::vmaxq_f32(vmax[4 * j], vmax[4 * j + 2]);
            }
            for j in 0..1 {
                vmax[8 * j] = aarch64::vmaxq_f32(vmax[8 * j], vmax[8 * j + 4]);
            }
            let max = aarch64::vmaxvq_f32(vmax[0]);

            let d = max / 127.0;
            let mut qs = [0_i8; 32];

            for j in 0..8 {
                let v = aarch64::vdivq_f32(vsrc[j], aarch64::vdupq_n_f32(d));
                let vi = aarch64::vcvtq_s32_f32(v);
                qs[4 * j] = aarch64::vgetq_lane_s32(vi, 0) as i8;
                qs[4 * j + 1] = aarch64::vgetq_lane_s32(vi, 1) as i8;
                qs[4 * j + 2] = aarch64::vgetq_lane_s32(vi, 2) as i8;
                qs[4 * j + 3] = aarch64::vgetq_lane_s32(vi, 3) as i8;
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
    assert!(abs.len() == bbs.len());

    if bbs.len() % 2 == 0 {
        return vec_dot_q8_0_q8_0_unrolled(abs, bbs);
    }
    vec_dot_q8_0_q8_0_rolled(abs, bbs)
}

fn vec_dot_q8_0_q8_0_rolled(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    unsafe {
        use std::arch::aarch64;
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let zerov = aarch64::vdupq_n_s32(0);

        for i in 0..bbs.len() {
            let ab0 = abs.get_unchecked(i);
            let bb0 = bbs.get_unchecked(i);

            let av00 = aarch64::vld1q_s8(ab0.qs.as_ptr());
            let av01 = aarch64::vld1q_s8(ab0.qs.as_ptr().add(16));

            let bv00 = aarch64::vld1q_s8(bb0.qs.as_ptr());
            let bv01 = aarch64::vld1q_s8(bb0.qs.as_ptr().add(16));

            sumv0 = aarch64::vmlaq_n_f32(
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av00, bv00),
                    aarch64::vdotq_s32(zerov, av01, bv01),
                )),
                f16::to_f32(ab0.d) * f16::to_f32(bb0.d),
            );
        }

        aarch64::vaddvq_f32(sumv0)
    }
}

fn vec_dot_q8_0_q8_0_unrolled(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    assert!(
        bbs.len() % 2 == 0,
        "bbs.len() must be a multiple of 64, got: {}",
        bbs.len()
    );

    unsafe {
        use std::arch::aarch64;
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        let zerov = aarch64::vdupq_n_s32(0);

        for i in (0..bbs.len()).step_by(2) {
            let ab0 = abs.get_unchecked(i);
            let ab1 = abs.get_unchecked(i + 1);
            let bb0 = bbs.get_unchecked(i);
            let bb1 = bbs.get_unchecked(i + 1);

            let av00 = aarch64::vld1q_s8(ab0.qs.as_ptr());
            let av01 = aarch64::vld1q_s8(ab0.qs.as_ptr().add(16));
            let av10 = aarch64::vld1q_s8(ab1.qs.as_ptr());
            let av11 = aarch64::vld1q_s8(ab1.qs.as_ptr().add(16));

            let bv00 = aarch64::vld1q_s8(bb0.qs.as_ptr());
            let bv01 = aarch64::vld1q_s8(bb0.qs.as_ptr().add(16));
            let bv10 = aarch64::vld1q_s8(bb1.qs.as_ptr());
            let bv11 = aarch64::vld1q_s8(bb1.qs.as_ptr().add(16));

            sumv0 = aarch64::vmlaq_n_f32(
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av00, bv00),
                    aarch64::vdotq_s32(zerov, av01, bv01),
                )),
                f16::to_f32(ab0.d) * f16::to_f32(bb0.d),
            );

            sumv1 = aarch64::vmlaq_n_f32(
                sumv1,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av10, bv10),
                    aarch64::vdotq_s32(zerov, av11, bv11),
                )),
                f16::to_f32(ab1.d) * f16::to_f32(bb1.d),
            );
        }

        aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
    }
}
