#![allow(clippy::missing_safety_doc)]
use core::arch::aarch64::float32x4_t;
use core::arch::aarch64::uint16x4_t;
use core::arch::aarch64::uint16x8_t;
use core::arch::asm;
use core::mem::MaybeUninit;
use core::ptr;

use half::f16;

/// reference: https://github.com/starkat99/half-rs/pull/98/files

#[allow(non_camel_case_types)]
type float16x8_t = uint16x8_t;
#[allow(non_camel_case_types)]
type float16x4_t = uint16x4_t;

/// Convert to higher precision
/// Takes the 64 bits and convert them as [`float32x4_t`]
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FCVTL--FCVTL2--vector-)
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vcvt_f16_f32(i: float16x4_t) -> float32x4_t {
    let result: float32x4_t;
    asm!(
        "fcvtl {0:v}.4s, {1:v}.4h",
        out(vreg) result,
        in(vreg) i,
        options(pure, nomem, nostack, preserves_flags));
    result
}

/// Convert to higher precision
/// Takes the top 64 bits and convert them as [`float32x4_t`]
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FCVTL--FCVTL2--vector-)
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vget_high_f16_f32(i: float16x8_t) -> float32x4_t {
    let result: float32x4_t;
    asm!(
        "fcvtl2 {0:v}.4s, {1:v}.8h",
        out(vreg) result,
        in(vreg) i,
        options(pure, nomem, nostack, preserves_flags));
    result
}

/// Convert to higher precision
/// Takes the lower 64 bits and convert them as [`float32x4_t`]
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FCVTL--FCVTL2--vector-)
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vget_low_f16_f32(i: float16x8_t) -> float32x4_t {
    let result: float32x4_t;
    asm!(
        "fcvtl {0:v}.4s, {1:v}.4h",
        out(vreg) result,
        in(vreg) i,
        options(pure, nomem, nostack, preserves_flags));
    result
}

/// Floating point addition
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FADD--vector-)
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vaddq_f16(a: float16x8_t, b: float16x8_t) -> float16x8_t {
    let result: float16x8_t;
    asm!(
        "fadd {0:v}.8h, {1:v}.8h, {2:v}.8h",
        out(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack, preserves_flags));
    result
}

/// Floating point multiplication
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FADD--vector-)
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vmulq_f16(a: float16x8_t, b: float16x8_t) -> float16x8_t {
    let result: float16x8_t;
    asm!(
        "fmul {0:v}.8h, {1:v}.8h, {2:v}.8h",
        out(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack, preserves_flags));
    result
}

/// Floating point multiplication
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FADD--vector-)
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vget_lane_f16<const LANE: i32>(a: float16x8_t) -> u16 {
    todo!("lane!");
    // let result: u16;
    // match LANE {
    //    0=> asm!(
    //         "dup {0:h}, {1:v}.8h[0]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(pure, nomem, nostack, preserves_flags)),
    //    1=> asm!(
    //         "dup {0:v}, {1:v}.8h[1]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(nomem, nostack, preserves_flags)),
    //    2=> asm!(
    //         "dup {0:v}, {1:v}.8h[2]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(nomem, nostack, preserves_flags)),
    //    3=> asm!(
    //         "dup {0:v}, {1:v}.8h[3]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(nomem, nostack, preserves_flags)),
    //    4=> asm!(
    //         "dup {0:v}, {1:v}.8h[4]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(nomem, nostack, preserves_flags)),
    //    5=> asm!(
    //         "dup {0:v}, {1:v}.8h[5]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(nomem, nostack, preserves_flags)),
    //    6=> asm!(
    //         "dup {0:v}, {1:v}.8h[6]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(nomem, nostack, preserves_flags)),
    //    7=> asm!(
    //         "dup {0:v}, {1:v}.8h[7]",
    //         out(vreg) result,
    //         in(vreg) a,
    //         options(nomem, nostack, preserves_flags)),
    //     _ => unimplemented!("get_lane_f16 - {LANE}")
    // }
    // result
}

#[inline]
pub unsafe fn vfmaq_laneq_f16<const LANE: i32>(
    a: float16x8_t,
    b: float16x8_t,
    c: float16x8_t,
) -> float16x8_t {
    let c = vget_lane_f16::<LANE>(c);
    let result = core::mem::transmute([c, c, c, c, c, c, c, c]);
    vfmaq_f16(a, b, result)
}

/// Casts [`float16x8t`] to raw pointer.
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vst1q_f16(ptr: *mut f16, val: float16x8_t) {
    ptr::copy_nonoverlapping(&val, ptr.cast(), 8);
    // asm!(
    //     "vst1q_f16 {0:s}, {1:h}",
    //     out(vreg) ptr,
    //     in(vreg) val,
    //     options(pure, nomem, nostack, preserves_flags));
}

/// Casts pointer to [`float16x8t`].
/// This functions assumes pointer is aligned
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vld1q_f16(ptr: *const f16) -> float16x8_t {
    let mut result = MaybeUninit::<float16x8_t>::uninit();
    ptr::copy_nonoverlapping(ptr.cast(), &mut result, 8);
    // asm!(
    //     "vld1q_f16 {0:s}, {1:h}",
    //     out(vreg) result,
    //     in(vreg) ptr,
    //     options(pure, nomem, nostack, preserves_flags));
    result.assume_init()
}

/// Broadcast value into [`float16x8_t`]
/// Fused multiply add [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FMLA--vector-)
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vfmaq_f16(mut a: float16x8_t, b: float16x8_t, c: float16x8_t) -> float16x8_t {
    asm!(
        "fmla {0:v}.8h, {1:v}.8h, {2:v}.8h",
        inout(vreg) a,
        in(vreg) b,
        in(vreg) c,
        options(nomem, nostack, preserves_flags));
    a
}

/// Broadcast value into [`float16x8_t`]
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn vdupq_n_f16(a: u16) -> float16x8_t {
    let result: float16x8_t;
    asm!(
        "dup {0:v}.8h, {1:h}",
        out(vreg) result,
        in(vreg) a,
        options(pure, nomem, nostack, preserves_flags));
    result
}
