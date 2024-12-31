#![allow(clippy::missing_safety_doc)]
use core::arch::aarch64::float32x4_t;
use core::arch::aarch64::uint16x4_t;
use core::arch::aarch64::uint16x8_t;
use core::arch::asm;
use std::arch::aarch64::int32x4_t;
use std::arch::aarch64::int8x16_t;

use half::f16;

/// reference: https://github.com/starkat99/half-rs/pull/98/files

#[allow(non_camel_case_types)]
type float16x8_t = uint16x8_t;
#[allow(non_camel_case_types)]
type float16x4_t = uint16x4_t;

/// Convert to higher precision
/// Takes the 64 bits and convert them as [`float32x4_t`]
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FCVTL--FCVTL2--vector-)
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

/// Convert to lower precision
#[inline]
pub unsafe fn vcvt_f32_f16(i: float32x4_t) -> float16x4_t {
    let result: float16x4_t;
    asm!(
        "fcvtn {0:v}.4h, {1:v}.4s",
        out(vreg) result,
        in(vreg) i,
        options(pure, nomem, nostack, preserves_flags));
    result
}

/// Convert to higher precision
/// Takes the top 64 bits and convert them as [`float32x4_t`]
/// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FCVTL--FCVTL2--vector-)
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

/// Casts pointer to [`float16x8t`].
/// This functions assumes pointer is aligned
#[inline]
pub unsafe fn vld1q_f16(ptr: *const f16) -> float16x8_t {
    core::arch::aarch64::vld1q_u16(ptr as *const u16) as float16x8_t
}

#[inline]
pub unsafe fn vst1q_f16(ptr: *mut f16, a: float16x8_t) {
    core::arch::aarch64::vst1q_u16(ptr as *mut u16, a as uint16x8_t);
}

/// Broadcast value into [`float16x8_t`]
/// Fused multiply add [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FMLA--vector-)
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
#[inline]
pub unsafe fn vdupq_n_f16(a: u16) -> float16x8_t {
    core::arch::aarch64::vdupq_n_u16(a) as float16x8_t
}

#[inline]
pub unsafe fn vaddvq_f16(a: float16x8_t) -> f32 {
    use core::arch::aarch64;
    let vhigh = vget_high_f16_f32(a);
    let vlow = vget_low_f16_f32(a);
    aarch64::vaddvq_f32(vhigh) + aarch64::vaddvq_f32(vlow)
}

/// calling this is much slower than using the intrinsics.
#[inline]
pub unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    use core::arch::aarch64::*;
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}
