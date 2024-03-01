pub mod api;
pub use api::CpuTensorBuf;

pub mod buf_f32;

pub mod buf_q8_0;

use buf_q8_0::BlockQ8_0;
pub use buf_q8_0::QuantBufQ8_0;

#[cfg(all(target_arch = "aaarch64", target_feature = "neon"))]
mod buf_q8_0_aarch64_neon;
#[cfg(all(target_arch = "aaarch64", target_feature = "neon"))]
use buf_q8_0_aarch64_neon::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod buf_q8_0_x86_64_avx2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use buf_q8_0_x86_64_avx2::*;

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
mod buf_q8_0_fallback;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
use buf_q8_0_fallback::*;
