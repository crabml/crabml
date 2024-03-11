#![feature(portable_simd)]
#![feature(slice_as_chunks)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]
#![feature(lazy_cell)]

#[allow(unreachable_patterns)]
pub mod backends;
pub mod error;
pub mod gguf;
pub mod tensor;
pub mod tokenizer;
