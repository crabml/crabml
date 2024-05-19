#![feature(portable_simd)]
#![feature(slice_as_chunks)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]
#![feature(iter_array_chunks)]
#![feature(lint_reasons)]
#![allow(clippy::map_entry)]
#![allow(clippy::comparison_chain)]

pub mod cpu;
pub mod error;
pub mod gguf;
pub mod tensor;
pub mod tokenizer;
