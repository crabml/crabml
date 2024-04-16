#![feature(portable_simd)]
#![feature(slice_as_chunks)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]
#![feature(thread_local)]
#![feature(lazy_cell)]
#![feature(iter_array_chunks)]
#![feature(lint_reasons)]
#![allow(clippy::map_entry)]
#![allow(clippy::comparison_chain)]

#[allow(unreachable_patterns)]
pub mod backends;
pub mod error;
pub mod gguf;
pub mod tensor;
pub mod tokenizer;
