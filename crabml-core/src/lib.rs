#![feature(portable_simd)]
#![feature(slice_as_chunks)]
#![feature(stdsimd)]
#![feature(thread_local)]
#![feature(lazy_cell)]

#[allow(unreachable_patterns)]
pub mod backends;
pub mod error;
pub mod gguf;
pub mod tensor;
pub mod tokenizer;
