#![feature(portable_simd)]
#![feature(slice_as_chunks)]
#![feature(stdsimd)]

pub mod backends;
pub mod error;
pub mod gguf;
pub mod tensor;
pub mod tokenizer;
