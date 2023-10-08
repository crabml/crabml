# crabml

crabml is an ongoing experiment that aims to reimplement GGML using Rust.

Currently it can inference a 3B Q8_0 quantized Llama model at a dog slow speed.

Its goals are:

- focus on inference only.
- limit tensor operators to the bare minimum required for LLM inference.
- allow inference on cheap hardware.
- prioritize SIMD ahead of GPU.
