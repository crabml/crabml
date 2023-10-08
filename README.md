# crabml

crabml is an ongoing experiment that aims to reimplement GGML using Rust. Its goals are:

- focus on inference only.
- limit tensor operators to the bare minimum required for LLM inference.
- allow inference on cheap hardware.
- prioritize SIMD ahead of GPU.
