# crabml

[![](https://img.shields.io/discord/1111711408875393035?logo=discord&label=discord)](https://discord.gg/wbzqddT3QC)

crabml is an ongoing experiment that aims to reimplement GGML using Rust.

Currently it can inference a 3B Q8_0 quantized Llama model at a dog slow speed.

Its design goals are:

- focus on inference only.
- limit tensor operators to the bare minimum required for LLM inference.
- fast enough inferencing on cheap hardwares.
- `mmap()` from day one.
- prioritize SIMD ahead of GPU.

## Build

```
RUSTFLAGS="-C target-feature=+neon" cargo build --release
./target/release/crabml-cli -m ./testdata/open-llama-3b-q8_0.gguf "captain america" --steps 100 -t 0.8 -p 1.0
```