# crabml

[![](https://img.shields.io/discord/1111711408875393035?logo=discord&label=discord)](https://discord.gg/wbzqddT3QC)

`crabml` is focusing on the reimplementation of GGML using the Rust programming language.

> The project is currently an active experiment with the capability to run inference on a Q8_0 quantized Llama 3B model. While the inference is currently not optimized for speed, `crabml` is a promising endeavor for efficient machine learning inferencing.

## Project Goals

`crabml` is designed with the following objectives in mind:

- Focus on inference only.
- Limit tensor operators to the bare minimum required for LLM inference.
- Achieve fast enough inferencing on cheap hardwares.
- `mmap()` from day one.
- Prioritize SIMD ahead of GPU.

## Usage

### Building the Project

To build `crabml`, set the `RUSTFLAGS` environment variable to enable specific target features. For example, to enable NEON on ARM architectures, use `RUSTFLAGS="-C target-feature=+neon"`. Then build the project with the following command:

```bash
cargo build --release
```

This command compiles the project in release mode, which optimizes the binary for performance.

### Running an Example

After building the project, you can run an example inference by executing the `crabml-cli` binary with appropriate arguments. For instance, to use the `tinyllamas-stories-15m-f32.gguf` model to generate text based on the prompt "captain america", execute the command below:

```bash
./target/release/crabml-cli \
  -m ./testdata/tinyllamas-stories-15m-f32.gguf \
  "captain america" --steps 100 \
  -t 0.8 -p 1.0
```

In this command:

- `-m` specifies the checkpoint file.
- `--steps` defines the number of tokens to generate.
- `-t` sets the temperature, which controls the randomness of the output.
- `-p` sets the probability of sampling from the top-p.

## License

This contribution is licensed under Apache License, Version 2.0, ([LICENSE](LICENSE) or <http://www.apache.org/licenses/LICENSE-2.0>)
