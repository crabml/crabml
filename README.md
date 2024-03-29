# crabml

[![](https://img.shields.io/discord/1111711408875393035?logo=discord&label=discord)](https://discord.gg/wbzqddT3QC)

`crabml` is a llama.cpp-compatible AI inference engine written in Rust, which runs everywhere with the help of WebGPU.

## Project Goals

`crabml` is designed with the following objectives in mind:

- Focus on inference only.
- Runs everywhere using WebGPU.
- Accelerated inferencing on cheap hardwares.
- `mmap()` from day one.

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
