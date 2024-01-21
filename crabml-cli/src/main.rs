use std::io::Write;
use std::time::Instant;

use clap::Parser;
use crabml::backends::cpu::CpuTensorDevice;
use crabml::backends::wgpu::WgpuTensorDevice;
use crabml::backends::wgpu::WgpuTensorDeviceOptions;
use crabml::error::Result;
use crabml::gguf::GGUFFileLoader;
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::model::WgpuLlama2Model;
use crabml_llama2::sampler::Llama2Sampler;
use crabml_llama2::CpuLlama2Model;

#[derive(Parser, Debug)]
struct CommandArgs {
    /// The checkpoint file to load
    #[arg(short, long, default_value_t = format!("./testdata/tinyllamas-stories-15m-f32.gguf"))]
    model: String,

    // The number of tokens to generate
    #[arg(short, long, default_value_t = 300)]
    steps: usize,

    // The probability of sampling from the top-p.
    #[arg(short, long, default_value_t = 0.9)]
    probability: f32,

    #[arg(short, long, default_value_t = 1.0)]
    temperature: f32,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// The prompt
    prompt: String,
}

fn main() -> Result<()> {
    let args = CommandArgs::parse();
    let start_time = Instant::now();

    // configure rayon
    let threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let gl = GGUFFileLoader::new(&args.model)?;
    let gf = gl.open()?;

    let device_cpu = CpuTensorDevice::new();
    let model_cpu = CpuLlama2Model::load(&gf, device_cpu)?;
    let conf = model_cpu.conf();

    let device_wgpu = WgpuTensorDevice::new(
        WgpuTensorDeviceOptions::new().with_staging_buf_bytes(conf.vocab_size * 4),
    );
    // let model_wgpu = WgpuLlama2Model::from_cpu(&model_cpu, device_wgpu)?;

    let mut sampler = Llama2Sampler::new(conf.vocab_size, args.temperature, args.probability);
    let mut runner = Llama2Runner::try_from(&model_cpu)?;

    if args.verbose {
        for tensor in gf.tensor_infos() {
            println!(
                "- {} \t\t\t {} \t {:?}",
                tensor.name(),
                tensor.typ(),
                tensor.dimensions()
            );
        }
        println!("loaded model: {}ms", start_time.elapsed().as_millis());
    }

    let mut output = runner.generate(&args.prompt, args.steps, &mut sampler)?;
    print!("{}", &args.prompt);
    for token in output.by_ref() {
        print!("{}", token?);
        std::io::stdout().flush().unwrap();
    }
    println!();
    println!(
        "{} tokens/s, {} threads",
        output.average_tokens_per_seconds(),
        threads
    );

    Ok(())
}
