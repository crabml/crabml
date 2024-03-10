extern crate jemallocator;

use std::io::Write;
use std::time::Instant;

use clap::Parser;
use crabml::backends::cpu::CpuTensorDevice;
use crabml::backends::cpu::CpuTensorDeviceOptions;
use crabml::error::Result;
use crabml::gguf::GGUFFileLoader;
use crabml::tensor::TensorMetrics;
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::sampler::Llama2Sampler;
use crabml_llama2::CpuLlama2Model;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

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

    #[arg(short = 'T', long, default_value_t = 2)]
    threads: usize,

    /// The prompt
    prompt: String,
}

fn main() -> Result<()> {
    let args = CommandArgs::parse();
    let start_time = Instant::now();

    // configure rayon
    let mut threads = args.threads;
    if threads == 0 {
        threads = num_cpus::get();
    }

    let gl = GGUFFileLoader::new(&args.model)?;
    let gf = gl.open()?;

    let metrics = TensorMetrics::default();
    let device_cpu = CpuTensorDevice::new(CpuTensorDeviceOptions {
        thread_pool_size: Some(threads),
        metrics: Some(metrics.clone()),
        ..Default::default()
    });

    let model_cpu = CpuLlama2Model::load(&gf, device_cpu.clone())?;
    let conf = model_cpu.conf.clone();

    // let device_wgpu = WgpuTensorDevice::new(
    //     WgpuTensorDeviceOptions::new().with_staging_buf_bytes(conf.vocab_size * 4),
    // );
    // let model_wgpu = WgpuLlama2Model::from_cpu(&model_cpu, device_wgpu)?;

    let mut sampler = Llama2Sampler::new(
        conf.vocab_size,
        args.temperature,
        args.probability,
        device_cpu.exp_cache(),
    );
    let mut runner = Llama2Runner::new(&model_cpu, metrics.clone(), true)?;

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

    loop {
        let _t = metrics.total_walltime.track();
        match output.next() {
            Some(token) => {
                print!("{}", token?);
                std::io::stdout().flush().unwrap();
            }
            None => {
                break;
            }
        }

        if args.verbose {
            for (k, v) in metrics.as_vec().iter() {
                println!("{}: {}", k, v);
            }
        }
        metrics.reset();
    }

    println!();
    println!(
        "{} tokens/s, {} threads",
        output.average_tokens_per_seconds(),
        threads
    );

    Ok(())
}
