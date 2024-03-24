extern crate jemallocator;

use std::io::Write;
use std::time::Instant;

use clap::Parser;
use clap::ValueEnum;
use crabml::backends::cpu::CpuTensorDevice;
use crabml::backends::wgpu::WgpuTensorDevice;
use crabml::backends::wgpu::WgpuTensorDeviceOptions;
use crabml::error::Result;
use crabml::gguf::GGUFFileLoader;
use crabml::tensor::Tensor;
use crabml::tensor::TensorMetrics;
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::sampler::Llama2Sampler;
use crabml_llama2::CpuLlama2Model;
use crabml_llama2::WgpuLlama2Model;

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

    #[arg(short = 'D', long, default_value_t = DeviceType::Cpu)]
    device: DeviceType,
}

#[derive(Clone, Debug, ValueEnum)]
enum DeviceType {
    Cpu,
    Wgpu,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Wgpu => write!(f, "wgpu"),
        }
    }
}

fn run<U: Tensor>(
    args: &CommandArgs,
    runner: &mut Llama2Runner<U>,
    sampler: &mut Llama2Sampler,
    metrics: &TensorMetrics,
) -> Result<()> {
    let prefill_started_at = Instant::now();
    let (prefill_pos, prev_token, token) = runner.prefill(&args.prompt, sampler)?;
    let prefill_elapsed = prefill_started_at.elapsed();
    if args.verbose {
        dump_metrics(metrics);
    }
    metrics.reset();

    let mut output = runner.generate(prefill_pos, prev_token, token, args.steps, sampler);
    let mut generated_tokens = 0;
    let generation_started_at = Instant::now();

    print!("{}", &args.prompt);
    loop {
        let _t = metrics.total_walltime.track();
        match output.next() {
            Some(token) => {
                generated_tokens += 1;
                print!("{}", token?);
                std::io::stdout().flush().unwrap();
            }
            None => {
                break;
            }
        }

        if args.verbose {
            dump_metrics(metrics);
        }
        metrics.reset();
    }

    let generation_elapsed = generation_started_at.elapsed().as_secs_f64();
    let generated_tokens_per_second = generated_tokens as f64 / generation_elapsed;

    println!();
    println!(
        "prompt: {} tokens, {}ms",
        prefill_pos,
        prefill_elapsed.as_millis()
    );
    println!(
        "{} tokens/s, {} threads",
        generated_tokens_per_second, args.threads
    );

    Ok(())
}

fn dump_metrics(metrics: &TensorMetrics) {
    println!();
    if metrics.forward_walltime.as_millis() == 0.0 {
        return;
    }
    let mut metric_values = metrics.as_vec();
    metric_values.sort_by_key(|v| (v.1 * 1000.0) as u32);
    for (k, v) in metric_values.iter() {
        println!("{0: <40} | {1: <10}", k, v);
    }
}

fn main() -> Result<()> {
    let args = CommandArgs::parse();
    let start_time = Instant::now();

    // configure rayon
    let mut threads = args.threads;
    if threads == 0 {
        threads = num_cpus::get();
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let gl = GGUFFileLoader::new(&args.model)?;
    let gf = gl.open()?;

    let metrics = TensorMetrics::default();
    let device_cpu = CpuTensorDevice::new().with_metrics(metrics.clone());
    let model_cpu = CpuLlama2Model::load(&gf, device_cpu.clone())?;
    let conf = model_cpu.conf.clone();

    let mut sampler = Llama2Sampler::new(
        conf.vocab_size,
        args.temperature,
        args.probability,
        device_cpu.exp_cache(),
    );

    if args.verbose {
        for tensor in gf.tensor_infos() {
            println!(
                "- {} \t\t\t {} \t {:?}",
                tensor.name(),
                tensor.typ(),
                tensor.dimensions()
            );
        }
    }

    match args.device {
        DeviceType::Cpu => {
            let mut runner = Llama2Runner::new(&model_cpu, metrics.clone(), conf.seq_len, true)?;
            println!("loaded model: {}ms", start_time.elapsed().as_millis());
            run(&args, &mut runner, &mut sampler, &metrics)?;
        }
        DeviceType::Wgpu => {
            let device_wgpu = WgpuTensorDevice::new(
                WgpuTensorDeviceOptions::new().with_staging_buf_bytes(conf.vocab_size * 4),
            );
            let model_wgpu = WgpuLlama2Model::from_cpu(&model_cpu, device_wgpu)?;

            let mut runner = Llama2Runner::new(&model_wgpu, metrics.clone(), conf.seq_len, false)?;
            run(&args, &mut runner, &mut sampler, &metrics)?;
        }
    }

    Ok(())
}
