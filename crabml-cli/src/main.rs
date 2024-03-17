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
    // it seems printing to stdout is not that fast, so we use a separate thread to keep the generation not blocked by the printing
    let (tx, rx) = std::sync::mpsc::sync_channel(32);
    let print_thread = std::thread::spawn(move || {
        let mut step = 0;
        while let Ok(Some(token)) = rx.recv() {
            print!("{}", token);
            step += 1;
            if step % 2 == 0 {
                std::io::stdout().flush().unwrap();
            }
        }
        std::io::stdout().flush().unwrap();
    });

    let mut output = runner.generate(&args.prompt, args.steps, sampler)?;
    print!("{}", &args.prompt);

    loop {
        let _t = metrics.total_walltime.track();
        match output.next() {
            Some(token) => tx.send(Some(token?)).unwrap(),
            None => {
                tx.send(None).unwrap();
                break;
            }
        }

        if args.verbose {
            println!();
            let mut metric_values = metrics.as_vec();
            metric_values.sort_by_key(|v| (v.1 * 1000.0) as u32);
            for (k, v) in metric_values.iter() {
                println!("{0: <40} | {1: <10}", k, v);
            }
        }
        metrics.reset();
    }

    print_thread.join().unwrap();
    println!();
    println!(
        "{} tokens/s, {} threads",
        output.average_tokens_per_seconds(),
        args.threads
    );

    Ok(())
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
        println!("loaded model: {}ms", start_time.elapsed().as_millis());
    }

    match args.device {
        DeviceType::Cpu => {
            let mut runner = Llama2Runner::new(&model_cpu, metrics.clone(), true)?;
            run(&args, &mut runner, &mut sampler, &metrics)?;
        }
        DeviceType::Wgpu => {
            let device_wgpu = WgpuTensorDevice::new(
                WgpuTensorDeviceOptions::new().with_staging_buf_bytes(conf.vocab_size * 4),
            );
            let model_wgpu = WgpuLlama2Model::from_cpu(&model_cpu, device_wgpu)?;

            let mut runner = Llama2Runner::new(&model_wgpu, metrics.clone(), false)?;
            run(&args, &mut runner, &mut sampler, &metrics)?;
        }
    }

    Ok(())
}
