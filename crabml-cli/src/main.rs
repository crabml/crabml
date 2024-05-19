extern crate jemallocator;

use std::io::Write;
use std::time::Instant;

use clap::Parser;
use clap::ValueEnum;
use crabml::error::Result;
use crabml::gguf::GGUFFile;
use crabml::gguf::GGUFFileLoader;
use crabml::gguf::GGUFMetadataValueType;
use crabml::tensor::Tensor;
use crabml::tensor::TensorMetrics;
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::model::CpuLlamaModelLoader;
use crabml_llama2::GpuLlamaModel;
use crabml_llama2::Llama2Chat;
use crabml_wgpu::WgpuTensor;
use crabml_wgpu::WgpuTensorDevice;
use crabml_wgpu::WgpuTensorDeviceOptions;
use rustyline::error::ReadlineError;
use rustyline::Editor;

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

    #[arg(short, long, default_value_t = false)]
    chat: bool,

    /// mlock the mmaped file, it can help run faster without swapping
    #[arg(long, default_value_t = false)]
    mlock: bool,

    /// The prompt, if it's in chat mode, it will play as the system prompt
    prompt: Option<String>,

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

fn run<T: Tensor>(runner: &mut Llama2Runner<T>, args: &CommandArgs) -> Result<()> {
    if args.chat {
        run_chat(runner, args)?;
    } else {
        run_generate(runner, args)?;
    }

    Ok(())
}

fn run_chat<T: Tensor>(runner: &mut Llama2Runner<T>, args: &CommandArgs) -> Result<()> {
    let mut system_prompt = args.prompt.clone();
    let mut rl = Editor::<()>::new();
    loop {
        let line = match rl.readline(">> ") {
            Ok(line) => {
                if line.is_empty() {
                    continue;
                } else if line == "quit" {
                    break;
                }
                line
            }
            Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("{:?}", err);
                break;
            }
        };

        let mut chat = Llama2Chat::new(runner, &line, system_prompt.clone())?;

        // only put system prompt in the first round
        if system_prompt.is_some() {
            system_prompt = None;
        }

        // TODO: handle the user input while generating
        let reply_iter = chat.reply()?;
        for token in reply_iter {
            print!("{}", token?);
            std::io::stdout().flush().unwrap();
        }
        chat.finish()?;
        println!();
    }

    Ok(())
}

fn run_generate<U: Tensor>(runner: &mut Llama2Runner<U>, args: &CommandArgs) -> Result<()> {
    let metrics = runner.metrics.clone();
    let prefill_started_at = Instant::now();
    let prompt = args.prompt.clone().unwrap_or("".to_string());
    let (prefill_pos, _prev_token, token) = runner.prefill(&prompt, true, false)?;
    let prefill_elapsed = prefill_started_at.elapsed();
    if args.verbose {
        dump_metrics(&runner.metrics);
    }
    runner.metrics.reset();

    let mut output = runner.generate(prefill_pos, token, Some(args.steps));
    let mut generated_tokens = 0;
    let generation_started_at = Instant::now();

    print!("{}", &prompt);
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
            dump_metrics(&metrics);
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
        println!("{0: <40} | {1: <4}", k, v);
    }
    println!(
        "{0: <40} | {1: <4}",
        "non_matmul",
        metrics.forward_walltime.as_millis()
            - metrics.matmul_walltime.as_millis()
            - metrics.batch_matmul_walltime.as_millis()
    );
}

fn dump_gguf_metadata(gf: &GGUFFile) {
    for (key, value) in gf.metadata().as_hashmap() {
        if value.typ() != GGUFMetadataValueType::Array {
            eprintln!("{}: {:?}", key, value);
        }
    }
    for tensor in gf.tensor_infos() {
        eprintln!(
            "- {} \t\t\t {} \t {:?}",
            tensor.name(),
            tensor.typ(),
            tensor.dimensions()
        );
    }
}

fn main() -> Result<()> {
    let args = CommandArgs::parse();
    let start_time = Instant::now();

    let mut thread_num = args.threads;
    if thread_num == 0 {
        thread_num = num_cpus::get();
    }

    // it may takes a while to open the file if mlock is enabled
    eprintln!("loading model...");
    let gl = GGUFFileLoader::new(&args.model, args.mlock)?;
    let gf = gl.open()?;

    if args.verbose {
        dump_gguf_metadata(&gf);
    }

    let model_cpu = CpuLlamaModelLoader::new()
        .with_thread_num(thread_num)
        .with_temperature(args.temperature)
        .with_probability(args.probability)
        .load(&gf)?;
    let conf = model_cpu.conf.clone();

    match args.device {
        DeviceType::Cpu => {
            let mut runner = Llama2Runner::new(&model_cpu, conf.seq_len, true)?;
            eprintln!("model loaded: {}ms", start_time.elapsed().as_millis());
            run(&mut runner, &args)?;
        }
        DeviceType::Wgpu => {
            let device_wgpu = WgpuTensorDevice::new(
                WgpuTensorDeviceOptions::new().with_staging_buf_bytes(conf.vocab_size * 4),
            );
            let model_wgpu = GpuLlamaModel::<WgpuTensor>::from_cpu(&model_cpu, device_wgpu)?;

            let mut runner = Llama2Runner::new(&model_wgpu, conf.seq_len, false)?;
            run(&mut runner, &args)?;
        }
    }

    Ok(())
}
