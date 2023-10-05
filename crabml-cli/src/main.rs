use clap::Parser;
use crabml_llama2::sampler::Llama2Sampler;
use std::io::Write;
use crabml::error::Result;
use crabml::gguf::GGUFFileLoader;
use crabml_llama2::llama2::{Llama2Model, Llama2Runner};

#[derive(Parser, Debug)]
struct CommandArgs {
    /// The checkpoint file to load
    #[arg(short, long, default_value_t = format!("./testdata/tinyllamas-stories-260k-f32.gguf"))]
    model: String,

    // The number of tokens to generate
    #[arg(short, long, default_value_t = 300)]
    steps: usize,

    // The probability of sampling from the top-p.
    #[arg(short, long, default_value_t = 0.9)]
    probability: f32,

    #[arg(short, long, default_value_t = 1.0)]
    temperature: f32,

    /// The prompt
    prompt: String,
}

fn main() -> Result<()> {
    let args = CommandArgs::parse();

    // configure rayon
    let threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let gl = GGUFFileLoader::new(&args.model)?;
    let gf = gl.open()?;
    let lm = Llama2Model::from(&gf)?;

    let mut sampler = Llama2Sampler::new(lm.conf().vocab_size, args.temperature, args.probability);
    let mut runner = Llama2Runner::new(&lm.conf(), lm.weights(), lm.tokenizer())?;

    let mut output = runner.generate(&args.prompt, args.steps, &mut sampler)?;
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