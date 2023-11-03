use std::ops::AddAssign;
use std::time::Duration;
use std::time::Instant;
use std::vec;

use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGUFFile;
use crabml::gguf::GGUFMetadata;
use crabml::tensor::cpu::raw_tensor::CpuTensorPoolRef;
use crabml::tensor::tensor::Tensor;
use crabml::tensor::tensor::TensorArithmetics;
use crabml::tensor::CpuTensor;
use crabml::tensor::tensor::TensorBatchMatmul;
use crabml::tokenizer::BpeTokenizer;

use crate::sampler::Llama2Sampler;

#[derive(Debug, Copy, Clone)]
pub struct Llama2Config {
    pub embedding_dim: usize, // the dim of embedding
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_dim: usize,
}

impl Llama2Config {
    pub fn kv_dim(&self) -> usize {
        (self.embedding_dim * self.n_kv_heads) / self.n_heads
    }

    fn head_size(&self) -> usize {
        self.embedding_dim / self.n_heads
    }
}

pub struct Llama2Weights<'a> {
    // token embedding table
    token_embedding_table: CpuTensor<'a>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<CpuTensor<'a>>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<CpuTensor<'a>>, // (layer, dim)
    // weights for matmuls
    wq: Vec<CpuTensor<'a>>, // (layer, embedding_dim, embedding_dim)
    wk: Vec<CpuTensor<'a>>, // (layer, kv_dim, embedding_dim)
    wv: Vec<CpuTensor<'a>>, // (layer, kv_dim, embedding_dim)
    wo: Vec<CpuTensor<'a>>, // (layer, embedding_dim, embedding_dim)
    // weights for ffn
    w1: Vec<CpuTensor<'a>>, // (layer, hidden_dim, embedding_dim)
    w2: Vec<CpuTensor<'a>>, // (layer, embedding_dim, hidden_dim)
    w3: Vec<CpuTensor<'a>>, // (layer, hidden_dim, embedding_dim)
    // final rmsnorm
    rms_final_weight: CpuTensor<'a>, // (dim, )
    // (optional) classifier weights for the logits, on the last layer
    wcls: CpuTensor<'a>, // (vocab_size, dim)
}

pub struct Llama2Model<'a> {
    conf: Llama2Config,
    weights: Llama2Weights<'a>,
    tokenizer: BpeTokenizer,
    metadata: &'a GGUFMetadata<'a>,
}

impl<'a> Llama2Model<'a> {
    pub fn from(gf: &'a GGUFFile<'a>, pool: CpuTensorPoolRef<'a>) -> Result<Self> {
        let conf = Self::load_config(gf);
        let weights = Self::load_weights(gf, conf.n_layers, pool)?;
        let tokenizer = Self::load_tokenizer(gf);
        Ok(Self {
            conf,
            weights,
            tokenizer,
            metadata: gf.metadata(),
        })
    }

    pub fn conf(&self) -> &Llama2Config {
        &self.conf
    }

    pub fn weights(&self) -> &Llama2Weights<'a> {
        &self.weights
    }

    pub fn metadata(&self) -> &'a GGUFMetadata<'a> {
        self.metadata
    }

    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    fn load_weights(gf: &'a GGUFFile<'a>, n_layers: usize, pool: CpuTensorPoolRef<'a>) -> Result<Llama2Weights<'a>> {
        // [64 (dim), 512 (vocab_size)]
        let token_embedding_table = Self::load_tensor(gf, "token_embd.weight", pool.clone())?;
        let mut wq = vec![];
        let mut wk = vec![];
        let mut wv = vec![];
        let mut wo = vec![];
        let mut w1 = vec![];
        let mut w2 = vec![];
        let mut w3 = vec![];
        let mut rms_att_weight = vec![];
        let mut rms_ffn_weight = vec![];
        for layer in 0..n_layers {
            wq.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_q.weight", layer), pool.clone()
            )?);
            wk.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_k.weight", layer), pool.clone(),
            )?);
            wv.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_v.weight", layer), pool.clone()
            )?);
            wo.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_output.weight", layer), pool.clone()
            )?);
            // (hidden_dim:172, embedding_dim:64)
            w1.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_gate.weight", layer), pool.clone()
            )?);
            w2.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_down.weight", layer), pool.clone()
            )?);
            w3.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_up.weight", layer), pool.clone()
            )?);
            rms_att_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_norm.weight", layer), pool.clone()
            )?);
            rms_ffn_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_norm.weight", layer), pool.clone()
            )?);
        }
        let rms_final_weight = Self::load_tensor(gf, "output_norm.weight", pool.clone())?;
        let wcls = Self::load_tensor(gf, "output.weight", pool.clone())?;
        Ok(Llama2Weights {
            token_embedding_table,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_att_weight,
            rms_ffn_weight,
            rms_final_weight,
            wcls,
        })
    }

    pub(crate) fn load_tensor(gf: &'a GGUFFile<'a>, name: &str, pool: CpuTensorPoolRef<'a>) -> Result<CpuTensor<'a>> {
        let info = match gf.get_tensor_info(name) {
            None => {
                return Err(Error {
                    kind: ErrorKind::IOError,
                    message: format!("failed to find tensor {}", name),
                    cause: None,
                });
            }
            Some(info) => info.clone(),
        };

        // the dimensions stored in GGUF seems in a reverse order of numpy's shape
        let dims = info
            .dimensions()
            .iter()
            .rev()
            .map(|v| *v)
            .collect::<Vec<_>>();

        let tensor = CpuTensor::from_bytes(info.data(), info.typ(), &dims, pool.clone())?;
        Ok(tensor)
    }

    fn load_tokenizer(gf: &GGUFFile) -> BpeTokenizer {
        let vocab = gf
            .metadata()
            .get_string_array("tokenizer.ggml.tokens")
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let vocab_scores = gf
            .metadata()
            .get_f32_array("tokenizer.ggml.scores")
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let eos_token = gf
            .metadata()
            .get_u32("tokenizer.ggml.eos_token_id")
            .unwrap() as usize;
        let bos_token = gf
            .metadata()
            .get_u32("tokenizer.ggml.bos_token_id")
            .unwrap() as usize;
        BpeTokenizer::new(vocab, vocab_scores, bos_token, eos_token)
    }

    fn load_config(gf: &GGUFFile) -> Llama2Config {
        // let rope_dims = gf.metadata().get_u32("llama.rope.dimension_count").unwrap();
        let n_heads = gf.metadata().get_u32("llama.attention.head_count").unwrap() as usize;
        let n_layers = gf.metadata().get_u32("llama.block_count").unwrap() as usize;
        let hidden_dim = gf.metadata().get_u32("llama.feed_forward_length").unwrap() as usize;
        let n_kv_heads = gf
            .metadata()
            .get_u32("llama.attention.head_count_kv")
            .unwrap() as usize;
        let seq_len = gf.metadata().get_u32("llama.context_length").unwrap() as usize;
        let vocab_size = gf
            .metadata()
            .get_string_array("tokenizer.ggml.tokens")
            .unwrap()
            .len();
        let embedding_dim = gf.metadata().get_u32("llama.embedding_length").unwrap() as usize;
        let rms_norm_eps = gf
            .metadata()
            .get_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap();
        let n_rot = gf.metadata().get_u32("llama.rope.dimension_count").unwrap() as usize;
        Llama2Config {
            n_heads,
            n_kv_heads,
            n_layers,
            embedding_dim,
            hidden_dim,
            seq_len,
            vocab_size,
            rms_norm_eps,
            rope_dim: n_rot,
        }
    }
}

struct Llama2State<'a> {
    logits: Vec<f32>, // output logits (vocab_size, )
    // ProbIndex *probindex; // buffer used in top-p sampling
    key_cache: Vec<CpuTensor<'a>>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<CpuTensor<'a>>, // (layer, seq_len, kv_dim)
}

pub struct Llama2Runner<'a> {
    conf: Llama2Config,
    state: Llama2State<'a>,
    weights: &'a Llama2Weights<'a>,
    tokenizer: &'a BpeTokenizer,
    pool: CpuTensorPoolRef<'a>,
}

impl<'a> Llama2Runner<'a> {
    pub fn new(
        conf: &Llama2Config,
        weights: &'a Llama2Weights<'a>,
        tokenizer: &'a BpeTokenizer,
        pool: CpuTensorPoolRef<'a>,
    ) -> Result<Self> {
        let state = Llama2State {
            logits: vec![0.0; conf.vocab_size],
            key_cache: (0..conf.n_layers)
                .map(|_| {
                    CpuTensor::new(
                        Vec::with_capacity(128 * conf.n_kv_heads * conf.head_size()),
                        &[0, conf.n_kv_heads, conf.head_size()],
                        pool.clone(),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            value_cache: (0..conf.n_layers)
                .map(|_| {
                    CpuTensor::new(
                        Vec::with_capacity(128 * conf.n_kv_heads * conf.head_size()),
                        &[0, conf.n_kv_heads, conf.head_size()],
                        pool.clone()
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        };

        Ok(Self {
            conf: *conf,
            state,
            weights,
            tokenizer,
            pool,
        })
    }

    pub fn generate(
        &'a mut self,
        prompt: &str,
        steps: usize,
        sampler: &'a mut Llama2Sampler,
    ) -> Result<Llama2RunnerOutputGenerator<'a>> {
        Llama2RunnerOutputGenerator::new(self, sampler, prompt, steps, self.conf.seq_len)
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&mut [f32]> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_size = self.conf.head_size();

        // copy the token embedding into x
        let mut x = CpuTensor::alloc(&[embed_dim], self.pool.clone())?;
        x.copy_from(&self.weights.token_embedding_table, &[token, 0], embed_dim)?;

        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_attn_orig = x.clone();

            // attention rnsnorm
            x = {
                x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
                x = x.mul_inplace(&self.weights.rms_att_weight[l])?;
                x
            };

            // matmul qkv for every head
            let (q, k, v) = {
                // wq: (embed_dim, embed_dim) @ x (embed_dim, ) => (embed_dim, )
                // wk: (kv_dim, embed_dim) @ x (embed_dim, ) => (kv_dim, )
                // wv: (kv_dim, embed_dim) @ x (embed_dim, ) => (kv_dim, )
                let q = self.weights.wq[l].matmul(&x)?;
                let k = self.weights.wk[l].matmul(&x)?;
                let v = self.weights.wv[l].matmul(&x)?;

                (q, k, v)
            };

            // ROPE
            let (q, k) = {
                let q = q.view(&[n_heads, head_size])?;
                let k = k.view(&[n_kv_heads, head_size])?;

                let q = q.rope_inplace(pos, self.conf.rope_dim)?;
                let k = k.rope_inplace(pos, self.conf.rope_dim)?;
                (q, k)
            };

            // save to kv cache
            {
                let v = v.view(&[n_kv_heads, head_size])?;

                self.state.key_cache[l].extend(&k)?;
                self.state.value_cache[l].extend(&v)?;
            };

            // multi query attention
            x = {
                let q = q.view(&[n_heads, head_size])?;
                let k_cache = self.state.key_cache[l].as_ref();
                let v_cache = self.state.value_cache[l].as_ref();

                // - key_cache: [seq, kv_head, head_size]
                // - key_cache = key_cache.repeat(1, n_head / n_kv_head, 1) => [seq, n_head, head_size]
                // - key_cache = key_cache.transpose(1, 0, 2) => [n_head, seq, head_size]
                // - q: [n_head, head_size]
                // - attn_score = batch_matmul(key_cache, q) => [n_head, seq]
                // - softmax(attn_score, axis=1) => [n_head, seq]
                // - val_cache: [seq, kv_head, head_size]
                // - val_cache = val_cache.repeat(1, n_head / n_kv_head, 1) => [seq, n_head, head_size]
                // - val_cache = val_cache.transpose(1, 2, 0) => [n_head, head_size, seq]
                // - out = batch_matmul(val_cache, atten_scores) => [n_head, head_size]

                // get attention scores
                let k_cache = k_cache
                    .repeat(&[1, n_heads / n_kv_heads, 1])?
                    .transpose(&[1, 0, 2])?;
                // (n_heads, n_seq, head_size) @ (n_head, head_size) => (n_heads, n_seq)
                let attn = k_cache.batch_matmul(&q)?;
                let attn = attn.div_scalar_inplace((head_size as f32).sqrt())?;
                let attn = attn.softmax_inplace(1)?;

                // get the weighted sum of the values and attention scores
                let v_cache = v_cache
                    .repeat(&[1, n_heads / n_kv_heads, 1])?
                    .transpose(&[1, 2, 0])?;
                // (n_heads, head_size, n_seq) @ (n_heads, n_seq) => (n_heads, head_size)
                let x_with_attn = v_cache.batch_matmul(&attn)?; // (n_heads, head_size)
                let x_with_attn = x_with_attn.view(&[embed_dim])?;

                // final matmul to get the output of the attention
                self.weights.wo[l].matmul(&x_with_attn)?
            };

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;

            // ffn
            x = {
                // save for redidual connection
                let x_orig_ffn = x.clone();

                // ffn rmsnorm
                x = {
                    x = x.rms_norm_inplace(1e-5)?;
                    x = x.mul_inplace(&self.weights.rms_ffn_weight[l])?;
                    x
                };

                // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
                // first calculate self.w1(x) and self.w3(x)
                // w1: (hidden_dim, embed_dim) @ x (embed_dim, ) => (hidden_dim, )
                // w3: (hidden_dim, embed_dim) @ x (embed_dim, ) => (hidden_dim, )
                let mut h1 = self.weights.w1[l].matmul(&x)?;
                let h2 = self.weights.w3[l].matmul(&x)?;

                // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
                h1 = h1.silu_inplace()?;

                // elementwise multiply with w3(x)
                h1 = h1.mul_inplace(&h2)?;

                // final matmul to get the output of the ffn
                x = self.weights.w2[l].matmul(&h1)?;

                // residual connection
                x = x.add_inplace(&x_orig_ffn)?;
                x
            }
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x
        };

        // classifier into logits
        let logits = self.weights.wcls.matmul(&x)?; // (vocab_size,

        self.state.logits = logits.iter().collect::<Vec<_>>();
        Ok(&mut self.state.logits)
    }
}

pub struct Llama2RunnerOutputGenerator<'a> {
    pos: usize,
    steps: usize,
    seq_len: usize,
    prompt_tokens: Vec<usize>,
    token: usize,
    sampler: &'a mut Llama2Sampler,
    runner: &'a mut Llama2Runner<'a>,
    total_time: Duration,
}

impl<'a> Llama2RunnerOutputGenerator<'a> {
    fn new(
        runner: &'a mut Llama2Runner<'a>,
        sampler: &'a mut Llama2Sampler,
        prompt: &str,
        steps: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let prompt_tokens = runner.tokenizer.encode(prompt, true, false)?;
        if prompt_tokens.is_empty() {
            return Err(Error {
                kind: ErrorKind::BadInput,
                message: "something is wrong, expected at least 1 prompt token".to_string(),
                cause: None,
            });
        }

        let token = prompt_tokens[0];
        Ok(Self {
            pos: 0,
            steps,
            token,
            prompt_tokens,
            sampler,
            runner,
            seq_len,
            total_time: Duration::new(0, 0),
        })
    }

    pub fn average_tokens_per_seconds(&self) -> f32 {
        let total_time = self.total_time.as_secs_f32();
        self.pos as f32 / total_time
    }

    fn forward_next(&mut self) -> Result<Option<String>> {
        if self.pos >= self.steps + self.prompt_tokens.len() {
            return Ok(None);
        }
        if self.pos >= self.seq_len {
            return Ok(None);
        }

        // forward the transformer to get logits for the next token
        let start_time = Instant::now();
        let logits = self.runner.forward(self.token, self.pos)?;

        // advance the state state machine
        let (next_token, is_prompt) = if self.pos < self.prompt_tokens.len() - 1 {
            // if we are still processing the input prompt, force the next prompt token
            (self.prompt_tokens[self.pos + 1], true)
        } else {
            // otherwise sample the next token from the logits
            let token = self.sampler.sample(logits)?;
            (token, false)
        };
        self.total_time.add_assign(start_time.elapsed());

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next_token == 1 {
            return Ok(None);
        }

        let prev_token = self.token;
        self.pos += 1;
        self.token = next_token;

        if is_prompt {
            return Ok(Some("".to_string()));
        }

        Ok(Some(self.runner.tokenizer.decode(prev_token, self.token)?))
    }
}

impl<'a> Iterator for Llama2RunnerOutputGenerator<'a> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let r = self.forward_next().transpose();
            if let Some(Ok(s)) = &r {
                if s.is_empty() {
                    continue;
                }
            }
            return r;
        }
    }
}

#[cfg(test)]
mod tests {
    use crabml::{gguf::GGUFFileLoader, tensor::cpu::raw_tensor::CpuTensorPool};

    use super::*;

    #[test]
    fn test_generate_f32() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15M-f32.gguf")?;
        let gf = gl.open()?;
        let pool = CpuTensorPool::new();
        let lm = Llama2Model::from(&gf, pool.clone())?;

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::new(&lm.conf, &lm.weights, &lm.tokenizer, pool)?;
        let output = runner.generate("Lily is a cat", 30, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(
            s,
            " who likes to play with yarn. She has many colors of yarn in her box. She likes to make shapes with yarn and show"
        );
        Ok(())
    }

    #[test]
    fn test_generate_q8_0() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-q8_0.gguf")?;
        let gf = gl.open()?;
        let pool = CpuTensorPool::new();
        let lm = Llama2Model::from(&gf, pool.clone())?;
        assert_eq!(lm.conf().rope_dim, 48);
        assert_eq!(lm.conf().head_size(), 48);

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::new(&lm.conf, &lm.weights, &lm.tokenizer, pool)?;
        let output = runner.generate("Lily is a cute cat, ", 10, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }
}
