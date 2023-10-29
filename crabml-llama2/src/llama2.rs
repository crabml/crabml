use std::ops::AddAssign;
use std::time::Duration;
use std::time::Instant;
use std::vec;

use crabml::backends::cpu::CpuTensor;
use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGUFFile;
use crabml::gguf::GGUFMetadata;
use crabml::tensor::tensor::Tensor;
use crabml::tensor::tensor::TensorBackendRef;
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
    token_embedding_table: Tensor<'a>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<Tensor<'a>>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<Tensor<'a>>, // (layer, dim)
    // weights for matmuls
    wq: Vec<Tensor<'a>>, // (layer, embedding_dim, embedding_dim)
    wk: Vec<Tensor<'a>>, // (layer, kv_dim, embedding_dim)
    wv: Vec<Tensor<'a>>, // (layer, kv_dim, embedding_dim)
    wo: Vec<Tensor<'a>>, // (layer, embedding_dim, embedding_dim)
    // weights for ffn
    w1: Vec<Tensor<'a>>, // (layer, hidden_dim, embedding_dim)
    w2: Vec<Tensor<'a>>, // (layer, embedding_dim, hidden_dim)
    w3: Vec<Tensor<'a>>, // (layer, hidden_dim, embedding_dim)
    // final rmsnorm
    rms_final_weight: Tensor<'a>, // (dim, )
    // (optional) classifier weights for the logits, on the last layer
    wcls: Tensor<'a>, // (vocab_size, dim)
}

pub struct Llama2Model<'a> {
    conf: Llama2Config,
    weights: Llama2Weights<'a>,
    tokenizer: BpeTokenizer,
    metadata: &'a GGUFMetadata<'a>,
}

impl<'a> Llama2Model<'a> {
    pub fn from(gf: &'a GGUFFile<'a>, backend: TensorBackendRef<'a>) -> Result<Self> {
        let conf = Self::load_config(gf);
        let tokenizer = Self::load_tokenizer(gf);
        let weights = Self::load_weights(gf, conf.n_layers, backend.clone())?;
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

    fn load_weights(
        gf: &'a GGUFFile<'a>,
        n_layers: usize,
        backend: TensorBackendRef<'a>,
    ) -> Result<Llama2Weights<'a>> {
        // [64 (dim), 512 (vocab_size)]
        let token_embedding_table = Self::load_tensor(gf, "token_embd.weight", backend.clone())?;
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
                &format!("blk.{}.attn_q.weight", layer),
                backend.clone(),
            )?);
            wk.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_k.weight", layer),
                backend.clone(),
            )?);
            wv.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_v.weight", layer),
                backend.clone(),
            )?);
            wo.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_output.weight", layer),
                backend.clone(),
            )?);
            // (hidden_dim:172, embedding_dim:64)
            w1.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_gate.weight", layer),
                backend.clone(),
            )?);
            w2.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_down.weight", layer),
                backend.clone(),
            )?);
            w3.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_up.weight", layer),
                backend.clone(),
            )?);
            rms_att_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_norm.weight", layer),
                backend.clone(),
            )?);
            rms_ffn_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_norm.weight", layer),
                backend.clone(),
            )?);
        }
        let rms_final_weight = Self::load_tensor(gf, "output_norm.weight", backend.clone())?;
        let wcls = Self::load_tensor(gf, "output.weight", backend.clone())?;
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

    pub(crate) fn load_tensor(
        gf: &'a GGUFFile<'a>,
        name: &str,
        backend: TensorBackendRef<'a>,
    ) -> Result<Tensor<'a>> {
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

        let cpu_tensor = CpuTensor::from_raw_bytes(info.data(), info.typ(), dims)?;
        let tensor = Tensor::from_cpu(cpu_tensor, backend)?;
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
    key_cache: Vec<Tensor<'a>>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<Tensor<'a>>, // (layer, seq_len, kv_dim)
}

pub struct Llama2Runner<'a, 'b>
where 'a: 'b
{
    conf: Llama2Config,
    state: Llama2State<'a>,
    weights: &'b Llama2Weights<'a>,
    tokenizer: &'b BpeTokenizer,
    backend: TensorBackendRef<'a>,
}

impl<'a, 'b> Llama2Runner<'a, 'b> {
    pub fn new(
        conf: &Llama2Config,
        weights: &'b Llama2Weights<'a>,
        tokenizer: &'b BpeTokenizer,
        backend: TensorBackendRef<'a>,
    ) -> Result<Self> {
        let state = Llama2State {
            logits: vec![0.0; conf.vocab_size],
            key_cache: (0..conf.n_layers)
                .map(|_| {
                    let cpu_tensor = CpuTensor::new(
                        Vec::with_capacity(128 * conf.n_kv_heads * conf.head_size()),
                        vec![0, conf.n_kv_heads, conf.head_size()],
                    )
                    .unwrap();
                    Tensor::from_cpu(cpu_tensor, backend.clone())
                })
                .collect::<Result<Vec<_>>>()?,
            value_cache: (0..conf.n_layers)
                .map(|_| {
                    let cpu_tensor = CpuTensor::new(
                        Vec::with_capacity(128 * conf.n_kv_heads * conf.head_size()),
                        vec![0, conf.n_kv_heads, conf.head_size()],
                    )
                    .unwrap();
                    Tensor::from_cpu(cpu_tensor, backend.clone())
                })
                .collect::<Result<Vec<_>>>()?,
        };

        Ok(Self {
            conf: *conf,
            state,
            weights,
            tokenizer,
            backend,
        })
    }

    pub fn generate(
        self,
        prompt: &str,
        steps: usize,
        sampler: Llama2Sampler,
    ) -> Result<Llama2RunnerOutputGenerator<'a, 'b>> {
        let seq_len = self.conf.seq_len;
        Llama2RunnerOutputGenerator::new(self, sampler, prompt, steps, seq_len)
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&mut [f32]> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_size = self.conf.head_size();
        let backend = self.backend.clone();

        // copy the token embedding into x
        let mut x = Tensor::zeros(&[embed_dim], backend.clone())?;
        x.copy_from(&self.weights.token_embedding_table, &[token, 0], embed_dim)?;

        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_attn_orig = x.clone();

            // attention rnsnorm
            x = {
                let x = x.rms_norm(self.conf.rms_norm_eps)?;
                let x = x.mul(&self.weights.rms_att_weight[l])?;
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
                let q = q.rope(pos, self.conf.rope_dim)?;
                let k = k.rope(pos, self.conf.rope_dim)?;

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
                // let k_cache = self.state.key_cache[l].as_ref();
                // let v_cache = self.state.value_cache[l].as_ref();

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
                let attn = attn.div_scalar((head_size as f32).sqrt())?;
                let attn = attn.softmax(1)?;

                // get the weighted sum of the values and attention scores
                let v_cache = v_cache
                    .repeat(&[1, n_heads / n_kv_heads, 1])?
                    .transpose(&[1, 2, 0])?;
                // (n_heads, head_size, n_seq) @ (n_heads, n_seq) => (n_heads, head_size)
                let x_with_attn = v_cache.batch_matmul(&attn)?; // (n_heads, head_size)
                let x_with_attn = x_with_attn.view(&[embed_dim])?;

                // final matmul to get the output of the attention
                let x = self.weights.wo[l].matmul(&x_with_attn)?;
                x
            };

            // residual connection back into x
            x = x.add(&x_attn_orig)?;

            // ffn
            x = {
                // save for redidual connection
                let x_orig_ffn = x.clone();

                // ffn rmsnorm
                x = {
                    let x = x.rms_norm(self.conf.rms_norm_eps)?;
                    let x = x.mul(&self.weights.rms_ffn_weight[l])?;
                    x
                };

                // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
                // first calculate self.w1(x) and self.w3(x)
                // w1: (hidden_dim, embed_dim) @ x (embed_dim, ) => (hidden_dim, )
                // w3: (hidden_dim, embed_dim) @ x (embed_dim, ) => (hidden_dim, )
                let h1 = self.weights.w1[l].matmul(&x)?;
                let h2 = &self.weights.w3[l].matmul(&x)?;

                // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
                let h1 = h1.silu()?;

                // elementwise multiply with w3(x)
                let h1 = h1.mul(h2)?;

                // final matmul to get the output of the ffn
                let x = self.weights.w2[l].matmul(&h1)?;

                // residual connection
                let x = x.add(&x_orig_ffn)?;
                x
            }
        }

        // final rmsnorm
        x = {
            let x = x.rms_norm(self.conf.rms_norm_eps)?;
            let x = x.mul(&self.weights.rms_final_weight)?;
            x
        };

        // classifier into logits
        let logits = self.weights.wcls.matmul(&x)?; // (vocab_size,
        logits.export(&mut self.state.logits)?;

        Ok(&mut self.state.logits)
    }
}

pub struct Llama2RunnerOutputGenerator<'a, 'b> {
    pos: usize,
    steps: usize,
    seq_len: usize,
    prompt_tokens: Vec<usize>,
    token: usize,
    sampler: Llama2Sampler,
    runner: Llama2Runner<'a, 'b>,
    total_time: Duration,
}

impl<'a, 'b> Llama2RunnerOutputGenerator<'a, 'b> {
    fn new(
        runner: Llama2Runner<'a, 'b>,
        sampler: Llama2Sampler,
        prompt: &str,
        steps: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let prompt_tokens = runner.tokenizer.encode(prompt, true, false)?;
        if prompt_tokens.is_empty() {
            return Err(Error {
                kind: ErrorKind::InvalidArgs,
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

impl<'a, 'b> Iterator for Llama2RunnerOutputGenerator<'a, 'b> {
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
    use crabml::backends::cpu::backend::CpuTensorBackend;
    use crabml::gguf::GGUFFileLoader;
    use crabml::tensor::tensor::TensorBackend;

    use super::*;

    #[test]
    fn test_generate_f32() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15M-f32.gguf")?;
        let gf = gl.open()?;
        let backend = CpuTensorBackend::new();
        let lm = Llama2Model::from(&gf, backend.clone())?;

        let sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let runner = Llama2Runner::new(&lm.conf, &lm.weights, &lm.tokenizer, backend.clone())?;
        let output = runner.generate("Lily is a cat", 30, sampler)?;
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
        let backend = CpuTensorBackend::new();
        let lm = Llama2Model::from(&gf, backend.clone())?;
        assert_eq!(lm.conf().rope_dim, 48);
        assert_eq!(lm.conf().head_size(), 48);

        let sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let runner = Llama2Runner::new(&lm.conf, &lm.weights, &lm.tokenizer, backend.clone())?;
        let output = runner.generate("Lily is a cute cat, ", 10, sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }
}
