use crate::math::accum;
use crate::math::matmul;
use crate::math::rmsnorm;
use crate::math::rmsnorm_inplace;
use crate::math::softmax;
use crate::sampler::Llama2Sampler;
use crate::tokenizer;
use crate::tokenizer::Llama2Tokenizer;
use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGUFFile;
use crabml::gguf::GGUFFileLoader;
use crabml::tensor::arithmetic::tensor_1d_softmax_inplace;
use crabml::tensor::arithmetic::tensor_2d_matmul;
use crabml::tensor::arithmetic::tensor_2d_rms_norm;
use crabml::tensor::arithmetic::tensor_add_inplace;
use crabml::tensor::arithmetic::tensor_copy_chunk;
use crabml::tensor::arithmetic::tensor_mul;
use crabml::tensor::arithmetic::tensor_mul_inplace;
use crabml::tensor::arithmetic::tensor_multi_query_attention;
use crabml::tensor::arithmetic::tensor_rope_inplace;
use crabml::tensor::Tensor;
use crabml::tensor::arithmetic::tensor_silu_inplace;
use rayon::prelude::*;
use std::ops::AddAssign;
use std::slice;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::vec;

#[derive(Debug, Copy, Clone)]
pub struct Llama2Config {
    pub embedding_dim: usize, // the dim of embedding
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

impl Llama2Config {
    pub fn kv_dim(&self) -> usize {
        (self.embedding_dim * self.n_kv_heads) / self.n_heads
    }

    fn head_size(&self) -> usize {
        self.embedding_dim / self.n_heads
    }
}

#[derive(Default)]
pub struct Llama2Weights<'a> {
    // token embedding table
    token_embedding_table: Tensor<'a>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<Tensor<'a>>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<Tensor<'a>>, // (layer, dim)
    // weights for matmuls
    wq: Vec<Tensor<'a>>, // (layer, embedding_dim, embedding_dim)
    wk: Vec<Tensor<'a>>, // (layer, embedding_dim, kv_dim)
    wv: Vec<Tensor<'a>>, // (layer, embeddin_dim, kv_dim)
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
    tokenizer: Llama2Tokenizer,
}

impl<'a> Llama2Model<'a> {
    pub fn from(gf: &'a GGUFFile<'a>) -> Result<Self> {
        let conf = Self::load_config(gf);
        let weights = Self::load_weights(gf, conf.n_layers)?;
        let tokenizer = Self::load_tokenizer(gf);
        Ok(Self {
            conf,
            weights,
            tokenizer,
        })
    }

    fn load_weights(gf: &'a GGUFFile<'a>, n_layers: usize) -> Result<Llama2Weights<'a>> {
        // [64 (dim), 512 (vocab_size)]
        let token_embedding_table = Self::load_tensor(gf, "token_embd.weight")?;
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
            )?);
            wk.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_k.weight", layer),
            )?);
            wv.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_v.weight", layer),
            )?);
            wo.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_output.weight", layer),
            )?);
            // (hidden_dim:172, embedding_dim:64)
            w1.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_gate.weight", layer),
            )?);
            w2.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_down.weight", layer),
            )?);
            w3.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_up.weight", layer),
            )?);
            rms_att_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_norm.weight", layer),
            )?);
            rms_ffn_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_norm.weight", layer),
            )?);
        }
        let rms_final_weight = Self::load_tensor(gf, "output_norm.weight")?;
        let wcls = Self::load_tensor(gf, "output.weight")?;
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

    pub(crate) fn load_tensor(gf: &'a GGUFFile<'a>, name: &str) -> Result<Tensor<'a>> {
        let info = match gf.get_tensor_info(name) {
            None => {
                return Err(Error {
                    kind: ErrorKind::IOError,
                    message: format!("failed to find tensor {}", name),
                    cause: None,
                })
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

        let tensor = Tensor::from_raw_bytes(info.data(), dims)?.with_name(name.to_string());
        Ok(tensor)
    }

    fn load_tokenizer(gf: &GGUFFile) -> Llama2Tokenizer {
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
        Llama2Tokenizer::new(vocab, vocab_scores, 27, bos_token, eos_token)
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
        Llama2Config {
            n_heads,
            n_kv_heads,
            n_layers,
            embedding_dim,
            hidden_dim,
            seq_len,
            vocab_size,
        }
    }
}

struct Llama2State<'a> {
    x: Vec<f32>,         // activation at current time stamp (embedding_dim,)
    xb: Vec<f32>,        // same, but inside a residual branch (embedding_dim,)
    xb2: Vec<f32>,       // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,        // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,       // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,         // query (dim, )
    k: Vec<f32>,         // key (kv_dim, )
    v: Vec<f32>,         // value (kv_dim, )
    attn: Vec<Vec<f32>>, // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>,    // output logits (vocab_size, )
    // ProbIndex *probindex; // buffer used in top-p sampling
    key_cache: Vec<Tensor<'a>>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<Tensor<'a>>, // (layer, seq_len, kv_dim)
}

pub struct Llama2Runner<'a> {
    conf: Llama2Config,
    state: Llama2State<'a>,
    weights: Llama2Weights<'a>,
    tokenizer: &'a Llama2Tokenizer,
}

impl<'a> Llama2Runner<'a> {
    pub fn new(
        conf: &Llama2Config,
        weights: Llama2Weights<'a>,
        tokenizer: &'a Llama2Tokenizer,
    ) -> Result<Self> {
        let state = Llama2State {
            x: vec![0.0; conf.embedding_dim],
            xb: vec![0.0; conf.embedding_dim],
            xb2: vec![0.0; conf.embedding_dim],
            hb: vec![0.0; conf.hidden_dim],
            hb2: vec![0.0; conf.hidden_dim],
            q: vec![0.0; conf.embedding_dim],
            k: vec![0.0; conf.kv_dim()],
            v: vec![0.0; conf.kv_dim()],
            attn: (0..conf.n_heads)
                .map(|_| vec![0.0; conf.embedding_dim])
                .collect(),
            logits: vec![0.0; conf.vocab_size],
            key_cache: (0..conf.n_layers)
                .map(|_| Tensor::zeros(vec![conf.seq_len, conf.n_kv_heads, conf.head_size()]))
                .collect::<Result<Vec<_>>>()?,
            value_cache: (0..conf.n_layers)
                .map(|_| Tensor::zeros(vec![conf.seq_len, conf.n_kv_heads, conf.head_size()]))
                .collect::<Result<Vec<_>>>()?,
        };

        Ok(Self {
            conf: *conf,
            state,
            weights,
            tokenizer,
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

    // input: self.state.x
    // output: self.state.xb
    fn ffn(&mut self, l: usize) -> Result<()> {
        let hidden_dim = self.conf.hidden_dim;

        // ffn rmsnorm
        rmsnorm(
            &mut self.state.xb,
            &self.state.x,
            self.weights.rms_ffn_weight[l].ref_buf(),
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            &mut self.state.hb,
            &self.state.xb,
            self.weights.w1[l].ref_buf(),
        );
        matmul(
            &mut self.state.hb2,
            &self.state.xb,
            self.weights.w3[l].ref_buf(),
        );

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim {
            self.state.hb[i] = self.state.hb[i] * (1.0 / (1.0 + (-self.state.hb[i]).exp()));
        }

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            self.state.hb[i] *= self.state.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(
            &mut self.state.xb,
            &self.state.hb,
            &self.weights.w2[l].ref_buf(),
        );

        // residual connection
        accum(&mut self.state.x, &self.state.xb);

        Ok(())
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&mut [f32]> {
        let embed_dim = self.conf.embedding_dim;
        let kv_dim = self.conf.kv_dim();
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_size = self.conf.head_size();
        let hidden_dim = self.conf.hidden_dim;

        // copy the token embedding into x
        let content_row = self.weights.token_embedding_table.subtensor(token)?;
        self.state.x.copy_from_slice(content_row.ref_buf());
        let mut x = Tensor::new(self.state.x.to_vec(), vec![embed_dim])?;

        // forward all the layers
        for l in 0..self.conf.n_layers {

            // attention rnsnorm
            let x_with_rms_norm_att = {
                let mut x_with_rms_norm =
                    Tensor::zeros(vec![embed_dim])?.with_name("x_with_rms_norm");
                tensor_2d_rms_norm(&mut x_with_rms_norm, &x, 1e-5)?;

                let mut x_with_rms_norm_att =
                    Tensor::zeros(vec![embed_dim])?.with_name("x_with_rms_norm_att");
                tensor_mul(
                    &mut x_with_rms_norm_att,
                    &self.weights.rms_att_weight[l],
                    &x_with_rms_norm,
                )?;
                x_with_rms_norm_att
            };

            // matmul qkv for every head
            let (q, k, v) = {
                // .q(embedding_dim, ) = (xb(1, embedding_dim) * wq(embedding_dim, embedding_dim)).T
                // .k(kv_dim, ) = (xb(1, embedding_dim) * wq(embedding_dim, kv_dim)).T
                // .v(kv_dim, ) = (xb(1, embedding_dim) * wv(embedding_dim, kv_dim)).T
                let mut q = Tensor::zeros(vec![embed_dim])?.with_name("q");
                let mut k = Tensor::zeros(vec![kv_dim])?.with_name("k");
                let mut v = Tensor::zeros(vec![kv_dim])?.with_name("v");
                tensor_2d_matmul(&mut q, &self.weights.wq[l], &x_with_rms_norm_att)?;
                tensor_2d_matmul(&mut k, &self.weights.wk[l], &x_with_rms_norm_att)?;
                tensor_2d_matmul(&mut v, &self.weights.wv[l], &x_with_rms_norm_att)?;

                (q, k, v)
            };

            // ROPE
            let (k, q) = {
                // k (kv_dim, ) => k (n_kv_head, head_size)
                let mut k = k.view(&[n_kv_heads, head_size])?;
                let mut q = q.view(&[n_heads, head_size])?;
                tensor_rope_inplace(&mut q, &mut k, pos, 1.0, 10000_f32)?;
                (k, q)
            };

            // save to kv cache
            {
                let k = k.view(&[kv_dim])?;
                let v = v.view(&[kv_dim])?;
                tensor_copy_chunk(&mut self.state.key_cache[l], pos, &k)?;
                tensor_copy_chunk(&mut self.state.value_cache[l], pos, &v)?;
            };

            // multihead attention. iterate over all heads
            // output to self.state.xb
            // q: (n_heads, head_size)
            // key_cache: (seq, n_kv_heads, head_size)
            // attn_scores: (seq, )
            // value_cache: (seq, kv_heads, head_size)
            let x_with_attn_wo = {
                let q = q.view(&[n_heads, head_size])?;
                let mut attn_scores =
                    Tensor::zeros(vec![self.conf.seq_len])?.with_name("attn_scores");
                let mut x_with_attn =
                    Tensor::zeros(vec![n_heads, head_size])?.with_name("x_with_attn");
                tensor_multi_query_attention(
                    &mut x_with_attn,
                    &mut attn_scores,
                    &q,
                    &self.state.key_cache[l],
                    &self.state.value_cache[l],
                    pos,
                )?;
                let x_with_attn = x_with_attn.view(&[embed_dim])?;

                // final matmul to get the output of the attention
                let mut x_with_attn_wo =
                    Tensor::zeros(vec![embed_dim])?.with_name("x_with_attn_wo");
                tensor_2d_matmul(&mut x_with_attn_wo, &self.weights.wo[l], &x_with_attn)?;
                x_with_attn_wo
            };

            // residual connection back into x
            tensor_add_inplace(&mut x, &x_with_attn_wo)?;
            self.state.x.copy_from_slice(x.ref_buf());

            // ffn
            {
                // ffn rmsnorm
                let x_with_rms_norm_ffn = {
                    let mut x_with_rms_norm =
                        Tensor::zeros(vec![embed_dim])?.with_name("x_with_rms_norm");
                    tensor_2d_rms_norm(&mut x_with_rms_norm, &x, 1e-5)?;

                    let mut x_with_rms_norm_ffn =
                        Tensor::zeros(vec![embed_dim])?.with_name("x_with_rms_norm_ffn");
                    tensor_mul(
                        &mut x_with_rms_norm_ffn,
                        &self.weights.rms_ffn_weight[l],
                        &x_with_rms_norm,
                    )?;
                    x_with_rms_norm_ffn
                };

                let mut h1 = Tensor::zeros(vec![hidden_dim])?.with_name("h1");
                let mut h2 = Tensor::zeros(vec![hidden_dim])?.with_name("h2");

                // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
                // first calculate self.w1(x) and self.w3(x)
                tensor_2d_matmul(&mut h1, &self.weights.w1[l], &x_with_rms_norm_ffn)?;
                tensor_2d_matmul(&mut h2, &self.weights.w3[l], &x_with_rms_norm_ffn)?;

                // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
                tensor_silu_inplace(&mut h1)?;

                // elementwise multiply with w3(x)
                tensor_mul_inplace(&mut h1, &h2)?;

                // final matmul to get the output of the ffn
                let mut x_ffn_out = Tensor::zeros(vec![embed_dim])?.with_name("x_ffn_out");
                tensor_2d_matmul(&mut x_ffn_out, &self.weights.w2[l], &h1)?;

                // residual connection
                tensor_add_inplace(&mut x, &x_ffn_out)?;
            }
        }
        self.state.x.copy_from_slice(x.ref_buf());

        // final rmsnorm
        rmsnorm_inplace(&mut self.state.x, self.weights.rms_final_weight.ref_buf());

        // classifier into logits
        matmul(
            &mut self.state.logits,
            &self.state.x,
            self.weights.wcls.ref_buf(),
        );

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
    use super::*;

    #[test]
    fn test_accum() {
        let mut a = [1.0, 2.0];
        let b = [1.2, 3.0];
        accum(&mut a, &b);
        assert_eq!(a[0], 2.2);
        assert_eq!(a[1], 5.0);
    }

    #[test]
    fn test_matmul() {
        let wvec = vec![1.0, 2.0, 3.0, 1.0, 5.0, 1.0];
        let w = Tensor::new(&wvec, vec![2, 3]).unwrap(); // (2,3)
        let x = [2.0, 4.0, 8.0]; // (3,)
        let out: &mut [f32; 2] = &mut [0.0, 0.0]; // (2, )
        matmul(out, &x, w.ref_buf());
        assert_eq!(out[0], 34.0);
        assert_eq!(out[1], 30.0);
    }

    #[test]
    fn test_gguf_tokenizer() -> Result<()> {
        let gf_loader = GGUFFileLoader::new("../testdata/tinyllamas-stories-260k-f32.gguf")?;
        let gf = gf_loader.open()?;
        let lm = Llama2Model::from(&gf)?;
        let tk = lm.tokenizer;

        assert_eq!(tk.decode(2, 3)?, "\u{0}");
        assert_eq!(tk.decode(2, 5)?, "\u{2}");
        assert_eq!(tk.decode(2, 6)?, "\u{3}");
        assert_eq!(tk.decode(2, 100)?, "a");

        let tests = vec![
            (
                "hello, world",
                "<s> - he - ll - o - , - <0x20> - w - or - ld - </s>",
            ),
            ("tiktok", "<s> - t - i - k - t - o - k - </s>"),
        ];

        for tt in tests {
            let tokens = tk.encode(tt.0, true, true)?;
            let tokens_in_string = tokens
                .iter()
                .map(|t| tk.vocab()[*t].clone())
                .collect::<Vec<String>>()
                .join(" - ");
            assert_eq!(tokens_in_string, tt.1, "failed to encode {}", tt.0);
        }
        Ok(())
    }

    #[test]
    fn test_generate_gguf() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/tinyllamas-stories-260k-f32.gguf")?;
        let gf = gl.open()?;
        let lm = Llama2Model::from(&gf)?;

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::new(&lm.conf, lm.weights, &lm.tokenizer)?;
        let output = runner.generate("Lily is a cat ", 30, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(
            s,
            ". She was a shals to almals. She loved to shals to her mommy."
        );
        Ok(())
    }
}
