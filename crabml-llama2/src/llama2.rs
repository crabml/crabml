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
use crabml::tensor::arithmetic::tensor_2d_matmul;
use crabml::tensor::arithmetic::tensor_2d_rms_norm;
use crabml::tensor::arithmetic::tensor_mul;
use crabml::tensor::Tensor;
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

struct Llama2State {
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
    key_cache: Vec<Vec<Vec<f32>>>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<Vec<Vec<f32>>>, // (layer, seq_len, kv_dim)
}

pub struct Llama2Runner<'a> {
    conf: Llama2Config,
    state: Llama2State,
    weights: Llama2Weights<'a>,
    tokenizer: &'a Llama2Tokenizer,
}

impl<'a> Llama2Runner<'a> {
    pub fn new(
        conf: &Llama2Config,
        weights: Llama2Weights<'a>,
        tokenizer: &'a Llama2Tokenizer,
    ) -> Self {
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
                .map(|_| {
                    (0..conf.seq_len)
                        .map(|_| vec![0.0; conf.kv_dim()])
                        .collect()
                })
                .collect(),
            value_cache: (0..conf.n_layers)
                .map(|_| {
                    (0..conf.seq_len)
                        .map(|_| vec![0.0; conf.kv_dim()])
                        .collect()
                })
                .collect(),
        };

        Self {
            conf: *conf,
            state,
            weights,
            tokenizer,
        }
    }

    pub fn generate(
        &'a mut self,
        prompt: &str,
        steps: usize,
        sampler: &'a mut Llama2Sampler,
    ) -> Result<Llama2RunnerOutputGenerator<'a>> {
        Llama2RunnerOutputGenerator::new(self, sampler, prompt, steps, self.conf.seq_len)
    }

    fn rope(&mut self, pos: usize, kv_dim: usize, head_size: usize) {
        for i in (0..kv_dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0 / 10000_f32.powf(head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < kv_dim { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
            for v in 0..rotn {
                let vec = if v == 0 {
                    &mut self.state.q
                } else {
                    &mut self.state.k
                };
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }

    fn multi_head_attention(&mut self, l: usize, pos: usize) {
        let head_size = self.conf.head_size();
        let kv_heads_per_head = self.conf.n_heads / self.conf.n_kv_heads;

        self.state
            .attn
            .par_iter_mut()
            .zip(self.state.xb.par_chunks_exact_mut(head_size))
            .enumerate()
            .for_each(|(h, (attn, xb))| {
                let kvh = h / kv_heads_per_head;
                // get the query vector for this head
                let q = &self.state.q[kvh * head_size..kvh * head_size + head_size];
                // iterate over all timesteps, including the current one
                for t in 0..(pos + 1) {
                    let k =
                        &self.state.key_cache[l][t][kvh * head_size..kvh * head_size + head_size];
                    // calculate the attention score as the dot product of q and k
                    let mut score = (0..head_size).map(|i| q[i] * k[i]).sum::<f32>();
                    score /= (head_size as f32).sqrt();
                    // save the score to the attention buffer
                    attn[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(&mut attn[0..pos + 1]);

                // weighted sum of the values, store back into xb
                xb.fill(0.0);
                for t in 0..pos + 1 {
                    let v =
                        &self.state.value_cache[l][t][kvh * head_size..kvh * head_size + head_size];
                    // get the attention weight for this timestep
                    let a = attn[t];
                    // accumulate the weighted value into xb
                    for i in 0..head_size {
                        xb[i] += a * v[i]
                    }
                }
            });
    }

    // input: self.state.x
    // output: self.state.xb
    fn ffn(&mut self, l: usize) -> Result<()> {
        let hidden_dim = self.conf.hidden_dim;

        // ffn rmsnorm
        rmsnorm(
            &mut self.state.xb,
            &self.state.x,
            self.weights.rms_ffn_weight[l].flat(),
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            &mut self.state.hb,
            &self.state.xb,
            self.weights.w1[l].flat(),
        );
        matmul(
            &mut self.state.hb2,
            &self.state.xb,
            self.weights.w3[l].flat(),
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
            &self.weights.w2[l].flat(),
        );

        // residual connection
        accum(&mut self.state.x, &self.state.xb);

        Ok(())
    }

    fn kv_cache(&mut self, l: usize, pos: usize) {
        let key_cache_row = &mut self.state.key_cache[l][pos];
        let value_cache_row = &mut self.state.value_cache[l][pos];
        key_cache_row.copy_from_slice(&self.state.k);
        value_cache_row.copy_from_slice(&self.state.v);
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&mut [f32]> {
        let embed_dim = self.conf.embedding_dim;
        let kv_dim = self.conf.kv_dim();
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_size = self.conf.head_size();

        // copy the token embedding into x
        let content_row = self.weights.token_embedding_table.subtensor(token)?;
        self.state.x.copy_from_slice(content_row.flat());

        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_t = Tensor::new(&self.state.x, vec![embed_dim])?;

            // attention rnsnorm
            let x_with_rms_norm_att = {
                let mut x_with_rms_norm =
                    Tensor::zeros(vec![embed_dim])?.with_name("x_with_rms_norm");
                tensor_2d_rms_norm(&mut x_with_rms_norm, &x_t, 1e-5)?;

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
                matmul(&mut self.state.q, &self.state.xb, self.weights.wq[l].flat());
                matmul(&mut self.state.k, &self.state.xb, self.weights.wk[l].flat());
                matmul(&mut self.state.v, &self.state.xb, self.weights.wv[l].flat());

                let mut q = Tensor::zeros(vec![embed_dim])?.with_name("q");
                let mut k = Tensor::zeros(vec![kv_dim])?.with_name("k");
                let mut v = Tensor::zeros(vec![kv_dim])?.with_name("v");
                tensor_2d_matmul(&mut q, &self.weights.wq[l], &x_with_rms_norm_att)?;
                tensor_2d_matmul(&mut k, &self.weights.wk[l], &x_with_rms_norm_att)?;
                tensor_2d_matmul(&mut v, &self.weights.wv[l], &x_with_rms_norm_att)?;

                self.state.q.copy_from_slice(q.flat());
                self.state.k.copy_from_slice(k.flat());
                self.state.v.copy_from_slice(v.flat());
                self.state.xb.copy_from_slice(x_with_rms_norm_att.flat());
                
                (q, k, v)
            };

            // ROPE
            {
                // k (kv_dim, ) => k (n_kv_head, head_size)
                let k = k.view(&[n_kv_heads, head_size])?;
                let v = v.view(&[n_kv_heads, head_size])?;
            }

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            self.rope(pos, self.conf.kv_dim(), self.conf.head_size());

            // save key,value at this time step (pos) to our kv cache
            // save .k, .v to kv_cache[l][pos]
            self.kv_cache(l, pos);

            // multihead attention. iterate over all heads
            // output to self.state.xb
            self.multi_head_attention(l, pos);

            // final matmul to get the output of the attention
            matmul(
                &mut self.state.xb2,
                &self.state.xb,
                self.weights.wo[l].flat(),
            );

            // residual connection back into x
            accum(&mut self.state.x, &self.state.xb2);

            // ffn
            self.ffn(l)?;
        }

        // final rmsnorm
        rmsnorm_inplace(&mut self.state.x, self.weights.rms_final_weight.flat());

        // classifier into logits
        matmul(
            &mut self.state.logits,
            &self.state.x,
            self.weights.wcls.flat(),
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
        matmul(out, &x, w.flat());
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
        let mut runner = Llama2Runner::new(&lm.conf, lm.weights, &lm.tokenizer);
        let output = runner.generate("Hello world", 15, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "s \nOn a little by and waly. She want");
        Ok(())
    }
}
