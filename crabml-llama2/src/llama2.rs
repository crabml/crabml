use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::AddAssign;
use std::slice;
use std::time::Duration;
use std::time::Instant;
use std::vec;
use crabml::tensor::Tensor;
use crabml::error::Result;
use crabml::error::ErrorKind;
use crabml::error::Error;
use crabml::gguf::GGUFFile;
use crabml::gguf::GGUFFileLoader;

fn accum(a: &mut [f32], b: &[f32]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += b;
    }
}

fn softmax(a: &mut [f32]) {
    let max = a.iter().fold(f32::NAN, |a, b| a.max(*b));
    let mut sum = 0.0;
    for a in a.iter_mut() {
        *a = (*a - max).exp();
        sum += *a;
    }
    for a in a.iter_mut() {
        *a /= sum;
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], w: &[f32]) {
    let ss = x.iter().fold(0.0, |s, n| s + n * n);
    let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
    // normalize and scale
    for i in 0..o.len() {
        o[i] = x[i] * w[i] / rms;
    }
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32]) {
    let ss = x.iter().fold(0.0, |s, n| s + n * n);
    let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
    // normalize and scale
    for i in 0..x.len() {
        x[i] = x[i] * w[i] / rms;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    // W (d,n) @ x (n,) -> xout (d,)
    let x_dim = x.len();
    xout.iter_mut().enumerate().for_each(|(i, xo)| {
        *xo = 0.0;
        for j in 0..x.len() {
            *xo += w[i * x_dim + j] * x[j];
        }
    });
}


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

pub trait Llama2Loader {
    fn load(&self) -> Result<(Llama2Config, Llama2Weights, Llama2Tokenizer)>;
}

pub struct Llama2GgufLoader {
    inner: GGUFFileLoader,
}

impl Llama2GgufLoader {
    pub fn new(path: &str) -> Result<Self> {
        let inner = GGUFFileLoader::new(path).map_err(|err| Error {
            kind: ErrorKind::IOError,
            message: format!("failed to open file {}: {}", path, err),
            cause: Some(Box::new(err)),
        })?;

        Ok(Self { inner })
    }

    fn load_weights<'a>(gf: &GGUFFile<'a>, n_layers: usize) -> Result<Llama2Weights<'a>> {
        // [64 (dim), 512 (vocab_size)]
        let token_embedding_table = {
            let (tensor, dims) = Self::load_tensor(gf, "token_embd.weight")?;
            let tensor = tensor.view(&[dims[1], dims[0]])?;
            tensor
        };
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
            )?.0);
            wk.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_k.weight", layer),
            )?.0);
            wv.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_v.weight", layer),
            )?.0);
            wo.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_output.weight", layer),
            )?.0);

            // (hidden_dim:172, embedding_dim:64)
            w1.push({
                let (tensor, dims) = Self::load_tensor(gf, &format!("blk.{}.ffn_gate.weight", layer))?;
                let tensor = tensor.view(&[dims[1], dims[0]])?;
                tensor
            });
            w2.push({
                let (tensor, dims) = Self::load_tensor(gf, &format!("blk.{}.ffn_down.weight", layer))?;
                let tensor = tensor.view(&[dims[1], dims[0]])?;
                tensor
            });
            w3.push({
                let (tensor, dims) = Self::load_tensor(gf, &format!("blk.{}.ffn_up.weight", layer))?;
                let tensor = tensor.view(&[dims[1], dims[0]])?;
                tensor
            });
            rms_att_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_norm.weight", layer),
            )?.0);
            rms_ffn_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_norm.weight", layer),
            )?.0);
        }
        let rms_final_weight = Self::load_tensor(gf, "output_norm.weight")?.0;
        let wcls = {
            let (tensor, dims) = Self::load_tensor(gf, "output.weight")?;
            let tensor = tensor.view(&[dims[1], dims[0]])?;
            tensor
        };
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

    pub(crate) fn load_tensor<'a>(gf: &GGUFFile<'a>, name: &str) -> Result<(Tensor<'a>, Vec<usize>)> {
        let info = match gf.get_tensor_info(name) {
            None => {
                return Err(Error {
                    kind: ErrorKind::IOError,
                    message: format!("failed to find tensor {}", name),
                    cause: None,
                })
            }
            Some(info) => info,
        };

        let len = info.data().len();
        assert_eq!(
            len % std::mem::size_of::<f32>(),
            0,
            "Length of slice must be multiple of f32 size"
        );
        let new_len = len / std::mem::size_of::<f32>();
        let ptr = info.data().as_ptr() as *const f32;
        let f32_data = unsafe { slice::from_raw_parts(ptr, new_len) };
        let tensor = Tensor::new(f32_data, info.dimensions().to_vec())?;
        Ok((tensor, info.dimensions().to_vec()))
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

impl Llama2Loader for Llama2GgufLoader {
    fn load(&self) -> Result<(Llama2Config, Llama2Weights, Llama2Tokenizer)> {
        let gf = self.inner.load().map_err(|err| Error {
            kind: ErrorKind::IOError,
            message: format!("failed to load gguf file"),
            cause: Some(Box::new(err)),
        })?;
        let config = Self::load_config(&gf);
        let tokenizer = Self::load_tokenizer(&gf);
        let weights = Self::load_weights(&gf, config.n_layers)?;
        Ok((config, weights, tokenizer))
    }
}

pub struct Llama2Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    token_buf_len: usize,
    byte_pieces: [u8; 256],
    vocab_index: HashMap<String, usize>,
    bos_token: usize,
    eos_token: usize,
}

impl Llama2Tokenizer {
    pub fn new(
        vocab: Vec<String>,
        vocab_scores: Vec<f32>,
        token_buf_len: usize,
        bos_token: usize,
        eos_token: usize,
    ) -> Self {
        let vocab_index = vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        let mut byte_pieces = [0u8; 256];
        for (i, p) in byte_pieces.iter_mut().enumerate() {
            *p = i as u8
        }

        Self {
            vocab,
            vocab_index,
            vocab_scores,
            token_buf_len,
            byte_pieces,
            bos_token,
            eos_token,
        }
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> Result<String> {
        let mut piece: &[u8] = self.vocab[token].as_bytes();
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if prev_token == 1 && piece[0] == b' ' {
            piece = &piece[1..];
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        if piece.starts_with(b"<0x") && piece[piece.len() - 1] == b'>' {
            let s = String::from_utf8_lossy(&piece[1..piece.len() - 1]);
            let s = s.trim_start_matches("0x");
            if let Ok(byte) = u8::from_str_radix(s, 16) {
                piece = &self.byte_pieces[(byte as usize)..(byte as usize) + 1]
            }
        }

        let mut s = String::from_utf8_lossy(piece).to_string();
        s = s.replace('▁', " ");
        Ok(s)
    }

    #[allow(dead_code)]
    pub fn decode_string(&self, tokens: &[usize]) -> Result<String> {
        let mut result = String::new();
        let mut prev_token = 0;
        for token in tokens {
            let piece = self.decode(prev_token, *token)?;
            result.push_str(&piece);
            prev_token = *token;
        }
        Ok(result)
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<usize>> {
        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        let mut token_buf = String::with_capacity(self.token_buf_len * 2 + 1 + 2);
        let mut tokens: Vec<usize> = vec![];

        if bos {
            tokens.push(self.bos_token);
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if !text.starts_with('\u{0}') {
            if let Some(dummy_prefix) = self.vocab_index.get(" ") {
                tokens.push(*dummy_prefix);
            }
        }

        let chars = text.chars();
        for ch in chars {
            token_buf.clear();
            token_buf.push(ch);
            if let Some(tok) = self.vocab_index.get(&token_buf) {
                // we found this codepoint in vocab, add it as a token
                tokens.push(*tok);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for byte in token_buf.bytes() {
                    tokens.push(byte as usize + 3);
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx: Option<usize> = None;
            let mut best_token: Option<usize> = None;
            let mut i = 0;

            while i < (tokens.len() - 1) {
                token_buf.clear();
                token_buf.push_str(&self.vocab[tokens[i]]);
                token_buf.push_str(&self.vocab[tokens[i + 1]]);
                if let Some(tok) = self.vocab_index.get(&token_buf) {
                    let new_score = self.vocab_scores[*tok];
                    if new_score > best_score {
                        best_score = new_score;
                        best_idx = Some(i);
                        best_token = Some(*tok);
                    }
                }
                i += 1;
            }

            if let Some(idx) = best_idx {
                tokens[idx] = best_token.unwrap();
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        if eos {
            tokens.push(self.eos_token);
        }

        Ok(tokens)
    }
}

pub struct Llama2Sampler {
    prob_index: Vec<(f32, usize)>,
    temperature: f32,
    topp: f32,
}

impl Llama2Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32) -> Self {
        Self {
            prob_index: vec![(0.0, 0); vocab_size],
            temperature,
            topp,
        }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> Result<usize> {
        if self.temperature == 0.0 {
            return Self::sample_argmax(logits);
        }

        // apply the temperature to the logits. the lower the temperature,
        // the more deterministic the sampling.
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits);

        // flip a (float) coin (this is our source of entropy for sampling)
        let mut rng = rand::thread_rng();
        let coin: f32 = rng.gen_range(0.0..1.0);

        // we sample from this distribution to get the next token
        if self.topp <= 0_f32 || self.topp >= 1.0_f32 {
            // simply sample from the predicted probability distribution
            Self::sample_multi(logits, coin);
        }

        Self::sample_topp(logits, self.topp, &mut self.prob_index, coin)
    }

    pub fn sample_multi(probs: &[f32], coin: f32) -> usize {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        let mut cdf = 0_f32;
        for (i, p) in probs.iter().enumerate() {
            cdf += p;
            if cdf > coin {
                return i;
            }
        }
        probs.len() - 1 // in case of rounding errors
    }

    pub fn sample_topp(
        probs: &[f32],
        topp: f32,
        prob_index: &mut [(f32, usize)],
        coin: f32,
    ) -> Result<usize> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        let cutoff = (1.0_f32 - topp) / (probs.len() - 1) as f32;
        let mut n0 = 0;
        for (i, prob) in probs.iter().enumerate() {
            if *prob >= cutoff {
                prob_index[n0] = (probs[i], i);
                n0 += 1;
            }
        }
        prob_index[..n0].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // truncate the list where cumulative probability exceeds topp
        let mut cumulative_prob = 0_f32;
        let mut last_idx = n0 - 1; // in case of rounding errors consider all elements
        for (i, prob) in prob_index[0..n0].iter().enumerate() {
            cumulative_prob += prob.0;
            if cumulative_prob > topp {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0_f32;
        for prob in prob_index[0..=last_idx].iter() {
            cdf += prob.0;
            if cdf > r {
                return Ok(prob.1);
            }
        }
        Ok(prob_index[last_idx].1) // in case of rounding errors
    }

    pub fn sample_argmax(probs: &[f32]) -> Result<usize> {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .ok_or_else(|| Error {
                kind: ErrorKind::Unexpected,
                message: format!("failed to sample from logits"),
                cause: None,
            })
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
    tokenizer: Llama2Tokenizer,
}

impl<'a> Llama2Runner<'a> {
    pub fn new(
        conf: &Llama2Config,
        weights: Llama2Weights<'a>,
        tokenizer: Llama2Tokenizer,
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

    fn head_size(&self) -> usize {
        self.conf.embedding_dim / self.conf.n_heads
    }

    fn kv_dim(&self) -> usize {
        (self.conf.embedding_dim * self.conf.n_kv_heads) / self.conf.n_heads
    }

    fn rope(&mut self, pos: usize) {
        for i in (0..self.conf.embedding_dim).step_by(2) {
            let head_dim = i % self.head_size();
            let freq = 1.0 / 10000_f32.powf(head_dim as f32 / self.head_size() as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < self.kv_dim() { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
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
        let head_size = self.head_size();
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
                    let k = &self.state.key_cache[l][t][kvh * head_size..kvh * head_size + head_size];
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
                    let v = &self.state.value_cache[l][t][kvh * head_size..kvh * head_size + head_size];
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
        // copy the token embedding into x
        let content_row = self.weights.token_embedding_table.subtensor(token)?;
        self.state.x.copy_from_slice(content_row.flat());

        // forward all the layers
        for l in 0..self.conf.n_layers {
            // attention rnsnorm
            rmsnorm(
                &mut self.state.xb,
                &self.state.x,
                self.weights.rms_att_weight[l].flat(),
            );

            // matmul qkv for every head
            // .q(embedding_dim, ) = xb(embedding_dim, ) * wq(embedding_dim, embedding_dim)
            // .k(kv_dim, ) = xb(embedding_dim, ) * wq(embedding_dim, kv_dim)
            // .v(kv_dim, ) = xb(embedding_dim, ) * wv(embedding_dim, kv_dim)
            matmul(&mut self.state.q, &self.state.xb, self.weights.wq[l].flat());
            matmul(&mut self.state.k, &self.state.xb, self.weights.wk[l].flat());
            matmul(&mut self.state.v, &self.state.xb, self.weights.wv[l].flat());

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            self.rope(pos);

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
    fn test_generate_gguf() -> Result<()> {
        let gguf_loader = Llama2GgufLoader::new("../testdata/tinyllamas-stories-260k-f32.gguf")?;

        let (conf, weights, tokenizer) = gguf_loader.load()?;
        let mut sampler = Llama2Sampler::new(conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::new(&conf, weights, tokenizer);
        let output = runner.generate("Hello world", 15, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(
            s,
            "s \nOn a little by and waly. She want"
        );
        Ok(())
    }

    #[test]
    fn test_gguf_tokenizer() -> Result<()> {
        let loader = Llama2GgufLoader::new("../testdata/tinyllamas-stories-260k-f32.gguf")?;
        let (_, _, tk) = loader.load()?;

        assert_eq!(tk.decode(2, 3)?, "\u{0}");
        assert_eq!(tk.decode(2, 5)?, "\u{2}");
        assert_eq!(tk.decode(2, 6)?, "\u{3}");
        assert_eq!(tk.decode(2, 100)?, "a");

        let tests = vec![
            ("hello, world", "<s> - he - ll - o - , - <0x20> - w - or - ld - </s>"),
            ("tiktok", "<s> - t - i - k - t - o - k - </s>"),
        ];

        for tt in tests {
            let tokens = tk.encode(tt.0, true, true)?;
            let tokens_in_string = tokens
                .iter()
                .map(|t| tk.vocab[*t].clone())
                .collect::<Vec<String>>()
                .join(" - ");
            assert_eq!(tokens_in_string, tt.1, "failed to encode {}", tt.0);
        }
        Ok(())
    }
}
