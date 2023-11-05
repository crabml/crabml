use std::ops::AddAssign;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;
use std::vec;

use crabml::backends::cpu::CpuTensor;
use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::tensor::Tensor;
use crabml::tokenizer::BpeTokenizer;

use crate::model::CpuLlama2Model;
use crate::model::Llama2Config;
use crate::model::Llama2Weights;
use crate::sampler::Llama2Sampler;

pub struct Llama2Runner<T: Tensor> {
    conf: Llama2Config,
    weights: Rc<Llama2Weights<T>>,
    tokenizer: Rc<BpeTokenizer>,
    device: T::Device,
    logits: Vec<f32>,            // output logits (vocab_size, )
    key_cache: Vec<Option<T>>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<Option<T>>, // (layer, seq_len, kv_dim)
}

impl<'a> TryFrom<&'a CpuLlama2Model<'a>> for Llama2Runner<CpuTensor<'a>> {
    type Error = crabml::error::Error;

    fn try_from(model: &'a CpuLlama2Model<'a>) -> Result<Self> {
        let conf = &model.conf;
        let device = model.device.clone();
        let weights = model.weights.clone();
        let tokenizer = model.tokenizer.clone();

        let logits = vec![0.0; conf.vocab_size];
        let key_cache = (0..conf.n_layers)
            .map(|_| {
                CpuTensor::new(
                    vec![],
                    &[0, conf.n_kv_heads, conf.head_size()],
                    device.clone(),
                )
                .map(|t| Some(t))
            })
            .collect::<Result<Vec<_>>>()?;
        let value_cache = (0..conf.n_layers)
            .map(|_| {
                CpuTensor::new(
                    vec![],
                    &[0, conf.n_kv_heads, conf.head_size()],
                    device.clone(),
                )
                .map(|t| Some(t))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            conf: *conf,
            logits,
            key_cache,
            value_cache,
            weights,
            tokenizer,
            device,
        })
    }
}

impl<'a, T: Tensor> Llama2Runner<T> {
    pub fn generate(
        &'a mut self,
        prompt: &str,
        steps: usize,
        sampler: &'a mut Llama2Sampler,
    ) -> Result<Llama2RunnerOutputGenerator<'a, T>> {
        Llama2RunnerOutputGenerator::new(self, sampler, prompt, steps, self.conf.seq_len)
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&mut [f32]> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_size = self.conf.head_size();

        // copy the token embedding into x
        let mut x = T::alloc(&[embed_dim], self.device.clone())?;
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
                let q = q.reshape(&[n_heads, head_size])?;
                let k = k.reshape(&[n_kv_heads, head_size])?;

                let q = q.rope_inplace(pos, self.conf.rope_dim)?;
                let k = k.rope_inplace(pos, self.conf.rope_dim)?;
                (q, k)
            };

            // save to kv cache
            {
                let v = v.reshape(&[n_kv_heads, head_size])?;

                if let Some(ref mut k_cache) = self.key_cache[l] {
                    k_cache.extend(&k)?;
                }
                if let Some(ref mut v_cache) = self.value_cache[l] {
                    v_cache.extend(&v)?;
                }
            };

            // multi query attention
            x = {
                let q = q.reshape(&[n_heads, head_size])?;

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
                let k_cache = self.key_cache[l].take().unwrap();
                let k_cache_strider_orig = k_cache.strider().clone();
                let k_cache = k_cache
                    .repeat(&[1, n_heads / n_kv_heads, 1])?
                    .transpose(&[1, 0, 2])?;
                // (n_heads, n_seq, head_size) @ (n_head, head_size) => (n_heads, n_seq)
                let attn = k_cache.batch_matmul(&q)?;
                let attn = attn.div_scalar_inplace((head_size as f32).sqrt())?;
                let attn = attn.softmax_inplace(1)?;
                self.key_cache[l].replace(k_cache.with_strider(k_cache_strider_orig)?);

                let v_cache = self.value_cache[l].take().unwrap();
                let v_cache_strider_orig = v_cache.strider().clone();
                // get the weighted sum of the values and attention scores
                let v_cache = v_cache
                    .repeat(&[1, n_heads / n_kv_heads, 1])?
                    .transpose(&[1, 2, 0])?;
                // (n_heads, head_size, n_seq) @ (n_heads, n_seq) => (n_heads, head_size)
                let x_with_attn = v_cache.batch_matmul(&attn)?; // (n_heads, head_size)
                let x_with_attn = x_with_attn.reshape(&[embed_dim])?;
                self.value_cache[l].replace(v_cache.with_strider(v_cache_strider_orig)?);

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

        self.logits = logits.export()?.collect::<Vec<_>>();
        Ok(&mut self.logits)
    }
}

pub struct Llama2RunnerOutputGenerator<'a, T: Tensor> {
    pos: usize,
    steps: usize,
    seq_len: usize,
    prompt_tokens: Vec<usize>,
    token: usize,
    sampler: &'a mut Llama2Sampler,
    runner: &'a mut Llama2Runner<T>,
    total_time: Duration,
}

impl<'a, T: Tensor> Llama2RunnerOutputGenerator<'a, T> {
    fn new(
        runner: &'a mut Llama2Runner<T>,
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

impl<'a, T: Tensor> Iterator for Llama2RunnerOutputGenerator<'a, T> {
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
    use crabml::backends::cpu::cpu_tensor::CpuTensorDevice;
    use crabml::gguf::GGUFFileLoader;

    use super::*;

    #[test]
    fn test_generate_f32() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-f32.gguf")?;
        let gf = gl.open()?;
        let lm = CpuLlama2Model::from(&gf)?;

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::try_from(&lm)?;
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
        let lm = CpuLlama2Model::from(&gf)?;
        assert_eq!(lm.conf().rope_dim, 48);
        assert_eq!(lm.conf().head_size(), 48);

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::try_from(&lm)?;
        let output = runner.generate("Lily is a cute cat, ", 10, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }
}
