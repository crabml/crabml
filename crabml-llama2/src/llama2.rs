use std::ops::AddAssign;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;
use std::vec;

use crabml::backends::cpu::CpuTensor;
use crabml::backends::wgpu::WgpuTensor;
use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::tensor::Tensor;
use crabml::tokenizer::BpeTokenizer;

use crate::model::CpuLlama2Model;
use crate::model::Llama2Config;
use crate::model::Llama2Weights;
use crate::model::WgpuLlama2Model;
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
        let seq_len = conf.seq_len;

        let logits = vec![0.0; conf.vocab_size];
        let key_cache = (0..conf.n_layers)
            .map(|_| {
                CpuTensor::alloc(
                    &[0, conf.n_heads, conf.head_size()],
                    Some(seq_len * conf.embedding_dim),
                    device.clone(),
                )
                .map(|t| Some(t))
            })
            .collect::<Result<Vec<_>>>()?;
        let value_cache = (0..conf.n_layers)
            .map(|_| {
                CpuTensor::alloc(
                    &[0, conf.n_heads, conf.head_size()],
                    Some(seq_len * conf.embedding_dim),
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

impl TryFrom<&WgpuLlama2Model> for Llama2Runner<WgpuTensor> {
    type Error = crabml::error::Error;

    fn try_from(model: &WgpuLlama2Model) -> Result<Self> {
        let conf = &model.conf;
        let device = model.device.clone();
        let weights = model.weights.clone();
        let tokenizer = model.tokenizer.clone();
        let logits = vec![0.0; conf.vocab_size];
        let seq_len = conf.seq_len;
        let key_cache = (0..conf.n_layers)
            .map(|_| {
                WgpuTensor::alloc(
                    &[0, conf.n_heads, conf.head_size()],
                    Some(seq_len * conf.embedding_dim),
                    device.clone(),
                )
                .map(|t| Some(t))
            })
            .collect::<Result<Vec<_>>>()?;
        let value_cache = (0..conf.n_layers)
            .map(|_| {
                WgpuTensor::alloc(
                    &[0, conf.n_heads, conf.head_size()],
                    Some(seq_len * conf.embedding_dim),
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
        let head_dim = self.conf.head_size();
        let rope_dim = self.conf.rope_dim.unwrap_or(head_dim);

        // copy the token embedding into x
        let mut x = T::alloc(&[embed_dim], None, self.device.clone())?;
        x.copy_from(&self.weights.token_embed, &[token, 0], embed_dim)?;

        x = x.mul_scalar_inplace((embed_dim as f32).sqrt())?;

        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_attn_orig = x.dup()?;

            // attention rnsnorm
            x = {
                x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
                x = x.mul_inplace(&self.weights.rms_att_weight[l])?;
                x = x.with_name(format!("attn_rmsnorm:{}:{}", l, pos));
                x
            };

            // matmul qkv for every head
            let (q, k, v) = {
                // wq: (embed_dim, embed_dim) @ x (embed_dim, ) => (embed_dim, )
                // wk: (kv_dim, embed_dim) @ x (embed_dim, ) => (kv_dim, )
                // wv: (kv_dim, embed_dim) @ x (embed_dim, ) => (kv_dim, )
                let q = self.weights.wq[l].matmul_vec(&x)?;
                let k = self.weights.wk[l].matmul_vec(&x)?;
                let v = self.weights.wv[l].matmul_vec(&x)?;

                (
                    q.with_name(format!("q:{}:{}", l, pos)),
                    k.with_name(format!("k:{}:{}", l, pos)),
                    v.with_name(format!("v:{}:{}", l, pos)),
                )
            };

            // ROPE
            let (q, k) = {
                let q = q.reshape(&[n_heads, head_dim])?;
                let k = k.reshape(&[n_kv_heads, head_dim])?;

                let q = q.rope_inplace(pos, rope_dim)?;
                let k = k.rope_inplace(pos, rope_dim)?;
                (
                    q.with_name(format!("q_roped:{}:{}", l, pos)),
                    k.with_name(format!("k_roped:{}:{}", l, pos)),
                )
            };

            // save to kv cache
            {
                let v = v
                    .reshape(&[n_kv_heads, head_dim])?
                    .repeat_n(n_heads / n_kv_heads)?;
                let k = k.repeat_n(n_heads / n_kv_heads)?;

                if let Some(ref mut k_cache) = self.key_cache[l] {
                    k_cache.extend(&k)?;
                }
                if let Some(ref mut v_cache) = self.value_cache[l] {
                    v_cache.extend(&v)?;
                }
            };

            // multi query attention
            x = {
                let q = q.reshape(&[n_heads, head_dim])?;

                // - key_cache: [seq, n_head, head_size]
                // - key_cache = key_cache.transpose(1, 0, 2) => [n_head, seq, head_size]
                // - q: [n_head, head_size]
                // - attn_score = batch_matmul(key_cache, q) => [n_head, seq]
                // - softmax(attn_score, axis=1) => [n_head, seq]
                // - val_cache: [seq, n_head, head_size]
                // - val_cache = val_cache.transpose(1, 2, 0) => [n_head, head_size, seq]
                // - out = batch_matmul(val_cache, atten_scores) => [n_head, head_size]

                // get attention scores
                let k_cache = self.key_cache[l].take().unwrap();
                let k_cache_strider_orig = k_cache.strider().clone();
                let k_cache = k_cache.transpose(&[1, 0, 2])?;
                // (n_heads, n_seq, head_size) @ (n_head, head_size) => (n_heads, n_seq)
                let q_scaled = q.mul_scalar_inplace(1.0 / (head_dim as f32).sqrt())?;
                let attn = k_cache.batch_matmul_vec(&q_scaled)?;
                let attn = attn.div_scalar_inplace((head_dim as f32).sqrt())?;
                let attn = attn
                    .softmax_inplace(1)?
                    .with_name(format!("k_cache_attn:{}:{}", l, pos));
                self.key_cache[l].replace(k_cache.with_strider(k_cache_strider_orig)?);

                let v_cache = self.value_cache[l].take().unwrap();
                let v_cache_strider_orig = v_cache.strider().clone();
                // get the weighted sum of the values and attention scores
                let v_cache = v_cache.transpose(&[1, 2, 0])?;
                // (n_heads, head_size, n_seq) @ (n_heads, n_seq) => (n_heads, head_size)
                let x_with_attn = v_cache.batch_matmul_vec(&attn)?; // (n_heads, head_size)
                let x_with_attn = x_with_attn.reshape(&[embed_dim])?;
                self.value_cache[l].replace(v_cache.with_strider(v_cache_strider_orig)?);

                // final matmul to get the output of the attention
                self.weights.wo[l].matmul_vec(&x_with_attn)?
            };

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;

            // ffn
            x = {
                // save for redidual connection
                let x_orig_ffn = x.dup()?;

                // ffn rmsnorm
                x = {
                    x = x.rms_norm_inplace(1e-5)?;
                    x = x.mul_inplace(&self.weights.rms_ffn_weight[l])?;
                    x
                };

                // Now for FFN in PyTorch we have: self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
                // first calculate self.w1(x) and self.w3(x)
                // w1: (hidden_dim, embed_dim) @ x (embed_dim, ) => (hidden_dim, )
                // w3: (hidden_dim, embed_dim) @ x (embed_dim, ) => (hidden_dim, )
                let mut h1 = self.weights.ffn_gate_weight[l].matmul_vec(&x)?;
                let h2 = self.weights.ffn_up_weight[l].matmul_vec(&x)?;

                // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
                h1 = h1.silu_inplace()?;

                // elementwise multiply with w3(x)
                h1 = h1.mul_inplace(&h2)?;

                // final matmul to get the output of the ffn
                x = self.weights.ffn_down_weight[l].matmul_vec(&h1)?;

                // residual connection
                x = x.add_inplace(&x_orig_ffn)?;
                x.with_name(format!("ffn_out:{}:{}", l, pos))
            }
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x.with_name(format!("final_rmsnorm:{}", pos))
        };

        // classifier into logits
        let logits = self
            .weights
            .output_weight
            .as_ref()
            .unwrap_or_else(|| &self.weights.token_embed)
            .matmul_vec(&x)?; // (vocab_size,
        logits.export(&mut self.logits)?;
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
    use approx::assert_relative_eq;
    use crabml::backends::cpu::CpuTensorDevice;
    use crabml::backends::cpu::CpuTensorDeviceOptions;
    use crabml::backends::wgpu::WgpuTensorDevice;
    use crabml::backends::wgpu::WgpuTensorDeviceOptions;
    use crabml::gguf::GGUFFileLoader;

    use super::*;

    #[test]
    fn test_generate_f32() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-f32.gguf")?;
        let gf = gl.open()?;

        let device = CpuTensorDevice::with_options(CpuTensorDeviceOptions {
            debug_named_tensors: false,
        });
        let lm = CpuLlama2Model::load(&gf, device.clone())?;

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

        let device = CpuTensorDevice::new();
        let lm = CpuLlama2Model::load(&gf, device)?;
        assert_eq!(lm.conf().rope_dim, Some(48));
        assert_eq!(lm.conf().head_size(), 48);

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::try_from(&lm)?;
        let output = runner.generate("Lily is a cute cat, ", 10, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }

    #[test]
    fn test_generate_f32_gpu() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-f32.gguf")?;
        let gf = gl.open()?;

        let device_cpu = CpuTensorDevice::with_options(CpuTensorDeviceOptions {
            debug_named_tensors: true,
        });
        let model_cpu = CpuLlama2Model::load(&gf, device_cpu.clone())?;

        let device_wgpu = WgpuTensorDevice::new(
            WgpuTensorDeviceOptions::new()
                .with_staging_buf_bytes(model_cpu.conf.vocab_size * 4)
                .with_debug_named_tensor(true),
        );
        let model_wgpu = WgpuLlama2Model::from_cpu(&model_cpu, device_wgpu.clone())?;

        let mut sampler = Llama2Sampler::new(model_cpu.conf.vocab_size, 0.0, 0.0);
        let mut runner_cpu = Llama2Runner::try_from(&model_cpu)?;
        let mut runner_wgpu = Llama2Runner::try_from(&model_wgpu)?;

        let output_cpu = runner_cpu
            .generate("Lily is a cat", 30, &mut sampler)?
            .collect::<Result<Vec<String>>>()?
            .join("");

        let output_wgpu = runner_wgpu
            .generate("Lily is a cat", 30, &mut sampler)?
            .collect::<Result<Vec<String>>>()?
            .join("");

        assert_relative_eq!(
            device_cpu.dump_debug_tensor("attn_rmsnorm:0:0").unwrap()[0..10],
            device_wgpu.dump_debug_tensor("attn_rmsnorm:0:0").unwrap()[0..10],
            epsilon = 1e-7
        );

        assert_relative_eq!(
            device_cpu.dump_debug_tensor("q_roped:0:0").unwrap()[..],
            device_wgpu.dump_debug_tensor("q_roped:0:0").unwrap()[..],
            epsilon = 1e-2
        );

        assert_relative_eq!(
            device_cpu.dump_debug_tensor("k_cache_attn:0:0").unwrap()[..],
            device_wgpu.dump_debug_tensor("k_cache_attn:0:0").unwrap()[..],
            epsilon = 1e-4
        );

        assert_relative_eq!(
            device_cpu.dump_debug_tensor("ffn_out:0:0").unwrap()[..],
            device_wgpu.dump_debug_tensor("ffn_out:0:0").unwrap()[..],
            epsilon = 1e-4
        );

        assert_relative_eq!(
            device_cpu.dump_debug_tensor("final_rmsnorm:0").unwrap()[..],
            device_wgpu.dump_debug_tensor("final_rmsnorm:0").unwrap()[..],
            epsilon = 1e-2
        );

        assert_eq!(
            output_cpu,
            " who likes to play with yarn. She has many colors of yarn in her box. She likes to make shapes with yarn and show"
        );

        assert_eq!(
            output_wgpu,
            " who likes to play with yarn. She has many colors of yarn in her box. She likes to make shapes with yarn and show"
        );

        Ok(())
    }

    #[test]
    fn test_generate_gemma_q8_0() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/gemma-2b.Q8_0.gguf")?;
        let gf = gl.open()?;

        let device = CpuTensorDevice::new();
        let lm = CpuLlama2Model::load(&gf, device)?;
        assert_eq!(lm.conf().rope_dim, None);
        assert_eq!(lm.conf().n_kv_heads, 1);
        assert_eq!(lm.conf().n_heads, 8);
        assert_eq!(lm.conf().rms_norm_eps, 1e-6);
        assert_eq!(lm.conf().embedding_dim, 2048);
        assert_eq!(lm.conf().head_size(), 256);

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::try_from(&lm)?;
        let output = runner.generate("a cat ", 10, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }
}
