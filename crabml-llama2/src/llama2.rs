use std::rc::Rc;
use std::vec;

use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGMLType;
use crabml::tensor::RopeMode;
use crabml::tensor::Tensor;
use crabml::tensor::TensorMetrics;
use crabml::tokenizer::BpeTokenizer;

use crate::model::Llama2Config;
use crate::model::Llama2Model;
use crate::model::Llama2Weights;
use crate::model::ModelArchitecture;
use crate::sampler::Llama2Sampler;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Activation {
    SiLU,
    GeLU,
}

pub struct Llama2Runner<T: Tensor> {
    conf: Llama2Config,
    weights: Rc<Llama2Weights<T>>,
    tokenizer: Rc<BpeTokenizer>,
    device: T::Device,
    logits: Vec<f32>,            // output logits (vocab_size, )
    key_cache: Vec<Option<T>>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<Option<T>>, // (layer, seq_len, kv_dim)
    metrics: TensorMetrics,
}

impl<'a, T: Tensor> Llama2Runner<T> {
    pub fn new(
        model: impl Llama2Model<T = T>,
        metrics: TensorMetrics,
        seq_len: usize,
        use_f16_kv_cache: bool,
    ) -> Result<Self> {
        let kv_cache_dtype = if use_f16_kv_cache {
            GGMLType::F16
        } else {
            GGMLType::F32
        };

        let conf = &model.conf();
        let device = model.device().clone();
        let weights = model.weights();
        let tokenizer = model.tokenizer();
        let logits = vec![0.0; conf.vocab_size];
        let key_cache = (0..conf.n_layers)
            .map(|_| {
                T::alloc(
                    &[conf.n_kv_heads, seq_len, conf.head_size()],
                    kv_cache_dtype,
                    device.clone(),
                )
                .map(|t| t.resize(1, 0).unwrap())
                .map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        let value_cache = (0..conf.n_layers)
            .map(|_| {
                T::alloc(
                    &[conf.n_kv_heads, seq_len, conf.head_size()],
                    kv_cache_dtype,
                    device.clone(),
                )
                .map(|t| t.resize(1, 0).unwrap())
                .map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conf: conf.clone(),
            logits,
            key_cache,
            value_cache,
            weights,
            tokenizer,
            device,
            metrics,
        })
    }

    // prefill the model with the prompt, return the next position and the first generated token
    pub fn prefill(
        &mut self,
        prompt: &str,
        sampler: &'a mut Llama2Sampler,
    ) -> Result<(usize, usize, usize)> {
        let prompt_tokens = self.tokenizer.encode(prompt, true, false)?;
        if prompt_tokens.is_empty() {
            return Err(Error {
                kind: ErrorKind::BadInput,
                message: "something is wrong, expected at least 1 prompt token".to_string(),
                cause: None,
            });
        }

        let mut logits: &mut [f32] = &mut [];
        for (pos, token) in prompt_tokens.iter().enumerate() {
            logits = self.forward(*token, pos)?;
        }
        let token = sampler.sample(logits)?;
        let last_token = *prompt_tokens.last().unwrap();

        Ok((prompt_tokens.len(), last_token, token))
    }

    pub fn generate(
        &'a mut self,
        pos: usize,
        prev_token: usize,
        token: usize,
        steps: usize,
        sampler: &'a mut Llama2Sampler,
    ) -> impl Iterator<Item = Result<String>> + '_ {
        let max_steps = (self.conf.seq_len - pos).min(steps);
        let first_token = self.tokenizer.decode(prev_token, token);
        let tokens_iter = (pos..pos + max_steps).scan(token, move |current_token, pos| {
            let logits = self.forward(*current_token, pos).unwrap();
            let new_token = sampler.sample(logits).unwrap();
            if new_token == self.tokenizer.eos_token() {
                return None;
            }
            let r = self.tokenizer.decode(*current_token, new_token).unwrap();
            *current_token = new_token;
            Some(Ok(r))
        });
        std::iter::once(first_token).chain(tokens_iter)
    }

    // simplify the test cases
    pub fn prefill_and_generate(
        &'a mut self,
        prompt: &str,
        steps: usize,
        sampler: &'a mut Llama2Sampler,
    ) -> Result<impl Iterator<Item = Result<String>> + '_> {
        let (pos, prev_token, token) = self.prefill(prompt, sampler)?;
        Ok(self.generate(pos, prev_token, token, steps, sampler))
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&mut [f32]> {
        let _t = self.metrics.forward_walltime.track();

        let x = match self.conf.architecture {
            ModelArchitecture::Llama => self.forward_llama(token, pos)?,
            ModelArchitecture::Gemma => self.forward_gemma(token, pos)?,
        };

        // classifier into logits
        // TODO: it'd be make sense to reuse the same buffer for the logits
        let output_weight = self
            .weights
            .output_weight
            .as_ref()
            .unwrap_or_else(|| &self.weights.token_embed);
        let logits = output_weight.matmul_vec(&x)?; // (vocab_size,
        logits.export(&mut self.logits)?;
        Ok(&mut self.logits)
    }

    fn forward_llama(&mut self, token: usize, pos: usize) -> Result<T> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_dim = self.conf.head_size();
        let rope_dim = self.conf.rope_dim.unwrap_or(head_dim);

        // copy the token embedding into x
        let mut x = T::alloc(&[embed_dim], GGMLType::F32, self.device.clone())?;
        x.copy_rows_from(&self.weights.token_embed, &[token])?;

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
                (q, k, v)
            };

            // ROPE
            let (q, k) = {
                let q = q.reshape(&[n_heads, head_dim])?;
                let k = k.reshape(&[n_kv_heads, head_dim])?;

                let q = q.rope_inplace(RopeMode::Llama, pos, rope_dim)?;
                let k = k.rope_inplace(RopeMode::Llama, pos, rope_dim)?;
                (
                    q.with_name(format!("q_roped:{}:{}", l, pos)),
                    k.with_name(format!("k_roped:{}:{}", l, pos)),
                )
            };

            x = self.forward_multi_query_attention(
                q, k, v, l, pos, n_kv_heads, n_heads, embed_dim, head_dim,
            )?;
            x = x.with_name(format!("attn_out:{}:{}", l, pos));

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;

            // ffn
            x = self.forward_ffn(x, l, Activation::SiLU)?;
            x = x.with_name(format!("ffn_out:{}:{}", l, pos));
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x.with_name(format!("final_rmsnorm:{}", pos))
        };

        Ok(x)
    }

    // The differences between GEMMA and LLAMA are:
    // 1. the way the ROPE is calculated.
    // 2. it uses GELU instead of SiLU.
    // 3. it scales the input embedding with sqrt(embed_dim).
    // 4. it adds a 1.0 to every weights on rmsnorm (rms_att_weight, rms_ffn_weight,
    //    rms_final_weight), this have been processed during GGUF format convert, so we
    //    don't need to do it here.
    fn forward_gemma(&mut self, token: usize, pos: usize) -> Result<T> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_dim = self.conf.head_size();
        let rope_dim = self.conf.rope_dim.unwrap_or(head_dim);

        // copy the token embedding into x
        let mut x = T::alloc(&[embed_dim], GGMLType::F32, self.device.clone())?;
        x.copy_rows_from(&self.weights.token_embed, &[token])?;

        // GEMMA only: scale the embedding with sqrt(embed_dim)
        x = x.scale_inplace((embed_dim as f32).sqrt())?;
        x = x.with_name("scaled_embed".to_string());

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
                (q, k, v)
            };

            // ROPE
            let (q, k) = {
                let q = q.reshape(&[n_heads, head_dim])?;
                let k = k.reshape(&[n_kv_heads, head_dim])?;

                let q = q.rope_inplace(RopeMode::Neox, pos, rope_dim)?;
                let k = k.rope_inplace(RopeMode::Neox, pos, rope_dim)?;
                (
                    q.with_name(format!("q_roped:{}:{}", l, pos)),
                    k.with_name(format!("k_roped:{}:{}", l, pos)),
                )
            };

            x = self.forward_multi_query_attention(
                q, k, v, l, pos, n_kv_heads, n_heads, embed_dim, head_dim,
            )?;

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;

            // ffn
            x = self.forward_ffn(x, l, Activation::GeLU)?;
            x = x.with_name(format!("ffn_out:{}:{}", l, pos));
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x.with_name(format!("final_rmsnorm:{}", pos))
        };

        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_multi_query_attention(
        &mut self,
        q: T,
        k: T,
        v: T,
        l: usize,
        pos: usize,
        n_kv_heads: usize,
        n_heads: usize,
        embed_dim: usize,
        head_dim: usize,
    ) -> Result<T> {
        // save to kv cache in layout of (seq, n_kv_heads, head_dim)
        {
            let _t = self.metrics.save_kvcache_walltime.track();
            let k = k
                .reshape(&[1, n_kv_heads, head_dim])?
                .transpose(&[1, 0, 2])?;
            let v = v
                .reshape(&[1, n_kv_heads, head_dim])?
                .transpose(&[1, 0, 2])?;

            if let Some(k_cache) = self.key_cache[l].as_mut() {
                k_cache.concatenate(&k, 1)?;
            };
            if let Some(v_cache) = self.value_cache[l].as_mut() {
                v_cache.concatenate(&v, 1)?;
            };
        };

        // multi query attention
        let x = {
            let q = q.reshape(&[n_heads, head_dim])?;

            // - key_cache: [n_kv_head, seq, head_size]
            // - q: [n_head, head_size]
            // - attn_score = batch_matmul(key_cache, q) => [n_head, seq]
            // - softmax(attn_score, axis=1) => [n_head, seq]
            // - val_cache: [n_kv_head, seq, head_size]
            // - val_cache = val_cache.transpose(0, 2, 1) => [n_kv_head, head_size, seq]
            // - out = batch_matmul(val_cache, atten_scores) => [n_head, head_size]

            // get attention scores
            let k_cache = self.key_cache[l].take().unwrap();
            let k_cache_strider_orig = k_cache.strider().clone();
            // (n_heads, n_seq, head_size) @ (n_head, head_size) => (n_heads, n_seq)
            let q = q.div_scalar_inplace((head_dim as f32).sqrt())?;
            let attn = k_cache.batch_matmul_vec(&q)?;
            let attn = attn
                .softmax_inplace(1)?
                .with_name(format!("k_cache_attn:{}:{}", l, pos));
            self.key_cache[l].replace(k_cache.with_strider(k_cache_strider_orig)?);

            let v_cache = self.value_cache[l].take().unwrap();
            let v_cache_strider_orig = v_cache.strider().clone();
            // get the weighted sum of the values and attention scores
            let v_cache = v_cache.transpose(&[0, 2, 1])?;
            // (n_kv_heads, head_size, n_seq) @ (n_heads, n_seq) => (n_heads, head_size)
            let x_with_attn = v_cache.batch_matmul_vec(&attn)?; // (n_heads, head_size)
            let x_with_attn = x_with_attn.reshape(&[embed_dim])?;
            self.value_cache[l].replace(v_cache.with_strider(v_cache_strider_orig)?);

            // final matmul to get the output of the attention
            self.weights.wo[l].matmul_vec(&x_with_attn)?
        };
        Ok(x)
    }

    fn forward_ffn(&self, mut x: T, l: usize, activation: Activation) -> Result<T> {
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
        h1 = match activation {
            Activation::SiLU => h1.silu_inplace()?,
            Activation::GeLU => h1.gelu_inplace()?,
        };

        // elementwise multiply with w3(x)
        h1 = h1.mul_inplace(&h2)?;

        // final matmul to get the output of the ffn
        x = self.weights.ffn_down_weight[l].matmul_vec(&h1)?;

        // residual connection
        x = x.add_inplace(&x_orig_ffn)?;
        Ok(x)
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
    use crate::CpuLlama2Model;
    use crate::WgpuLlama2Model;

    #[test]
    fn test_generate_f32() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-f32.gguf")?;
        let gf = gl.open()?;

        let device = CpuTensorDevice::with_options(CpuTensorDeviceOptions {
            debug_named_tensors: false,
        });
        let lm = CpuLlama2Model::load(&gf, device.clone())?;

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0, device.exp_cache());
        let mut runner = Llama2Runner::new(&lm, TensorMetrics::default(), 200, false)?;
        let output = runner.prefill_and_generate("Lily is a cat", 30, &mut sampler)?;
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
        let lm = CpuLlama2Model::load(&gf, device.clone())?;
        assert_eq!(lm.conf.rope_dim, Some(48));
        assert_eq!(lm.conf.head_size(), 48);

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0, device.exp_cache());
        let mut runner = Llama2Runner::new(&lm, TensorMetrics::default(), 200, false)?;
        let output = runner.prefill_and_generate("Lily is a cute cat, ", 10, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }

    #[test]
    fn test_generate_q8_0_with_f16_kvcache() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-q8_0.gguf")?;
        let gf = gl.open()?;

        let device = CpuTensorDevice::new();
        let lm = CpuLlama2Model::load(&gf, device.clone())?;
        assert_eq!(lm.conf.rope_dim, Some(48));
        assert_eq!(lm.conf.head_size(), 48);

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0, device.exp_cache());
        let mut runner = Llama2Runner::new(&lm, TensorMetrics::default(), 200, true)?;
        let output = runner.prefill_and_generate("Lily is a cute cat, ", 10, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }

    #[test]
    fn test_generate_f16() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/TinyLLama-v0-5M-F16.gguf")?;
        let gf = gl.open()?;

        let device = CpuTensorDevice::new();
        let lm = CpuLlama2Model::load(&gf, device.clone())?;
        assert_eq!(lm.conf.rope_dim, Some(4));
        assert_eq!(lm.conf.head_size(), 4);

        let mut sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0, device.exp_cache());
        let mut runner = Llama2Runner::new(&lm, TensorMetrics::default(), 200, false)?;
        let output = runner.prefill_and_generate("Lily is a cute cat, ", 10, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 year old. She likes to play with her friends");
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

        let mut sampler =
            Llama2Sampler::new(model_cpu.conf.vocab_size, 0.0, 0.0, device_cpu.exp_cache());
        let mut runner_cpu = Llama2Runner::new(&model_cpu, TensorMetrics::default(), 200, false)?;
        let mut runner_wgpu = Llama2Runner::new(&model_wgpu, TensorMetrics::default(), 200, false)?;

        let output_cpu = runner_cpu
            .prefill_and_generate("Lily is a cat", 30, &mut sampler)?
            .collect::<Result<Vec<String>>>()?
            .join("");

        let output_wgpu = runner_wgpu
            .prefill_and_generate("Lily is a cat", 30, &mut sampler)?
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
            device_cpu.dump_debug_tensor("attn_out:0:0").unwrap()[..],
            device_wgpu.dump_debug_tensor("attn_out:0:0").unwrap()[..],
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
}
