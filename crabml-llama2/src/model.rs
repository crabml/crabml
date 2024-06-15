use std::rc::Rc;
use std::vec;

use crabml::cpu::CpuTensor;
use crabml::cpu::CpuTensorDevice;
use crabml::cpu::CpuTensorDeviceOptions;
use crabml::cpu::CpuTensorDeviceRef;
use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGMLType;
use crabml::gguf::GGUFFile;
use crabml::tensor::Tensor;
use crabml::tensor::TensorMetrics;
use crabml::tokenizer::Tokenizer;

use crate::sampler::Llama2SamplerRef;
use crate::Llama2Sampler;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelArchitecture {
    Llama,
    Gemma,
    Qwen2,
    Phi2,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub architecture: ModelArchitecture,
    pub model_name: String,
    pub chat_template: String,
    pub embedding_dim: usize, // the dim of embedding
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_dim: Option<usize>,
}

impl LlamaConfig {
    pub fn kv_dim(&self) -> usize {
        (self.embedding_dim * self.n_kv_heads) / self.n_heads
    }

    pub fn head_size(&self) -> usize {
        self.embedding_dim / self.n_heads
    }
}

pub struct LlamaWeights<T: Tensor> {
    // token embedding table
    pub token_embed: T, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<T>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<T>, // (layer, dim)
    pub rms_att_bias: Vec<T>,
    // weights for matmuls
    pub wq: Vec<T>, // (layer, embedding_dim, embedding_dim)
    pub wk: Vec<T>, // (layer, kv_dim, embedding_dim)
    pub wv: Vec<T>, // (layer, kv_dim, embedding_dim)
    pub wo: Vec<T>, // (layer, embedding_dim, embedding_dim)
    pub wqkv: Vec<T>,
    pub bq: Vec<T>, // (layer, embedding_dim), bias for q, if not have bias, it's vec![]
    pub bk: Vec<T>, // (layer, embedding_dim), bias for k
    pub bv: Vec<T>, // (layer, embedding_dim), bias for v
    pub bo: Vec<T>,
    pub bqkv: Vec<T>,
    // weights for ffn
    pub ffn_gate_weight: Vec<T>, // (layer, hidden_dim, embedding_dim)
    pub ffn_down_weight: Vec<T>, // (layer, embedding_dim, hidden_dim)
    pub ffn_up_weight: Vec<T>,   // (layer, hidden_dim, embedding_dim)
    pub ffn_down_bias: Vec<T>,
    pub ffn_up_bias: Vec<T>,
    // final rmsnorm
    pub rms_final_weight: T, // (dim, )
    pub rms_final_bias: Option<T>,
    // (optional) classifier weights for the logits, on the last layer
    pub output_weight: Option<T>, // (vocab_size, dim)
}

pub trait LlamaModel {
    type T: Tensor;

    fn conf(&self) -> LlamaConfig;

    fn device(&self) -> <Self::T as Tensor>::DeviceRef;

    fn weights(&self) -> Rc<LlamaWeights<Self::T>>;

    fn tokenizer(&self) -> Rc<Tokenizer>;

    fn sampler(&self) -> Llama2SamplerRef;

    fn metrics(&self) -> &TensorMetrics;
}

pub struct CpuLlamaModel<'a> {
    pub conf: LlamaConfig,
    pub weights: Rc<LlamaWeights<CpuTensor<'a>>>,
    pub tokenizer: Rc<Tokenizer>,
    pub device: CpuTensorDeviceRef<'a>,
    pub sampler: Llama2SamplerRef,
    pub metrics: TensorMetrics,
}

impl<'a> LlamaModel for &CpuLlamaModel<'a> {
    type T = CpuTensor<'a>;

    fn conf(&self) -> LlamaConfig {
        self.conf.clone()
    }

    fn device(&self) -> CpuTensorDeviceRef<'a> {
        self.device.clone()
    }

    fn weights(&self) -> Rc<LlamaWeights<CpuTensor<'a>>> {
        self.weights.clone()
    }

    fn tokenizer(&self) -> Rc<Tokenizer> {
        self.tokenizer.clone()
    }

    fn sampler(&self) -> Llama2SamplerRef {
        self.sampler.clone()
    }

    fn metrics(&self) -> &TensorMetrics {
        &self.metrics
    }
}

pub struct CpuLlamaModelLoader {
    temprature: f32,

    probability: f32,

    device_options: CpuTensorDeviceOptions,
}

impl Default for CpuLlamaModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuLlamaModelLoader {
    pub fn new() -> Self {
        // this default value is suiteable for running tests
        Self {
            temprature: 0.0,
            probability: 0.0,
            device_options: CpuTensorDeviceOptions::default(),
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temprature = temperature;
        self
    }

    pub fn with_probability(mut self, probability: f32) -> Self {
        self.probability = probability;
        self
    }

    pub fn with_thread_num(mut self, thread_num: usize) -> Self {
        self.device_options.thread_num = thread_num;
        self
    }

    pub fn with_device_options(mut self, options: CpuTensorDeviceOptions) -> Self {
        self.device_options = options;
        self
    }

    pub fn load<'a>(self, gf: &'a GGUFFile<'a>) -> Result<CpuLlamaModel<'a>> {
        let device = CpuTensorDevice::with_options(self.device_options.clone());
        let metrics = device.metrics().clone();
        let conf = self.load_config(gf)?;
        let weights = self.load_weights(gf, conf.n_layers, device.clone())?;
        let tokenizer = self.load_tokenizer(gf)?;
        let sampler = Llama2Sampler::new(
            conf.vocab_size,
            self.temprature,
            self.probability,
            device.exp_cache(),
        );
        Ok(CpuLlamaModel {
            conf,
            weights: Rc::new(weights),
            device,
            tokenizer: Rc::new(tokenizer),
            sampler,
            metrics,
        })
    }

    fn load_weights<'a>(
        &self,
        gf: &'a GGUFFile<'a>,
        n_layers: usize,
        device: CpuTensorDeviceRef<'a>,
    ) -> Result<LlamaWeights<CpuTensor<'a>>> {
        // [64 (dim), 512 (vocab_size)]
        let token_embed = self.load_tensor(gf, "token_embd.weight", device.clone())?;
        let mut wq = vec![];
        let mut wk = vec![];
        let mut wv = vec![];
        let mut wo = vec![];
        let mut wqkv = vec![];
        let mut bq = vec![];
        let mut bk = vec![];
        let mut bv = vec![];
        let mut bo = vec![];
        let mut bqkv = vec![];

        let mut ffn_gate_weight = vec![];
        let mut ffn_down_weight = vec![];
        let mut ffn_up_weight = vec![];
        let mut ffn_up_bias = vec![];
        let mut ffn_down_bias = vec![];
        let mut rms_att_weight = vec![];
        let mut rms_ffn_weight = vec![];
        let mut rms_att_bias = vec![];

        match gf.architecture() {
            "llama" | "gemma" => {
                for layer in 0..n_layers {
                    wq.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_q.weight", layer),
                        device.clone(),
                    )?);
                    wk.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_k.weight", layer),
                        device.clone(),
                    )?);
                    wv.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_v.weight", layer),
                        device.clone(),
                    )?);
                    wo.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_output.weight", layer),
                        device.clone(),
                    )?);
                    // (hidden_dim:172, embedding_dim:64)
                    ffn_gate_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_gate.weight", layer),
                        device.clone(),
                    )?);
                    ffn_down_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_down.weight", layer),
                        device.clone(),
                    )?);
                    ffn_up_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_up.weight", layer),
                        device.clone(),
                    )?);
                    rms_att_weight.push(
                        self.load_tensor(
                            gf,
                            &format!("blk.{}.attn_norm.weight", layer),
                            device.clone(),
                        )?
                        .dequantize(GGMLType::F32)?,
                    );
                    rms_ffn_weight.push(
                        self.load_tensor(
                            gf,
                            &format!("blk.{}.ffn_norm.weight", layer),
                            device.clone(),
                        )?
                        .dequantize(GGMLType::F32)?,
                    );
                }
            }
            "qwen2" => {
                for layer in 0..n_layers {
                    wq.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_q.weight", layer),
                        device.clone(),
                    )?);
                    wk.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_k.weight", layer),
                        device.clone(),
                    )?);
                    wv.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_v.weight", layer),
                        device.clone(),
                    )?);
                    wo.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_output.weight", layer),
                        device.clone(),
                    )?);
                    // (hidden_dim:172, embedding_dim:64)
                    ffn_gate_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_gate.weight", layer),
                        device.clone(),
                    )?);
                    ffn_down_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_down.weight", layer),
                        device.clone(),
                    )?);
                    ffn_up_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_up.weight", layer),
                        device.clone(),
                    )?);
                    rms_att_weight.push(
                        self.load_tensor(
                            gf,
                            &format!("blk.{}.attn_norm.weight", layer),
                            device.clone(),
                        )?
                        .dequantize(GGMLType::F32)?,
                    );
                    rms_ffn_weight.push(
                        self.load_tensor(
                            gf,
                            &format!("blk.{}.ffn_norm.weight", layer),
                            device.clone(),
                        )?
                        .dequantize(GGMLType::F32)?,
                    );
                    bq.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_q.bias", layer),
                        device.clone(),
                    )?);
                    bk.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_k.bias", layer),
                        device.clone(),
                    )?);
                    bv.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_v.bias", layer),
                        device.clone(),
                    )?);
                }
            }
            "phi2" => {
                for layer in 0..n_layers {
                    wqkv.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_qkv.weight", layer),
                        device.clone(),
                    )?);
                    bqkv.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_qkv.bias", layer),
                        device.clone(),
                    )?);
                    wo.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_output.weight", layer),
                        device.clone(),
                    )?);
                    bo.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.attn_output.bias", layer),
                        device.clone(),
                    )?);
                    rms_att_weight.push(
                        self.load_tensor(
                            gf,
                            &format!("blk.{}.attn_norm.weight", layer),
                            device.clone(),
                        )?
                        .dequantize(GGMLType::F32)?,
                    );
                    rms_att_bias.push(
                        self.load_tensor(
                            gf,
                            &format!("blk.{}.attn_norm.bias", layer),
                            device.clone(),
                        )?
                        .dequantize(GGMLType::F32)?,
                    );
                    ffn_down_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_down.weight", layer),
                        device.clone(),
                    )?);
                    ffn_up_weight.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_up.weight", layer),
                        device.clone(),
                    )?);
                    ffn_down_bias.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_down.bias", layer),
                        device.clone(),
                    )?);
                    ffn_up_bias.push(self.load_tensor(
                        gf,
                        &format!("blk.{}.ffn_up.bias", layer),
                        device.clone(),
                    )?);
                }
            }
            arch => {
                return Err(Error {
                    kind: ErrorKind::ModelError,
                    message: format!("unsupported architecture {}", arch),
                    cause: None,
                });
            }
        }

        let rms_final_weight = self
            .load_tensor(gf, "output_norm.weight", device.clone())?
            .dequantize(GGMLType::F32)?;
        let rms_final_bias = if gf.architecture() == "phi2" {
            Some(
                self.load_tensor(gf, "output_norm.bias", device.clone())?
                    .dequantize(GGMLType::F32)?,
            )
        } else {
            None
        };

        // in Gemma, the output weight is None
        let output_weight = self.load_tensor_optional(gf, "output.weight", device)?;

        Ok(LlamaWeights {
            token_embed,
            wq,
            wk,
            wv,
            wo,
            wqkv,
            bq,
            bk,
            bv,
            bo,
            bqkv,
            ffn_gate_weight,
            ffn_down_weight,
            ffn_up_weight,
            ffn_down_bias,
            ffn_up_bias,
            rms_att_weight,
            rms_ffn_weight,
            rms_att_bias,
            rms_final_weight,
            rms_final_bias,
            output_weight,
        })
    }

    pub(crate) fn load_tensor_optional<'a>(
        &self,
        gf: &'a GGUFFile<'a>,
        name: &str,
        device: CpuTensorDeviceRef<'a>,
    ) -> Result<Option<CpuTensor<'a>>> {
        let info = match gf.get_tensor_info(name) {
            None => return Ok(None),
            Some(info) => info.clone(),
        };

        // the dimensions stored in GGUF seems in a reverse order of numpy's shape
        let dims = info.dimensions().iter().rev().copied().collect::<Vec<_>>();
        let tensor = CpuTensor::from_bytes(info.data(), info.typ(), &dims, device.clone())?;
        Ok(Some(tensor))
    }

    pub(crate) fn load_tensor<'a>(
        &self,
        gf: &'a GGUFFile<'a>,
        name: &str,
        device: CpuTensorDeviceRef<'a>,
    ) -> Result<CpuTensor<'a>> {
        match self.load_tensor_optional(gf, name, device)? {
            None => Err(Error {
                kind: ErrorKind::TensorNotFound,
                message: format!("failed to find tensor {}", name),
                cause: None,
            }),
            Some(v) => Ok(v),
        }
    }

    fn load_tokenizer(&self, gf: &GGUFFile) -> Result<Tokenizer> {
        // println!("{:?}", gf.metadata().as_hashmap().keys());
        // println!("{:?}", gf.metadata().get_string("tokenizer.ggml.model"));
        let vocab = gf
            .metadata()
            .get_string_array("tokenizer.ggml.tokens")
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let eos_token = gf
            .metadata()
            .get_u32("tokenizer.ggml.eos_token_id")
            .unwrap() as usize;
        let bos_token = gf
            .metadata()
            .get_u32("tokenizer.ggml.bos_token_id")
            .unwrap() as usize;
        let tokenizer_kind = gf
            .metadata()
            .get_string("tokenizer.ggml.model")
            .unwrap()
            .to_string();
        match tokenizer_kind.as_str() {
            "llama" => {
                // it seems that .to_vec() will raise an memory issue but it's ok with
                // iter().cloned().collect(), strange.
                #[allow(clippy::iter_cloned_collect)]
                let vocab_scores = gf
                    .metadata()
                    .get_f32_array("tokenizer.ggml.scores")
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>();
                Ok(Tokenizer::new_llama(
                    vocab,
                    vocab_scores,
                    bos_token,
                    eos_token,
                ))
            }
            "gpt2" => {
                let merges = gf
                    .metadata()
                    .get_string_array("tokenizer.ggml.merges")
                    .unwrap()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                Ok(Tokenizer::new_gpt2(vocab, merges, bos_token, eos_token))
            }
            other => Err(Error::new(
                ErrorKind::IOError,
                format!("unsupported tokenizer {}", other),
            )),
        }
    }

    fn load_config(&self, gf: &GGUFFile) -> Result<LlamaConfig> {
        // let rope_dims = gf.metadata().get_u32("llama.rope.dimension_count").unwrap();
        let (architecture, prefix) = match gf.metadata().get_string("general.architecture").unwrap()
        {
            "llama" => (ModelArchitecture::Llama, "llama"),
            "gemma" => (ModelArchitecture::Gemma, "gemma"),
            "qwen2" => (ModelArchitecture::Qwen2, "qwen2"),
            "phi2" => (ModelArchitecture::Phi2, "phi2"),
            arch => {
                return Err(Error {
                    kind: ErrorKind::ModelError,
                    message: format!("unsupported architecture {}", arch),
                    cause: None,
                });
            }
        };
        let model_name = gf
            .metadata()
            .get_string("general.name")
            .unwrap()
            .to_string();

        let n_heads = gf
            .metadata()
            .get_u32(&format!("{}.attention.head_count", prefix))
            .unwrap() as usize;
        let n_layers = gf
            .metadata()
            .get_u32(&format!("{}.block_count", prefix))
            .unwrap() as usize;
        let hidden_dim = gf
            .metadata()
            .get_u32(&format!("{}.feed_forward_length", prefix))
            .unwrap() as usize;
        let n_kv_heads = gf
            .metadata()
            .get_u32(&format!("{}.attention.head_count_kv", prefix))
            .unwrap() as usize;
        let seq_len = gf
            .metadata()
            .get_u32(&format!("{}.context_length", prefix))
            .unwrap() as usize;
        let vocab_size = gf
            .metadata()
            .get_string_array("tokenizer.ggml.tokens")
            .unwrap()
            .len();
        let chat_template = gf
            .metadata()
            .get_string("tokenizer.chat_template")
            .map(|s| s.to_string())
            .unwrap_or("".to_string());
        let embedding_dim = gf
            .metadata()
            .get_u32(&format!("{}.embedding_length", prefix))
            .unwrap() as usize;
        let rms_norm_eps = if prefix == "phi2" {
            gf.metadata()
                .get_f32(&format!("{}.attention.layer_norm_epsilon", prefix))
                .unwrap()
        } else {
            gf.metadata()
                .get_f32(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
                .unwrap()
        };
        let n_rot = gf
            .metadata()
            .get_u32(&format!("{}.rope.dimension_count", prefix))
            .map(|v| v as usize);

        Ok(LlamaConfig {
            architecture,
            model_name,
            n_heads,
            n_kv_heads,
            n_layers,
            embedding_dim,
            hidden_dim,
            seq_len,
            vocab_size,
            rms_norm_eps,
            rope_dim: n_rot,
            chat_template,
        })
    }
}

#[derive(Clone)]
pub struct GpuLlamaModel<T: Tensor> {
    pub conf: LlamaConfig,
    pub weights: Rc<LlamaWeights<T>>,
    pub tokenizer: Rc<Tokenizer>,
    pub device: T::DeviceRef,
    pub sampler: Llama2SamplerRef,
    pub metrics: TensorMetrics,
}

impl<T: Tensor> LlamaModel for &GpuLlamaModel<T> {
    type T = T;

    fn conf(&self) -> LlamaConfig {
        self.conf.clone()
    }

    fn weights(&self) -> Rc<LlamaWeights<Self::T>> {
        self.weights.clone()
    }

    fn device(&self) -> T::DeviceRef {
        self.device.clone()
    }

    fn tokenizer(&self) -> Rc<Tokenizer> {
        self.tokenizer.clone()
    }

    fn sampler(&self) -> Llama2SamplerRef {
        self.sampler.clone()
    }

    fn metrics(&self) -> &TensorMetrics {
        &self.metrics
    }
}

impl<T: Tensor> GpuLlamaModel<T> {
    pub fn from_cpu(cpu_model: &CpuLlamaModel, device: T::DeviceRef) -> Result<Self> {
        let weights = Self::convert_cpu_weights(&cpu_model.weights, device.clone())?;
        Ok(Self {
            conf: cpu_model.conf.clone(),
            weights: Rc::new(weights),
            tokenizer: cpu_model.tokenizer.clone(),
            sampler: cpu_model.sampler.clone(),
            metrics: cpu_model.metrics.clone(),
            device,
        })
    }

    fn convert_cpu_weights(
        weights: &LlamaWeights<CpuTensor>,
        device: T::DeviceRef,
    ) -> Result<LlamaWeights<T>> {
        let token_embedding_table = Self::convert_cpu_tensor(&weights.token_embed, device.clone())?;
        let wq = weights
            .wq
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let wk = weights
            .wk
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let wv = weights
            .wv
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let wo = weights
            .wo
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let wqkv = weights
            .wqkv
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let bq = weights
            .bq
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let bk = weights
            .bk
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let bv = weights
            .bv
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let bo = weights
            .bo
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let bqkv = weights
            .bqkv
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let ffn_gate_weight = weights
            .ffn_gate_weight
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let ffn_down_weight = weights
            .ffn_down_weight
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let ffn_up_weight = weights
            .ffn_up_weight
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let ffn_down_bias = weights
            .ffn_down_bias
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let ffn_up_bias = weights
            .ffn_up_bias
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let rms_att_weight = weights
            .rms_att_weight
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let rms_att_bias = weights
            .rms_att_bias
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let rms_ffn_weight = weights
            .rms_ffn_weight
            .iter()
            .map(|t| Self::convert_cpu_tensor(t, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        let rms_final_weight = Self::convert_cpu_tensor(&weights.rms_final_weight, device.clone())?;
        let rms_final_bias = weights.rms_final_bias.as_ref().map(|rms_final_bias| {
            Self::convert_cpu_tensor(rms_final_bias, device.clone()).unwrap()
        });
        let wcls = weights
            .output_weight
            .as_ref()
            .map(|output_weight| Self::convert_cpu_tensor(output_weight, device.clone()).unwrap());
        let weights = LlamaWeights {
            token_embed: token_embedding_table,
            wq,
            wk,
            wv,
            wo,
            wqkv,
            bq,
            bk,
            bv,
            bo,
            bqkv,
            ffn_gate_weight,
            ffn_down_weight,
            ffn_up_weight,
            ffn_down_bias,
            ffn_up_bias,
            rms_att_weight,
            rms_ffn_weight,
            rms_att_bias,
            rms_final_weight,
            rms_final_bias,
            output_weight: wcls,
        };
        Ok(weights)
    }

    fn convert_cpu_tensor(tensor: &CpuTensor, device: T::DeviceRef) -> Result<T> {
        let buf = tensor.buf();
        match tensor.dtype() {
            GGMLType::F32 => (),
            _ => {
                return Err(Error {
                    kind: ErrorKind::TensorError,
                    message: format!("unsupported tensor type on gpu {:?}", buf),
                    cause: None,
                });
            }
        };

        let gpu_tensor = T::from_cpu(
            buf.as_bytes(),
            tensor.shape(),
            GGMLType::F32,
            device.clone(),
        )?;
        Ok(gpu_tensor)
    }
}

#[cfg(test)]
mod tests {
    use crabml::error::Result;
    use crabml::gguf::GGMLType;
    use crabml::gguf::GGUFFileLoader;
    use crabml::tensor::Tensor;

    use crate::model::CpuLlamaModelLoader;

    #[test]
    fn test_load_q8_0() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-q8_0.gguf", false)?;
        let gf = gl.open()?;

        let lm = CpuLlamaModelLoader::new().load(&gf)?;
        assert_eq!(lm.conf.vocab_size, 32000);
        assert_eq!(lm.weights.wk[0].dtype(), GGMLType::Q8_0);
        assert_eq!(lm.weights.rms_att_weight[0].dtype(), GGMLType::F32);
        assert_eq!(lm.weights.rms_ffn_weight[0].dtype(), GGMLType::F32);
        assert_eq!(lm.weights.rms_final_weight.dtype(), GGMLType::F32);
        assert_eq!(lm.weights.token_embed.dtype(), GGMLType::Q8_0);
        Ok(())
    }
}
