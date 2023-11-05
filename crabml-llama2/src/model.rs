use std::rc::Rc;
use std::vec;

use crabml::backends::cpu::cpu_tensor::CpuTensorPool;
use crabml::backends::cpu::cpu_tensor::CpuTensorPoolRef;
use crabml::backends::cpu::CpuTensor;
use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGUFFile;
use crabml::gguf::GGUFMetadata;
use crabml::tensor::Tensor;
use crabml::tokenizer::BpeTokenizer;

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

    pub fn head_size(&self) -> usize {
        self.embedding_dim / self.n_heads
    }
}

pub struct Llama2Weights<T: Tensor> {
    // token embedding table
    pub token_embedding_table: T, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<T>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<T>, // (layer, dim)
    // weights for matmuls
    pub wq: Vec<T>, // (layer, embedding_dim, embedding_dim)
    pub wk: Vec<T>, // (layer, kv_dim, embedding_dim)
    pub wv: Vec<T>, // (layer, kv_dim, embedding_dim)
    pub wo: Vec<T>, // (layer, embedding_dim, embedding_dim)
    // weights for ffn
    pub w1: Vec<T>, // (layer, hidden_dim, embedding_dim)
    pub w2: Vec<T>, // (layer, embedding_dim, hidden_dim)
    pub w3: Vec<T>, // (layer, hidden_dim, embedding_dim)
    // final rmsnorm
    pub rms_final_weight: T, // (dim, )
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: T, // (vocab_size, dim)
}

pub struct CpuLlama2Model<'a> {
    pub conf: Llama2Config,
    pub weights: Rc<Llama2Weights<CpuTensor<'a>>>,
    pub tokenizer: Rc<BpeTokenizer>,
    pub pool: CpuTensorPoolRef<'a>,
    pub metadata: &'a GGUFMetadata<'a>,
}

impl<'a> CpuLlama2Model<'a> {
    pub fn from(gf: &'a GGUFFile<'a>) -> Result<Self> {
        let pool = CpuTensorPool::new();
        let conf = Self::load_config(gf);
        let weights = Self::load_weights(gf, conf.n_layers, pool.clone())?;
        let tokenizer = Self::load_tokenizer(gf);
        Ok(Self {
            conf,
            weights: Rc::new(weights),
            pool,
            tokenizer: Rc::new(tokenizer),
            metadata: gf.metadata(),
        })
    }

    pub fn conf(&self) -> &Llama2Config {
        &self.conf
    }

    pub fn weights(&self) -> Rc<Llama2Weights<CpuTensor<'a>>> {
        self.weights.clone()
    }

    pub fn metadata(&self) -> &'a GGUFMetadata<'a> {
        self.metadata
    }

    pub fn tokenizer(&self) -> Rc<BpeTokenizer> {
        self.tokenizer.clone()
    }

    fn load_weights(
        gf: &'a GGUFFile<'a>,
        n_layers: usize,
        pool: CpuTensorPoolRef<'a>,
    ) -> Result<Llama2Weights<CpuTensor<'a>>> {
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
                &format!("blk.{}.attn_q.weight", layer),
                pool.clone(),
            )?);
            wk.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_k.weight", layer),
                pool.clone(),
            )?);
            wv.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_v.weight", layer),
                pool.clone(),
            )?);
            wo.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_output.weight", layer),
                pool.clone(),
            )?);
            // (hidden_dim:172, embedding_dim:64)
            w1.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_gate.weight", layer),
                pool.clone(),
            )?);
            w2.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_down.weight", layer),
                pool.clone(),
            )?);
            w3.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_up.weight", layer),
                pool.clone(),
            )?);
            rms_att_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.attn_norm.weight", layer),
                pool.clone(),
            )?);
            rms_ffn_weight.push(Self::load_tensor(
                gf,
                &format!("blk.{}.ffn_norm.weight", layer),
                pool.clone(),
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

    pub(crate) fn load_tensor(
        gf: &'a GGUFFile<'a>,
        name: &str,
        pool: CpuTensorPoolRef<'a>,
    ) -> Result<CpuTensor<'a>> {
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
