const GGUF_MAGIC: u64 = 0x46554747;
const GGUF_VERSION: u64 = 2;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// General
const KEY_GENERAL_ARCHITECTURE: &str = "general.architecture";
const KEY_GENERAL_QUANTIZATION_VERSION: &str = "general.quantization_version";
const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";
const KEY_GENERAL_NAME: &str = "general.name";
const KEY_GENERAL_AUTHOR: &str = "general.author";
const KEY_GENERAL_URL: &str = "general.url";
const KEY_GENERAL_DESCRIPTION: &str = "general.description";
const KEY_GENERAL_LICENSE: &str = "general.license";
const KEY_GENERAL_SOURCE_URL: &str = "general.source.url";
const KEY_GENERAL_SOURCE_HF_REPO: &str = "general.source.hugginface.repository";
const KEY_GENERAL_FILE_TYPE: &str = "general.file_type";

// LLM
const KEY_CONTEXT_LENGTH: &str = "{arch}.context_length";
const KEY_EMBEDDING_LENGTH: &str = "{arch}.embedding_length";
const KEY_BLOCK_COUNT: &str = "{arch}.block_count";
const KEY_FEED_FORWARD_LENGTH: &str = "{arch}.feed_forward_length";
const KEY_USE_PARALLEL_RESIDUAL: &str = "{arch}.use_parallel_residual";
const KEY_TENSOR_DATA_LAYOUT: &str = "{arch}.tensor_data_layout";

// Attention
const KEY_ATTENTION_HEAD_COUNT: &str = "{arch}.attention.head_count";
const KEY_ATTENTION_HEAD_COUNT_KV: &str = "{arch}.attention.head_count_kv";
const KEY_ATTENTION_MAX_ALIBI_BIAS: &str = "{arch}.attention.max_alibi_bias";
const KEY_ATTENTION_CLAMP_KQV: &str = "{arch}.attention.clamp_kqv";
const KEY_ATTENTION_LAYERNORM_EPS: &str = "{arch}.attention.layer_norm_epsilon";
const KEY_ATTENTION_LAYERNORM_RMS_EPS: &str = "{arch}.attention.layer_norm_rms_epsilon";

// RoPE
const KEY_ROPE_DIMENSION_COUNT: &str = "{arch}.rope.dimension_count";
const KEY_ROPE_FREQ_BASE: &str = "{arch}.rope.freq_base";
const KEY_ROPE_SCALE_LINEAR: &str = "{arch}.rope.scale_linear";

// Tokenization
const KEY_TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
const KEY_TOKENIZER_LIST: &str = "tokenizer.ggml.tokens";
const KEY_TOKENIZER_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
const KEY_TOKENIZER_SCORES: &str = "tokenizer.ggml.scores";
const KEY_TOKENIZER_MERGES: &str = "tokenizer.ggml.merges";
const KEY_TOKENIZER_BOS_ID: &str = "tokenizer.ggml.bos_token_id";
const KEY_TOKENIZER_EOS_ID: &str = "tokenizer.ggml.eos_token_id";
const KEY_TOKENIZER_UNK_ID: &str = "tokenizer.ggml.unknown_token_id";
const KEY_TOKENIZER_SEP_ID: &str = "tokenizer.ggml.seperator_token_id";
const KEY_TOKENIZER_PAD_ID: &str = "tokenizer.ggml.padding_token_id";
const KEY_TOKENIZER_HF_JSON: &str = "tokenizer.huggingface.json";
const KEY_TOKENIZER_RWKV: &str = "tokenizer.rwkv.world";

#[derive(Debug)]
pub enum ModelArch {
    Llama = 0,
    Falcon = 1,
    GPT2 = 2,
    GPTJ = 3,
    GPTNEOX = 4,
    MPT = 5,
}

#[derive(Debug)]
pub enum MODEL_TENSOR {
    TOKEN_EMBD = 0,
    POS_EMBD = 1,
    OUTPUT = 2,
    OUTPUT_NORM = 3,
    ROPE_FREQS = 4,
    ATTN_Q = 5,
    ATTN_K = 6,
    ATTN_V = 7,
    ATTN_QKV = 8,
    ATTN_OUT = 9,
    ATTN_NORM = 10,
    ATTN_NORM_2 = 11,
    ATTN_ROT_EMBD = 12,
    FFN_GATE = 13,
    FFN_DOWN = 14,
    FFN_UP = 15,
    FFN_NORM = 16,
}