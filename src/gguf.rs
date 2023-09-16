use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::mem;

use int_enum::IntEnum;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u64 = 2;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// General
pub const KEY_GENERAL_ARCHITECTURE: &str = "general.architecture";
pub const KEY_GENERAL_QUANTIZATION_VERSION: &str = "general.quantization_version";
pub const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";
pub const KEY_GENERAL_NAME: &str = "general.name";
pub const KEY_GENERAL_AUTHOR: &str = "general.author";
pub const KEY_GENERAL_URL: &str = "general.url";
pub const KEY_GENERAL_DESCRIPTION: &str = "general.description";
pub const KEY_GENERAL_LICENSE: &str = "general.license";
pub const KEY_GENERAL_SOURCE_URL: &str = "general.source.url";
pub const KEY_GENERAL_SOURCE_HF_REPO: &str = "general.source.hugginface.repository";
pub const KEY_GENERAL_FILE_TYPE: &str = "general.file_type";

// LLM
pub const KEY_CONTEXT_LENGTH: &str = "{arch}.context_length";
pub const KEY_EMBEDDING_LENGTH: &str = "{arch}.embedding_length";
pub const KEY_BLOCK_COUNT: &str = "{arch}.block_count";
pub const KEY_FEED_FORWARD_LENGTH: &str = "{arch}.feed_forward_length";
pub const KEY_USE_PARALLEL_RESIDUAL: &str = "{arch}.use_parallel_residual";
pub const KEY_TENSOR_DATA_LAYOUT: &str = "{arch}.tensor_data_layout";

// Attention
pub const KEY_ATTENTION_HEAD_COUNT: &str = "{arch}.attention.head_count";
pub const KEY_ATTENTION_HEAD_COUNT_KV: &str = "{arch}.attention.head_count_kv";
pub const KEY_ATTENTION_MAX_ALIBI_BIAS: &str = "{arch}.attention.max_alibi_bias";
pub const KEY_ATTENTION_CLAMP_KQV: &str = "{arch}.attention.clamp_kqv";
pub const KEY_ATTENTION_LAYERNORM_EPS: &str = "{arch}.attention.layer_norm_epsilon";
pub const KEY_ATTENTION_LAYERNORM_RMS_EPS: &str = "{arch}.attention.layer_norm_rms_epsilon";

// RoPE
pub const KEY_ROPE_DIMENSION_COUNT: &str = "{arch}.rope.dimension_count";
pub const KEY_ROPE_FREQ_BASE: &str = "{arch}.rope.freq_base";
pub const KEY_ROPE_SCALE_LINEAR: &str = "{arch}.rope.scale_linear";

// Tokenization
pub const KEY_TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
pub const KEY_TOKENIZER_LIST: &str = "tokenizer.ggml.tokens";
pub const KEY_TOKENIZER_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
pub const KEY_TOKENIZER_SCORES: &str = "tokenizer.ggml.scores";
pub const KEY_TOKENIZER_MERGES: &str = "tokenizer.ggml.merges";
pub const KEY_TOKENIZER_BOS_ID: &str = "tokenizer.ggml.bos_token_id";
pub const KEY_TOKENIZER_EOS_ID: &str = "tokenizer.ggml.eos_token_id";
pub const KEY_TOKENIZER_UNK_ID: &str = "tokenizer.ggml.unknown_token_id";
pub const KEY_TOKENIZER_SEP_ID: &str = "tokenizer.ggml.seperator_token_id";
pub const KEY_TOKENIZER_PAD_ID: &str = "tokenizer.ggml.padding_token_id";
pub const KEY_TOKENIZER_HF_JSON: &str = "tokenizer.huggingface.json";
pub const KEY_TOKENIZER_RWKV: &str = "tokenizer.rwkv.world";

#[derive(Debug)]
pub enum GGUFTensorKey {
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

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntEnum)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    // k-quantizations
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    COUNT = 19,
}

impl TryFrom<u32> for GGMLType {
    type Error = GGUFError;

    fn try_from(v: u32) -> std::result::Result<Self, Self::Error> {
        Self::from_int(v).map_err(|err| GGUFError {
            kind: GGUFErrorKind::FormatError,
            message: format!("failed to decode the ggml type for {}", v),
            cause: Some(Box::new(err)),
        })
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IntEnum)]
pub enum GGUFMetadataValueType {
    // The value is a 8-bit unsigned integer.
    U8 = 0,
    // The value is a 8-bit signed little-endian integer.
    I8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    U16 = 2,
    // The value is a 16-bit signed little-endian integer.
    I16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    U32 = 4,
    // The value is a 32-bit signed little-endian integer.
    I32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    F32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array = 9,
    // The value is a 64-bit unsigned little-endian integer.
    U64 = 10,
    // The value is a 64-bit signed little-endian integer.
    I64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    F64 = 12,
}

impl TryFrom<u32> for GGUFMetadataValueType {
    type Error = GGUFError;

    fn try_from(v: u32) -> std::result::Result<Self, Self::Error> {
        Self::from_int(v).map_err(|err| GGUFError {
            kind: GGUFErrorKind::FormatError,
            message: format!("failed to decode the value type for {}", v),
            cause: Some(Box::new(err)),
        })
    }
}

#[derive(Debug, Clone)]
pub enum GGUFMetadataValue<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(u8),
    String(&'a str),
    Array(GGUFMetadataArray<'a>),
}

#[derive(Debug, Clone)]
pub enum GGUFMetadataArray<'a> {
    U8Array(&'a [u8]),
    I8Array(&'a [i8]),
    U16Array(&'a [u16]),
    I16Array(&'a [i16]),
    U32Array(&'a [u32]),
    I32Array(&'a [i32]),
    U64Array(&'a [u64]),
    I64Array(&'a [i64]),
    F32Array(&'a [f32]),
    F64Array(&'a [f64]),
    BoolArray(&'a [u8]),
    StringArray(Vec<&'a str>),
    NestedArray(Vec<GGUFMetadataArray<'a>>),
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum GGUFErrorKind {
    Unexpected,
    IOError,
    FormatError,
}

#[derive(Debug)]
pub struct GGUFError {
    kind: GGUFErrorKind,
    message: String,
    cause: Option<Box<dyn std::error::Error>>,
}

impl std::fmt::Display for GGUFError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message);
        if let Some(cause) = self.cause.as_ref() {
            write!(f, "\ncaused by: {}", cause);
        }
        Ok(())
    }
}

impl std::error::Error for GGUFError {}

pub type Result<T> = std::result::Result<T, GGUFError>;

pub struct GGUFBufReader<'a> {
    cursor: &'a [u8],
    read_bytes: usize,
}

impl<'a> GGUFBufReader<'a> {
    pub fn new(buf: &'a [u8]) -> GGUFBufReader {
        GGUFBufReader {
            cursor: buf,
            read_bytes: 0,
        }
    }

    pub fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.cursor.len() {
            return Err(GGUFError {
                kind: GGUFErrorKind::FormatError,
                message: format!(
                    "failed to read {} bytes from the buffer, only {} bytes left",
                    n,
                    self.cursor.len()
                ),
                cause: None,
            });
        }
        let v = &self.cursor[0..n];
        self.cursor = &self.cursor[n..];
        self.read_bytes += n;
        Ok(v)
    }

    pub fn cursor(&self) -> &'a [u8] {
        self.cursor
    }

    pub fn read_bytes(&self) -> usize {
        self.read_bytes
    }
}

pub struct GGUFMetadataReader<'a, 'b>
where
    'a: 'b,
{
    buf: &'b mut GGUFBufReader<'a>,
    version: u32,
}

macro_rules! define_gguf_metadata_value_read_fn {
    ($read_array_func:ident, $read_item_func:ident, $typ:ty) => {
        fn $read_array_func(&mut self, n: usize) -> Result<&'a [$typ]> {
            let typ_size = mem::size_of::<$typ>();
            let data = self.buf.read(n * typ_size)?;
            let transmuted_data = unsafe {
                assert!(data.len() % typ_size == 0);
                let ptr = data.as_ptr();
                mem::transmute(std::slice::from_raw_parts(ptr, data.len() / typ_size))
            };
            Ok(transmuted_data)
        }

        fn $read_item_func(&mut self) -> Result<$typ> {
            let arr = self.$read_array_func(1)?;
            Ok(arr[0])
        }
    };
}

impl<'a, 'b> GGUFMetadataReader<'a, 'b> {
    pub fn new(buf: &'b mut GGUFBufReader<'a>, version: u32) -> GGUFMetadataReader<'a, 'b> {
        GGUFMetadataReader { buf, version }
    }

    pub fn read_value(&mut self) -> Result<GGUFMetadataValue<'a>> {
        let n = self.read_u32()?;
        let typ = GGUFMetadataValueType::try_from(n)?;
        let v = match typ {
            GGUFMetadataValueType::U8 => GGUFMetadataValue::U8(self.read_u8()?),
            GGUFMetadataValueType::I8 => GGUFMetadataValue::I8(self.read_i8()?),
            GGUFMetadataValueType::U16 => GGUFMetadataValue::U16(self.read_u16()?),
            GGUFMetadataValueType::I16 => GGUFMetadataValue::I16(self.read_i16()?),
            GGUFMetadataValueType::U32 => GGUFMetadataValue::U32(self.read_u32()?),
            GGUFMetadataValueType::I32 => GGUFMetadataValue::I32(self.read_i32()?),
            GGUFMetadataValueType::F32 => GGUFMetadataValue::F32(self.read_f32()?),
            GGUFMetadataValueType::F64 => GGUFMetadataValue::F64(self.read_f64()?),
            GGUFMetadataValueType::U64 => GGUFMetadataValue::U64(self.read_u64()?),
            GGUFMetadataValueType::I64 => GGUFMetadataValue::I64(self.read_i64()?),
            GGUFMetadataValueType::String => GGUFMetadataValue::String(self.read_string()?),
            GGUFMetadataValueType::Bool => GGUFMetadataValue::Bool(self.read_u8()?),
            GGUFMetadataValueType::Array => GGUFMetadataValue::Array(self.read_array()?),
        };
        Ok(v)
    }

    pub fn read_array(&mut self) -> Result<GGUFMetadataArray<'a>> {
        let n = self.read_u32()?;
        let typ = GGUFMetadataValueType::try_from(n)?;
        let len = self.read_len()?;
        let arr = match typ {
            GGUFMetadataValueType::U8 => GGUFMetadataArray::U8Array(self.read_u8_array(len)?),
            GGUFMetadataValueType::I8 => GGUFMetadataArray::I8Array(self.read_i8_array(len)?),
            GGUFMetadataValueType::U16 => GGUFMetadataArray::U16Array(self.read_u16_array(len)?),
            GGUFMetadataValueType::I16 => GGUFMetadataArray::I16Array(self.read_i16_array(len)?),
            GGUFMetadataValueType::U32 => GGUFMetadataArray::U32Array(self.read_u32_array(len)?),
            GGUFMetadataValueType::I32 => GGUFMetadataArray::I32Array(self.read_i32_array(len)?),
            GGUFMetadataValueType::F32 => GGUFMetadataArray::F32Array(self.read_f32_array(len)?),
            GGUFMetadataValueType::F64 => GGUFMetadataArray::F64Array(self.read_f64_array(len)?),
            GGUFMetadataValueType::U64 => GGUFMetadataArray::U64Array(self.read_u64_array(len)?),
            GGUFMetadataValueType::I64 => GGUFMetadataArray::I64Array(self.read_i64_array(len)?),
            GGUFMetadataValueType::Bool => GGUFMetadataArray::BoolArray(self.read_u8_array(len)?),
            GGUFMetadataValueType::String => {
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    v.push(self.read_string()?);
                }
                GGUFMetadataArray::StringArray(v)
            }
            GGUFMetadataValueType::Array => {
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    v.push(self.read_array()?);
                }
                GGUFMetadataArray::NestedArray(v)
            }
        };
        Ok(arr)
    }

    define_gguf_metadata_value_read_fn!(read_u8_array, read_u8, u8);
    define_gguf_metadata_value_read_fn!(read_i8_array, read_i8, i8);
    define_gguf_metadata_value_read_fn!(read_u16_array, read_u16, u16);
    define_gguf_metadata_value_read_fn!(read_i16_array, read_i16, i16);
    define_gguf_metadata_value_read_fn!(read_u32_array, read_u32, u32);
    define_gguf_metadata_value_read_fn!(read_i32_array, read_i32, i32);
    define_gguf_metadata_value_read_fn!(read_u64_array, read_u64, u64);
    define_gguf_metadata_value_read_fn!(read_i64_array, read_i64, i64);
    define_gguf_metadata_value_read_fn!(read_f32_array, read_f32, f32);
    define_gguf_metadata_value_read_fn!(read_f64_array, read_f64, f64);

    pub fn read_string(&mut self) -> Result<&'a str> {
        let len = self.read_len()?;
        let buf = self.buf.read(len)?;
        let s = std::str::from_utf8(buf).map_err(|e| GGUFError {
            kind: GGUFErrorKind::FormatError,
            message: format!("Invalid UTF-8 string"),
            cause: Some(Box::new(e)),
        });
        s
    }

    /// Read the length for string & array. It would be an 32 bit unsigned integer on spec v1, but 64 
    /// bit on spec v2. This seems to be the only difference between v1 and v2, for more infomation:
    /// https://github.com/philpax/ggml/commit/b021b2577d4294800ece200c9f26c9c65b0f6f51
    fn read_len(&mut self) -> Result<usize> {
        let v = if self.version == 1 {
            self.read_u32()? as usize
        } else if self.version == 2 {
            self.read_u64()? as usize
        } else {
            panic!("unsupported version: {}", self.version);
        };
        Ok(v)
    }
}

pub struct GGUFHeader<'a> {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    magic: u32,

    // The version of the format implemented.
    // Must be `2` for version described in this spec.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    version: u32,

    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    tensor_count: u64,

    // The metadata key-value pairs.
    metadata_kv: HashMap<String, GGUFMetadataValue<'a>>,
}

impl<'a> GGUFHeader<'a> {
    fn decode(buf: &mut GGUFBufReader<'a>) -> Result<Self> {
        let mut r = GGUFMetadataReader::new(buf, 2);
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GGUFError {
                kind: GGUFErrorKind::FormatError,
                message: format!("Invalid magic number: {}", magic),
                cause: None,
            });
        }

        let version = r.read_u32()?;
        if version != 2 || version != 1 {
            return Err(GGUFError {
                kind: GGUFErrorKind::FormatError,
                message: format!(
                    "Unsupported version number: {}, only 1 & 2 is supported yet",
                    version
                ),
                cause: None,
            });
        }

        let tensor_count = r.read_u64()?;
        let metadata_kv_count = r.read_u64()?;
        let mut metadata_kv = HashMap::new();

        for _ in 0..metadata_kv_count {
            let key = r.read_string()?;
            let value = r.read_value()?;
            metadata_kv.insert(key.to_string(), value);
        }

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv,
        })
    }

    /// the global alignment to use, as described above. This can vary to allow for different alignment schemes,
    /// but it must be a multiple of 8. Some writers may not write the alignment. If the alignment is not specified,
    /// assume it is 32.
    pub fn alignment(&self) -> u64 {
        match self.metadata_kv.get(KEY_GENERAL_ALIGNMENT) {
            Some(GGUFMetadataValue::U64(v)) => *v,
            Some(GGUFMetadataValue::U32(v)) => *v as u64,
            Some(GGUFMetadataValue::U16(v)) => *v as u64,
            Some(GGUFMetadataValue::U8(v)) => *v as u64,
            Some(GGUFMetadataValue::I64(v)) if *v > 0 => *v as u64,
            Some(GGUFMetadataValue::I32(v)) if *v > 0 => *v as u64,
            Some(GGUFMetadataValue::I16(v)) if *v > 0 => *v as u64,
            Some(GGUFMetadataValue::I8(v)) if *v > 0 => *v as u64,
            _ => GGUF_DEFAULT_ALIGNMENT,
        }
    }

    /// describes what architecture this model implements. All lowercase ASCII, with only [a-z0-9]+ characters
    /// allowed. Known values include:
    ///
    /// - llama
    /// - mpt
    /// - gptneox
    /// - gptj
    /// - gpt2
    /// - bloom
    /// - falcon
    /// - rwkv
    ///
    pub fn architecture(&self) -> Result<&'a str> {
        match self.metadata_kv.get(KEY_GENERAL_ARCHITECTURE) {
            Some(GGUFMetadataValue::String(v)) => Ok(v),
            _ => Err(GGUFError {
                kind: GGUFErrorKind::FormatError,
                message: format!("Missing string metadata general.architecture"),
                cause: None,
            }),
        }
    }

    /// The version of the quantization format. Not required if the model is not quantized (i.e. no tensors are
    /// quantized). If any tensors are quantized, this must be present. This is separate to the quantization
    /// scheme of the tensors itself; the quantization version may change without changing the scheme's name
    /// (e.g. the quantization scheme is Q5_K, and the quantization version is 4).
    pub fn quantization_version(&self) -> Result<u32> {
        match self.metadata_kv.get(KEY_GENERAL_ARCHITECTURE) {
            Some(GGUFMetadataValue::U32(v)) => Ok(*v),
            _ => Err(GGUFError {
                kind: GGUFErrorKind::FormatError,
                message: format!("Missing U32 metadata general.architecture"),
                cause: None,
            }),
        }
    }

    pub fn get_metadata(&self, key: &str) -> Option<&GGUFMetadataValue<'a>> {
        self.metadata_kv.get(key)
    }
}

struct GGUFOnDiskTensorInfo {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    name: String,
    // The dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    dimensions: Vec<u64>,
    // The type of the tensor.
    typ: GGMLType,
    // The offset of the tensor's data in this file in bytes.
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    // Must be a multiple of `ALIGNMENT`.
    offset: u64,
}

impl GGUFOnDiskTensorInfo {
    pub fn decode(buf: &mut GGUFBufReader, version: u32) -> Result<Self> {
        let mut r = GGUFMetadataReader::new(buf, version);
        let name = r.read_string()?.to_string();
        let n_dimensions = r.read_u32()? as usize;
        let dimensions = r.read_u64_array(n_dimensions)?.to_vec();
        let typ = GGMLType::try_from(r.read_u32()?)?;
        let offset = r.read_u64()?;
        Ok(Self {
            name,
            dimensions,
            typ,
            offset,
        })
    }
}

struct GGUFTensorInfo<'a> {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    name: String,
    // The dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    dimensions: Vec<u64>,
    // The type of the tensor.
    typ: GGMLType,
    // The offset of the tensor's data in this file in bytes.
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    // Must be a multiple of `ALIGNMENT`.
    offset: u64,
    // The tensor data
    data: &'a [u8],
}

impl<'a> GGUFTensorInfo<'a> {
    pub fn new(
        name: String,
        dimensions: Vec<u64>,
        typ: GGMLType,
        offset: u64,
        data: &'a [u8],
    ) -> Self {
        Self {
            name,
            dimensions,
            typ,
            offset,
            data,
        }
    }
}

pub struct GGUFFile<'a> {
    // The header of the file.
    header: GGUFHeader<'a>,

    // Tensor infos, which can be used to locate the tensor data.
    tensor_infos: Vec<GGUFTensorInfo<'a>>,

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    tensor_data: &'a [u8],
}

impl<'a> GGUFFile<'a> {
    fn decode(buf: &mut GGUFBufReader<'a>) -> Result<Self> {
        let header = GGUFHeader::decode(buf)?;

        // load on disk tensor infos
        let mut on_disk_tensor_infos = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            let tensor_info = GGUFOnDiskTensorInfo::decode(buf, header.version)?;
            on_disk_tensor_infos.push(tensor_info);
        }

        // find the tensor_data position
        let position = buf.read_bytes();
        let alignment = header.alignment() as usize;
        let next_position = position - (position % alignment) + alignment;
        let _ = buf.read(next_position - position)?;
        let tensor_data = buf.cursor();

        // convert the on-disk tensor infos to in-memory
        let tensor_infos = Self::convert_tensor_infos(&on_disk_tensor_infos, tensor_data)?;

        Ok(Self {
            header,
            tensor_infos,
            tensor_data,
        })
    }

    fn convert_tensor_infos(
        tensor_infos: &[GGUFOnDiskTensorInfo],
        tensor_data: &'a [u8],
    ) -> Result<Vec<GGUFTensorInfo<'a>>> {
        let mut result = Vec::with_capacity(tensor_infos.len());
        for (i, tensor_info) in tensor_infos.iter().enumerate() {
            let next_offset = if i >= tensor_infos.len() {
                tensor_data.len()
            } else {
                tensor_infos[i + 1].offset as usize
            };
            let data = &tensor_data[tensor_info.offset as usize..next_offset];

            let item = GGUFTensorInfo::new(
                tensor_info.name.clone(),
                tensor_info.dimensions.clone(),
                tensor_info.typ,
                tensor_info.offset,
                &data,
            );
            result.push(item);
        }
        Ok(result)
    }
}

pub struct GGUFFileLoader {
    mmap: memmap2::Mmap,
}

impl GGUFFileLoader {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path).map_err(|err| GGUFError {
            kind: GGUFErrorKind::IOError,
            message: format!("failed to open the file: {}", path),
            cause: Some(Box::new(err)),
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|err| GGUFError {
                kind: GGUFErrorKind::IOError,
                message: format!("failed to mmap file: {}", path),
                cause: Some(Box::new(err)),
            })?
        };

        Ok(Self { mmap })
    }

    pub fn load(&self) -> Result<GGUFFile<'_>> {
        let buf = &mut GGUFBufReader::new(&self.mmap[..]);
        GGUFFile::decode(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() -> Result<()> {
        let loader = GGUFFileLoader::new("testdata/tinyllamas-stories-260k-f32.gguf")?;
        loader.load()?;
        Ok(())
    }
}
