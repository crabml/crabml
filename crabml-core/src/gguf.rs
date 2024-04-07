use std::collections::HashMap;
use std::fmt::Display;
use std::fs::File;
use std::mem;
use std::sync::Arc;

use int_enum::IntEnum;
use memmap2::Mmap;

use crate::error::Error;
use crate::error::ErrorKind;
use crate::error::Result;

const GGUF_MAGIC: u32 = 0x46554747;
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

#[repr(u32)]
#[derive(Debug, Clone, Copy, IntEnum)]
pub enum GGUFVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
}

impl Display for GGUFVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GGUFVersion::V1 => write!(f, "1"),
            GGUFVersion::V2 => write!(f, "2"),
            GGUFVersion::V3 => write!(f, "3"),
        }
    }
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
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    COUNT = 19,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::F16 => write!(f, "F16"),
            GGMLType::Q4_0 => write!(f, "Q4_0"),
            GGMLType::Q4_1 => write!(f, "Q4_1"),
            GGMLType::Q5_0 => write!(f, "Q5_0"),
            GGMLType::Q5_1 => write!(f, "Q5_1"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
            GGMLType::Q8_1 => write!(f, "Q8_1"),
            GGMLType::Q2K => write!(f, "Q2_K"),
            GGMLType::Q3K => write!(f, "Q3_K"),
            GGMLType::Q4K => write!(f, "Q4_K"),
            GGMLType::Q5K => write!(f, "Q5_K"),
            GGMLType::Q6K => write!(f, "Q6_K"),
            GGMLType::Q8K => write!(f, "Q8_K"),
            GGMLType::I8 => write!(f, "I8"),
            GGMLType::I16 => write!(f, "I16"),
            GGMLType::I32 => write!(f, "I32"),
            GGMLType::COUNT => write!(f, "COUNT"),
        }
    }
}

impl TryFrom<u32> for GGMLType {
    type Error = Error;

    fn try_from(v: u32) -> std::result::Result<Self, Self::Error> {
        Self::from_int(v).map_err(|err| Error {
            kind: ErrorKind::FormatError,
            message: format!("failed to decode the ggml type for {}", v),
            cause: Some(Arc::new(err)),
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
    type Error = Error;

    fn try_from(v: u32) -> std::result::Result<Self, Self::Error> {
        Self::from_int(v).map_err(|err| Error {
            kind: ErrorKind::FormatError,
            message: format!("failed to decode the value type for {}", v),
            cause: Some(Arc::new(err)),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
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

impl<'a> GGUFMetadataValue<'a> {
    pub fn typ(&self) -> GGUFMetadataValueType {
        match self {
            GGUFMetadataValue::U8(_) => GGUFMetadataValueType::U8,
            GGUFMetadataValue::I8(_) => GGUFMetadataValueType::I8,
            GGUFMetadataValue::U16(_) => GGUFMetadataValueType::U16,
            GGUFMetadataValue::I16(_) => GGUFMetadataValueType::I16,
            GGUFMetadataValue::U32(_) => GGUFMetadataValueType::U32,
            GGUFMetadataValue::I32(_) => GGUFMetadataValueType::I32,
            GGUFMetadataValue::U64(_) => GGUFMetadataValueType::U64,
            GGUFMetadataValue::I64(_) => GGUFMetadataValueType::I64,
            GGUFMetadataValue::F32(_) => GGUFMetadataValueType::F32,
            GGUFMetadataValue::F64(_) => GGUFMetadataValueType::F64,
            GGUFMetadataValue::Bool(_) => GGUFMetadataValueType::Bool,
            GGUFMetadataValue::String(_) => GGUFMetadataValueType::String,
            GGUFMetadataValue::Array(_) => GGUFMetadataValueType::Array,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
            return Err(Error {
                kind: ErrorKind::FormatError,
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
where 'a: 'b
{
    buf: &'b mut GGUFBufReader<'a>,
    version: GGUFVersion,
}

macro_rules! define_gguf_metadata_value_read_fn {
    ($read_array_func:ident, $read_item_func:ident, $typ:ty) => {
        fn $read_array_func(&mut self, n: usize) -> Result<&'a [$typ]> {
            let typ_size = mem::size_of::<$typ>();
            let data = self.buf.read(n * typ_size)?;
            let transmuted_data = unsafe {
                assert!(data.len() % typ_size == 0);
                let ptr = data.as_ptr();
                // assert!(ptr.align_offset(typ_size) == 0, "unaligned data: {:p}", ptr);
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
    pub fn new(buf: &'b mut GGUFBufReader<'a>, version: GGUFVersion) -> GGUFMetadataReader<'a, 'b> {
        GGUFMetadataReader { buf, version }
    }

    pub fn read_value(&mut self) -> Result<GGUFMetadataValue<'a>> {
        let typ = self.read_u32()?;
        let typ = GGUFMetadataValueType::try_from(typ)?;
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
        let typ = self.read_u32()?;
        let typ = GGUFMetadataValueType::try_from(typ)?;
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
        let s = std::str::from_utf8(buf).map_err(|e| Error {
            kind: ErrorKind::FormatError,
            message: "Invalid UTF-8 string".to_string(),
            cause: Some(Arc::new(e)),
        });
        s
    }

    /// Read the length for string & array. It would be an 32 bit unsigned integer on spec v1, but 64
    /// bit on spec v2. For more infomation:
    /// https://github.com/philpax/ggml/commit/b021b2577d4294800ece200c9f26c9c65b0f6f51
    fn read_len(&mut self) -> Result<usize> {
        let v = match self.version {
            GGUFVersion::V1 => self.read_u32()? as usize,
            GGUFVersion::V2 => self.read_u64()? as usize,
            GGUFVersion::V3 => self.read_u64()? as usize,
        };
        Ok(v)
    }

    /// compat v1 & v2 on the type change of the field dimensions[n]. for more infomation:
    /// https://github.com/philpax/ggml/commit/b021b2577d4294800ece200c9f26c9c65b0f6f51#diff-d553f5c3bea777978686f7fd4ed40a185a2d8cdec90cba5e2d8a4d5504148505L154
    fn read_len_array(&mut self, n: usize) -> Result<Vec<usize>> {
        let v = match self.version {
            GGUFVersion::V1 => self
                .read_u32_array(n)?
                .iter()
                .map(|v| *v as usize)
                .collect(),
            GGUFVersion::V2 | GGUFVersion::V3 => self
                .read_u64_array(n)?
                .iter()
                .map(|v| *v as usize)
                .collect(),
        };
        Ok(v)
    }
}

pub struct GGUFMetadata<'a> {
    metadata_kv: HashMap<String, GGUFMetadataValue<'a>>,
}

macro_rules! define_gguf_metadata_get_primitive_fn {
    ($get_item_func:ident, $get_array_func:ident, $typ:ty, $item_enum_kind:ident, $array_enum_kind:ident) => {
        pub fn $get_item_func(&self, key: &str) -> Option<$typ> {
            let val = self.metadata_kv.get(key)?;
            match val {
                GGUFMetadataValue::$item_enum_kind(val) => Some(*val),
                _ => None,
            }
        }

        pub fn $get_array_func(&self, key: &str) -> Option<&[$typ]> {
            let val = self.metadata_kv.get(key)?;
            let arr = match val {
                GGUFMetadataValue::Array(arr) => arr,
                _ => return None,
            };
            match arr {
                GGUFMetadataArray::$array_enum_kind(arr) => Some(arr),
                _ => None,
            }
        }
    };
}

impl<'a> GGUFMetadata<'a> {
    pub fn as_hashmap(&self) -> &HashMap<String, GGUFMetadataValue<'a>> {
        &self.metadata_kv
    }

    define_gguf_metadata_get_primitive_fn!(get_u8, get_u8_array, u8, U8, U8Array);
    define_gguf_metadata_get_primitive_fn!(get_i8, get_i8_array, i8, I8, I8Array);
    define_gguf_metadata_get_primitive_fn!(get_u16, get_u16_array, u16, U16, U16Array);
    define_gguf_metadata_get_primitive_fn!(get_i16, get_i16_array, i16, I16, I16Array);
    define_gguf_metadata_get_primitive_fn!(get_u32, get_u32_array, u32, U32, U32Array);
    define_gguf_metadata_get_primitive_fn!(get_i32, get_i32_array, i32, I32, I32Array);
    define_gguf_metadata_get_primitive_fn!(get_u64, get_u64_array, u64, U64, U64Array);
    define_gguf_metadata_get_primitive_fn!(get_i64, get_i64_array, i64, I64, I64Array);
    define_gguf_metadata_get_primitive_fn!(get_f32, get_f32_array, f32, F32, F32Array);
    define_gguf_metadata_get_primitive_fn!(get_bool, get_bool_array, u8, Bool, BoolArray);

    pub fn get_string(&self, key: &str) -> Option<&str> {
        let val = self.metadata_kv.get(key)?;
        match val {
            GGUFMetadataValue::String(val) => Some(val),
            _ => None,
        }
    }

    pub fn get_string_array(&self, key: &str) -> Option<&[&str]> {
        let val = self.metadata_kv.get(key)?;
        let arr = match val {
            GGUFMetadataValue::Array(arr) => arr,
            _ => return None,
        };
        match arr {
            GGUFMetadataArray::StringArray(arr) => Some(arr),
            _ => None,
        }
    }
}

struct GGUFHeader<'a> {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    #[allow(dead_code)]
    magic: u32,

    // The version of the format implemented.
    // Must be `2` for version described in this spec.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    version: GGUFVersion,

    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    tensor_count: usize,

    // The metadata key-value pairs.
    metadata: GGUFMetadata<'a>,

    // architecture is an required fields in the metadata
    architecture: String,
}

impl<'a> GGUFHeader<'a> {
    fn decode(buf: &mut GGUFBufReader<'a>) -> Result<Self> {
        let mut r = GGUFMetadataReader::new(buf, GGUFVersion::V2);
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error {
                kind: ErrorKind::FormatError,
                message: format!("Invalid magic number: {}", magic),
                cause: None,
            });
        }

        let version = r.read_u32()?;
        let version = GGUFVersion::from_int(version).map_err(|err| Error {
            kind: ErrorKind::FormatError,
            message: format!(
                "Unsupported version number: {}, only 1, 2 is supported yet",
                version
            ),
            cause: Some(Arc::new(err)),
        })?;
        r.version = version;

        let tensor_count = r.read_len()?;
        let metadata_kv_count = r.read_len()?;

        // load metadata
        let mut metadata_kv = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = r.read_string()?;
            let value = r.read_value()?;
            metadata_kv.insert(key.to_string(), value);
        }
        let metadata = GGUFMetadata { metadata_kv };

        // load the required fields
        let architecture = match metadata.get_string(KEY_GENERAL_ARCHITECTURE) {
            Some(s) => s.to_string(),
            _ => {
                return Err(Error {
                    kind: ErrorKind::FormatError,
                    message: "Missing string metadata general.architecture".to_string(),
                    cause: None,
                });
            }
        };

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata,
            architecture,
        })
    }

    /// the global alignment to use, as described above. This can vary to allow for different alignment schemes,
    /// but it must be a multiple of 8. Some writers may not write the alignment. If the alignment is not specified,
    /// assume it is 32.
    pub fn alignment(&self) -> u64 {
        match self.metadata.as_hashmap().get(KEY_GENERAL_ALIGNMENT) {
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
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// The version of the quantization format. Not required if the model is not quantized (i.e. no tensors are
    /// quantized). If any tensors are quantized, this must be present. This is separate to the quantization
    /// scheme of the tensors itself; the quantization version may change without changing the scheme's name
    /// (e.g. the quantization scheme is Q5_K, and the quantization version is 4).
    pub fn quantization_version(&self) -> Option<u32> {
        self.metadata.get_u32(KEY_GENERAL_QUANTIZATION_VERSION)
    }
}

struct GGUFOnDiskTensorInfo {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    name: String,
    // The dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    dimensions: Vec<usize>,
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
    pub fn decode(buf: &mut GGUFBufReader, version: GGUFVersion) -> Result<Self> {
        let mut r = GGUFMetadataReader::new(buf, version);
        let name = r.read_string()?.to_string();
        let n_dimensions = r.read_u32()? as usize;
        let dimensions = r.read_len_array(n_dimensions)?;
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

#[derive(Clone, Debug)]
pub struct GGUFTensorInfo<'a> {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    name: String,
    // The dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    dimensions: Vec<usize>,
    // The type of the tensor.
    typ: GGMLType,
    // The tensor data
    data: &'a [u8],
}

impl<'a> GGUFTensorInfo<'a> {
    pub fn new(name: String, dimensions: Vec<usize>, typ: GGMLType, data: &'a [u8]) -> Self {
        Self {
            name,
            dimensions,
            typ,
            data,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn typ(&self) -> GGMLType {
        self.typ
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
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
    _tensor_data: &'a [u8],
}

impl<'a> GGUFFile<'a> {
    fn decode(buf: &mut GGUFBufReader<'a>) -> Result<Self> {
        let header = GGUFHeader::decode(buf)?;

        // load on disk tensor infos
        let mut on_disk_tensor_infos = Vec::with_capacity(header.tensor_count);
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
            _tensor_data: tensor_data,
        })
    }

    fn convert_tensor_infos(
        tensor_infos: &[GGUFOnDiskTensorInfo],
        tensor_data: &'a [u8],
    ) -> Result<Vec<GGUFTensorInfo<'a>>> {
        let mut result = Vec::with_capacity(tensor_infos.len());
        for (i, tensor_info) in tensor_infos.iter().enumerate() {
            let next_offset = if i + 1 >= tensor_infos.len() {
                tensor_data.len()
            } else {
                tensor_infos[i + 1].offset as usize
            };
            let data = &tensor_data[tensor_info.offset as usize..next_offset];

            let item = GGUFTensorInfo::new(
                tensor_info.name.clone(),
                tensor_info.dimensions.clone(),
                tensor_info.typ,
                data,
            );
            result.push(item);
        }
        Ok(result)
    }

    pub fn architecture(&self) -> &str {
        self.header.architecture()
    }

    pub fn quantization_version(&self) -> Option<u32> {
        self.header.quantization_version()
    }

    pub fn version(&self) -> GGUFVersion {
        self.header.version
    }

    pub fn metadata(&self) -> &GGUFMetadata {
        &self.header.metadata
    }

    pub fn tensor_infos(&self) -> &[GGUFTensorInfo] {
        &self.tensor_infos
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<GGUFTensorInfo> {
        self.tensor_infos
            .iter()
            .find(|ti| ti.name() == name)
            .cloned()
    }
}

pub struct GGUFFileLoader {
    mmap: memmap2::Mmap,
}

impl GGUFFileLoader {
    pub fn new(path: &str, mlock: bool) -> Result<Self> {
        let file = File::open(path).map_err(|err| Error {
            kind: ErrorKind::IOError,
            message: format!("failed to open the file: {}", path),
            cause: Some(Arc::new(err)),
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|err| Error {
                kind: ErrorKind::IOError,
                message: format!("failed to mmap file: {}", path),
                cause: Some(Arc::new(err)),
            })?
        };
        mmap.advise(memmap2::Advice::WillNeed)
            .map_err(|err| Error {
                kind: ErrorKind::IOError,
                message: format!("failed to advise the mmap: {}", path),
                cause: Some(Arc::new(err)),
            })?;
        if mlock {
            mmap.lock().map_err(|err| Error {
                kind: ErrorKind::IOError,
                message: format!("failed to advise the mmap: {}", path),
                cause: Some(Arc::new(err)),
            })?;
        }
        Ok(Self { mmap })
    }

    pub fn open(&self) -> Result<GGUFFile<'_>> {
        let buf = &mut GGUFBufReader::new(&self.mmap[..]);
        GGUFFile::decode(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tensors() -> Result<()> {
        let loader = GGUFFileLoader::new("../testdata/tinyllamas-stories-260k-f32.gguf", false)?;
        let gf = loader.open()?;

        assert_eq!(gf.tensor_infos.len(), 48);
        assert_eq!(gf.tensor_infos[0].name(), "token_embd.weight");
        assert_eq!(gf.tensor_infos[0].data().len(), 131072);
        assert_eq!(gf.tensor_infos[0].data().len() % 32, 0);
        assert_eq!(gf.tensor_infos[0].typ().to_string(), "F32");
        assert_eq!(gf.tensor_infos[0].dimensions(), vec![64, 512]);

        let typs = gf
            .tensor_infos
            .iter()
            .map(|i| i.typ().to_string())
            .collect::<Vec<_>>();
        assert_eq!(typs, vec!["F32"; 48]);

        let tensor_strs = gf
            .tensor_infos()
            .iter()
            .map(|i| format!("{} - {:?} - {:?}", i.name(), i.typ(), i.dimensions()))
            .collect::<Vec<_>>();
        assert_eq!(tensor_strs, vec![
            "token_embd.weight - F32 - [64, 512]",
            "blk.0.attn_q.weight - F32 - [64, 64]",
            "blk.0.attn_k.weight - F32 - [64, 32]",
            "blk.0.attn_v.weight - F32 - [64, 32]",
            "blk.0.attn_output.weight - F32 - [64, 64]",
            "blk.0.ffn_gate.weight - F32 - [64, 172]",
            "blk.0.ffn_down.weight - F32 - [172, 64]",
            "blk.0.ffn_up.weight - F32 - [64, 172]",
            "blk.0.attn_norm.weight - F32 - [64]",
            "blk.0.ffn_norm.weight - F32 - [64]",
            "blk.1.attn_q.weight - F32 - [64, 64]",
            "blk.1.attn_k.weight - F32 - [64, 32]",
            "blk.1.attn_v.weight - F32 - [64, 32]",
            "blk.1.attn_output.weight - F32 - [64, 64]",
            "blk.1.ffn_gate.weight - F32 - [64, 172]",
            "blk.1.ffn_down.weight - F32 - [172, 64]",
            "blk.1.ffn_up.weight - F32 - [64, 172]",
            "blk.1.attn_norm.weight - F32 - [64]",
            "blk.1.ffn_norm.weight - F32 - [64]",
            "blk.2.attn_q.weight - F32 - [64, 64]",
            "blk.2.attn_k.weight - F32 - [64, 32]",
            "blk.2.attn_v.weight - F32 - [64, 32]",
            "blk.2.attn_output.weight - F32 - [64, 64]",
            "blk.2.ffn_gate.weight - F32 - [64, 172]",
            "blk.2.ffn_down.weight - F32 - [172, 64]",
            "blk.2.ffn_up.weight - F32 - [64, 172]",
            "blk.2.attn_norm.weight - F32 - [64]",
            "blk.2.ffn_norm.weight - F32 - [64]",
            "blk.3.attn_q.weight - F32 - [64, 64]",
            "blk.3.attn_k.weight - F32 - [64, 32]",
            "blk.3.attn_v.weight - F32 - [64, 32]",
            "blk.3.attn_output.weight - F32 - [64, 64]",
            "blk.3.ffn_gate.weight - F32 - [64, 172]",
            "blk.3.ffn_down.weight - F32 - [172, 64]",
            "blk.3.ffn_up.weight - F32 - [64, 172]",
            "blk.3.attn_norm.weight - F32 - [64]",
            "blk.3.ffn_norm.weight - F32 - [64]",
            "blk.4.attn_q.weight - F32 - [64, 64]",
            "blk.4.attn_k.weight - F32 - [64, 32]",
            "blk.4.attn_v.weight - F32 - [64, 32]",
            "blk.4.attn_output.weight - F32 - [64, 64]",
            "blk.4.ffn_gate.weight - F32 - [64, 172]",
            "blk.4.ffn_down.weight - F32 - [172, 64]",
            "blk.4.ffn_up.weight - F32 - [64, 172]",
            "blk.4.attn_norm.weight - F32 - [64]",
            "blk.4.ffn_norm.weight - F32 - [64]",
            "output_norm.weight - F32 - [64]",
            "output.weight - F32 - [64, 512]"
        ]);
        Ok(())
    }

    #[test]
    fn test_load_metadata() -> Result<()> {
        let loader = GGUFFileLoader::new("../testdata/tinyllamas-stories-260k-f32.gguf", false)?;
        let gf = loader.open()?;
        assert_eq!(gf.header.architecture(), "llama");
        assert_eq!(gf.header.alignment(), 32);

        let mut keys = gf
            .header
            .metadata
            .as_hashmap()
            .keys()
            .map(|i| i.to_string())
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(keys, vec![
            "general.architecture",
            "general.name",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.attention.layer_norm_rms_epsilon",
            "llama.block_count",
            "llama.context_length",
            "llama.embedding_length",
            "llama.feed_forward_length",
            "llama.rope.dimension_count",
            "llama.tensor_data_layout",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.model",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.tokens"
        ]);

        let tests = vec![
            ("general.architecture", "Some(String(\"llama\"))"),
            ("general.name", "Some(String(\"tinyllamas-stories-260k\"))"),
            ("llama.attention.head_count", "Some(U32(8))"),
            ("llama.attention.head_count_kv", "Some(U32(4))"),
            ("llama.attention.layer_norm_rms_epsilon", "Some(F32(1e-5))"),
            ("llama.block_count", "Some(U32(5))"),
            ("llama.context_length", "Some(U32(512))"),
            ("llama.embedding_length", "Some(U32(64))"),
            ("llama.feed_forward_length", "Some(U32(172))"),
            ("llama.rope.dimension_count", "Some(U32(8))"),
            (
                "llama.tensor_data_layout",
                "Some(String(\"Meta AI original pth\"))",
            ),
            ("tokenizer.ggml.bos_token_id", "Some(U32(1))"),
            ("tokenizer.ggml.eos_token_id", "Some(U32(2))"),
            ("tokenizer.ggml.model", "Some(String(\"llama\"))"),
            ("tokenizer.ggml.padding_token_id", "Some(U32(0))"),
            (
                "tokenizer.ggml.scores",
                "Some(Array(F32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -23.0, -24.0, -25.0, -26.0, -27.0, -28.0, -29.0, -30.0, -31.0, -32.0, -33.0, -34.0, -35.0, -36.0, -37.0, -38.0, -39.0, -40.0, -41.0, -42.0, -43.0, -44.0, -45.0, -46.0, -47.0, -48.0, -49.0, -50.0, -51.0, -52.0, -53.0, -54.0, -55.0, -56.0, -57.0, -58.0, -59.0, -60.0, -61.0, -62.0, -63.0, -64.0, -65.0, -66.0, -67.0, -68.0, -69.0, -70.0, -71.0, -72.0, -73.0, -74.0, -75.0, -76.0, -77.0, -78.0, -79.0, -80.0, -81.0, -82.0, -83.0, -84.0, -85.0, -86.0, -87.0, -88.0, -89.0, -90.0, -91.0, -92.0, -93.0, -94.0, -95.0, -96.0, -97.0, -98.0, -99.0, -100.0, -101.0, -102.0, -103.0, -104.0, -105.0, -106.0, -107.0, -108.0, -109.0, -110.0, -111.0, -112.0, -113.0, -114.0, -115.0, -116.0, -117.0, -118.0, -119.0, -120.0, -121.0, -122.0, -123.0, -124.0, -125.0, -126.0, -127.0, -128.0, -129.0, -130.0, -131.0, -132.0, -133.0, -134.0, -135.0, -136.0, -137.0, -138.0, -139.0, -140.0, -141.0, -142.0, -143.0, -144.0, -145.0, -146.0, -147.0, -148.0, -149.0, -150.0, -151.0, -152.0, -153.0, -154.0, -155.0, -156.0, -157.0, -158.0, -159.0, -160.0, -161.0, -162.0, -163.0, -164.0, -165.0, -166.0, -167.0, -168.0, -169.0, -170.0, -171.0, -172.0, -173.0, -174.0, -175.0, -176.0, -177.0, -178.0, -179.0, -180.0, -181.0, -182.0, -183.0, -184.0, -185.0, -186.0, -187.0, -188.0, -189.0, -190.0, -191.0, -192.0, -193.0, -194.0, -195.0, -196.0, -197.0, -198.0, -199.0, -200.0, -201.0, -202.0, -203.0, -204.0, -205.0, -206.0, -207.0, -208.0, -209.0, -210.0, -211.0, -212.0, -213.0, -214.0, -215.0, -216.0, -217.0, -218.0, -219.0, -220.0, -221.0, -222.0, -223.0, -224.0, -225.0, -226.0, -227.0, -228.0, -229.0, -230.0, -231.0, -232.0, -233.0, -234.0, -235.0, -236.0, -237.0, -238.0, -239.0, -240.0, -241.0, -242.0, -243.0, -244.0, -245.0, -246.0, -247.0, -248.0, -249.0, -250.0, -251.0, -252.0])))",
            ),
            (
                "tokenizer.ggml.token_type",
                "Some(Array(I32Array([2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))",
            ),
            (
                "tokenizer.ggml.tokens",
                "Some(Array(StringArray([\"<unk>\", \"<s>\", \"</s>\", \"<0x00>\", \"<0x01>\", \"<0x02>\", \"<0x03>\", \"<0x04>\", \"<0x05>\", \"<0x06>\", \"<0x07>\", \"<0x08>\", \"<0x09>\", \"<0x0A>\", \"<0x0B>\", \"<0x0C>\", \"<0x0D>\", \"<0x0E>\", \"<0x0F>\", \"<0x10>\", \"<0x11>\", \"<0x12>\", \"<0x13>\", \"<0x14>\", \"<0x15>\", \"<0x16>\", \"<0x17>\", \"<0x18>\", \"<0x19>\", \"<0x1A>\", \"<0x1B>\", \"<0x1C>\", \"<0x1D>\", \"<0x1E>\", \"<0x1F>\", \"<0x20>\", \"<0x21>\", \"<0x22>\", \"<0x23>\", \"<0x24>\", \"<0x25>\", \"<0x26>\", \"<0x27>\", \"<0x28>\", \"<0x29>\", \"<0x2A>\", \"<0x2B>\", \"<0x2C>\", \"<0x2D>\", \"<0x2E>\", \"<0x2F>\", \"<0x30>\", \"<0x31>\", \"<0x32>\", \"<0x33>\", \"<0x34>\", \"<0x35>\", \"<0x36>\", \"<0x37>\", \"<0x38>\", \"<0x39>\", \"<0x3A>\", \"<0x3B>\", \"<0x3C>\", \"<0x3D>\", \"<0x3E>\", \"<0x3F>\", \"<0x40>\", \"<0x41>\", \"<0x42>\", \"<0x43>\", \"<0x44>\", \"<0x45>\", \"<0x46>\", \"<0x47>\", \"<0x48>\", \"<0x49>\", \"<0x4A>\", \"<0x4B>\", \"<0x4C>\", \"<0x4D>\", \"<0x4E>\", \"<0x4F>\", \"<0x50>\", \"<0x51>\", \"<0x52>\", \"<0x53>\", \"<0x54>\", \"<0x55>\", \"<0x56>\", \"<0x57>\", \"<0x58>\", \"<0x59>\", \"<0x5A>\", \"<0x5B>\", \"<0x5C>\", \"<0x5D>\", \"<0x5E>\", \"<0x5F>\", \"<0x60>\", \"<0x61>\", \"<0x62>\", \"<0x63>\", \"<0x64>\", \"<0x65>\", \"<0x66>\", \"<0x67>\", \"<0x68>\", \"<0x69>\", \"<0x6A>\", \"<0x6B>\", \"<0x6C>\", \"<0x6D>\", \"<0x6E>\", \"<0x6F>\", \"<0x70>\", \"<0x71>\", \"<0x72>\", \"<0x73>\", \"<0x74>\", \"<0x75>\", \"<0x76>\", \"<0x77>\", \"<0x78>\", \"<0x79>\", \"<0x7A>\", \"<0x7B>\", \"<0x7C>\", \"<0x7D>\", \"<0x7E>\", \"<0x7F>\", \"<0x80>\", \"<0x81>\", \"<0x82>\", \"<0x83>\", \"<0x84>\", \"<0x85>\", \"<0x86>\", \"<0x87>\", \"<0x88>\", \"<0x89>\", \"<0x8A>\", \"<0x8B>\", \"<0x8C>\", \"<0x8D>\", \"<0x8E>\", \"<0x8F>\", \"<0x90>\", \"<0x91>\", \"<0x92>\", \"<0x93>\", \"<0x94>\", \"<0x95>\", \"<0x96>\", \"<0x97>\", \"<0x98>\", \"<0x99>\", \"<0x9A>\", \"<0x9B>\", \"<0x9C>\", \"<0x9D>\", \"<0x9E>\", \"<0x9F>\", \"<0xA0>\", \"<0xA1>\", \"<0xA2>\", \"<0xA3>\", \"<0xA4>\", \"<0xA5>\", \"<0xA6>\", \"<0xA7>\", \"<0xA8>\", \"<0xA9>\", \"<0xAA>\", \"<0xAB>\", \"<0xAC>\", \"<0xAD>\", \"<0xAE>\", \"<0xAF>\", \"<0xB0>\", \"<0xB1>\", \"<0xB2>\", \"<0xB3>\", \"<0xB4>\", \"<0xB5>\", \"<0xB6>\", \"<0xB7>\", \"<0xB8>\", \"<0xB9>\", \"<0xBA>\", \"<0xBB>\", \"<0xBC>\", \"<0xBD>\", \"<0xBE>\", \"<0xBF>\", \"<0xC0>\", \"<0xC1>\", \"<0xC2>\", \"<0xC3>\", \"<0xC4>\", \"<0xC5>\", \"<0xC6>\", \"<0xC7>\", \"<0xC8>\", \"<0xC9>\", \"<0xCA>\", \"<0xCB>\", \"<0xCC>\", \"<0xCD>\", \"<0xCE>\", \"<0xCF>\", \"<0xD0>\", \"<0xD1>\", \"<0xD2>\", \"<0xD3>\", \"<0xD4>\", \"<0xD5>\", \"<0xD6>\", \"<0xD7>\", \"<0xD8>\", \"<0xD9>\", \"<0xDA>\", \"<0xDB>\", \"<0xDC>\", \"<0xDD>\", \"<0xDE>\", \"<0xDF>\", \"<0xE0>\", \"<0xE1>\", \"<0xE2>\", \"<0xE3>\", \"<0xE4>\", \"<0xE5>\", \"<0xE6>\", \"<0xE7>\", \"<0xE8>\", \"<0xE9>\", \"<0xEA>\", \"<0xEB>\", \"<0xEC>\", \"<0xED>\", \"<0xEE>\", \"<0xEF>\", \"<0xF0>\", \"<0xF1>\", \"<0xF2>\", \"<0xF3>\", \"<0xF4>\", \"<0xF5>\", \"<0xF6>\", \"<0xF7>\", \"<0xF8>\", \"<0xF9>\", \"<0xFA>\", \"<0xFB>\", \"<0xFC>\", \"<0xFD>\", \"<0xFE>\", \"<0xFF>\", \"▁t\", \"he\", \"▁a\", \"▁s\", \"▁w\", \"nd\", \"▁the\", \"ed\", \"▁to\", \"▁b\", \"▁and\", \"▁h\", \"in\", \"▁f\", \"▁wa\", \"▁T\", \"it\", \"re\", \"ou\", \"▁l\", \"▁d\", \"▁c\", \"▁he\", \"▁p\", \"ay\", \"▁m\", \"er\", \"▁was\", \"om\", \"im\", \"on\", \"il\", \"▁The\", \"id\", \"is\", \"at\", \"ar\", \"▁sa\", \"▁n\", \"▁g\", \"ing\", \"▁ha\", \"▁S\", \"en\", \"an\", \"or\", \"le\", \"ll\", \"▁L\", \"▁th\", \"ot\", \"ily\", \"▁her\", \"▁it\", \"▁\\\"\", \"am\", \"ir\", \"et\", \"▁Lily\", \"▁u\", \"▁O\", \"▁H\", \"▁On\", \"▁in\", \"ut\", \"▁pl\", \"ri\", \"▁Tim\", \"ow\", \"▁day\", \"▁be\", \"ver\", \"ce\", \"ith\", \"ig\", \"▁o\", \"▁with\", \"▁said\", \"▁play\", \"▁She\", \"pp\", \"ck\", \"ld\", \"▁They\", \"my\", \"▁e\", \"▁his\", \"▁He\", \"oo\", \"▁y\", \"▁st\", \"▁up\", \"▁that\", \"▁r\", \"▁on\", \"ke\", \"ked\", \"st\", \"▁mom\", \"▁she\", \"▁I\", \"ve\", \"nt\", \"itt\", \"very\", \"▁you\", \"▁happ\", \"▁they\", \"end\", \"▁B\", \"ime\", \"▁big\", \"▁fri\", \"se\", \"▁of\", \"▁friend\", \"ittle\", \"▁little\", \"ent\", \"▁time\", \"un\", \"ad\", \"▁had\", \"▁we\", \"▁there\", \"▁so\", \"▁One\", \"her\", \"▁for\", \"all\", \"ould\", \"▁nam\", \"▁want\", \"▁M\", \"▁happy\", \"▁saw\", \"▁named\", \"ved\", \"▁li\", \"▁but\", \"▁very\", \"▁do\", \"▁lo\", \"ch\", \"▁Once\", \"▁ne\", \"▁Timmy\", \"es\", \"▁upon\", \"out\", \"▁k\", \"▁\", \"e\", \"a\", \"t\", \"o\", \"h\", \"n\", \"i\", \"d\", \"s\", \"r\", \"l\", \"y\", \"m\", \"w\", \"u\", \".\", \"p\", \"g\", \"c\", \"b\", \"f\", \",\", \"k\", \"T\", \"v\", \"\\\"\", \"S\", \"L\", \"'\", \"H\", \"O\", \"I\", \"!\", \"x\", \"B\", \"M\", \"A\", \"W\", \"j\", \"?\", \"z\", \"Y\", \"F\", \"J\", \"D\", \"q\", \"C\", \"N\", \"E\", \"P\", \"R\", \"K\", \"G\", \"-\", \"“\", \"”\", \":\", \"’\", \"Z\", \"V\", \"U\", \"3\", \"Q\", \";\", \"1\", \"–\", \"0\", \"X\", \"2\", \"5\", \"—\", \"‘\", \"9\", \"4\", \"é\", \"…\", \"8\", \")\", \"(\", \"6\", \"7\", \"/\", \"ñ\", \"$\", \"`\", \"+\", \"*\", \"\\u{a0}\", \"&\", \"\\\\\", \"%\", \"â\", \"€\", \"<\", \">\", \"|\", \"™\", \"[\", \"]\", \"~\", \"\\u{200a}\"])))",
            ),
        ];
        for (k, v) in tests {
            let got = gf.header.metadata.as_hashmap().get(k);
            assert_eq!(v, format!("{:?}", got));
        }

        Ok(())
    }
}
