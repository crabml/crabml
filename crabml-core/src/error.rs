use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ErrorKind {
    /// Unexpected error
    Unexpected,

    /// raised on any IO error, like EOF
    IOError,

    /// raised when tensor not found in the model
    TensorNotFound,

    /// raised when the model format is not supported
    ModelError,

    /// raised when user supplied invalid input arguments
    BadInput,

    /// raised when parsing GGUF or other kind of model files
    FormatError,

    /// raised on manipulating tensors, like dimension mismatch
    TensorError,

    /// raised on chat template is not found
    ChatTemplateNotFound,

    /// unimplemented yet
    NotImplemented,
}

#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub cause: Option<Arc<dyn StdError + Send + Sync>>,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)?;
        if let Some(cause) = self.cause.as_ref() {
            write!(f, "\ncaused by: {}", cause)?;
        }
        Ok(())
    }
}

impl StdError for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! error {
    ($kind:expr => $err:expr) => {
        $crate::error::Error {
            kind: $kind,
            message: String::new(),
            cause: Some(Arc::new($err)),
        }
    };
    ($kind:expr, $($arg:tt)*) => {
        $crate::error::Error {
            kind: $kind,
            message: format!($($arg)*),
            cause: None,
        }
    };
}

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error!($($arg)*))
    };
}
