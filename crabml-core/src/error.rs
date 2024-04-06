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
    pub cause: Option<Arc<dyn std::error::Error>>,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
            cause: None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)?;
        if let Some(cause) = self.cause.as_ref() {
            write!(f, "\ncaused by: {}", cause)?;
        }
        Ok(())
    }
}

impl<S: Into<String>> From<(ErrorKind, S)> for Error {
    fn from((kind, message): (ErrorKind, S)) -> Self {
        Self {
            kind,
            message: message.into(),
            cause: None,
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
