#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ErrorKind {
    /// Unexpected error
    Unexpected,

    /// raised on any IO error, like EOF
    IOError,

    /// raised when parsing GGUF or other kind of model files
    FormatError,

    /// raised on manuplating tensors, like dimension mismatch
    TensorError,
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub cause: Option<Box<dyn std::error::Error>>,
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

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;