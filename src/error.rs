use std::fmt;

#[derive(Debug, Clone)]
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidShape(String),
    InvalidIndex {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    BackendError(String),
    BroadcastError(String),
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            TensorError::InvalidIndex { index, shape } => {
                write!(f, "Invalid index {:?} for shape {:?}", index, shape)
            }
            TensorError::BackendError(msg) => write!(f, "Backend error: {}", msg),
            TensorError::BroadcastError(msg) => write!(f, "Broadcast error: {}", msg),
            TensorError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for TensorError {}

pub type Result<T> = std::result::Result<T, TensorError>;
