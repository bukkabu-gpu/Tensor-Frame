use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
        }
    }
}
