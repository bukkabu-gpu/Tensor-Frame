use crate::error::Result;

pub trait TensorOps {
    fn matmul(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;
    fn sum(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;
    fn mean(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;

    fn reshape(&self, shape: Vec<usize>) -> Result<Self>
    where
        Self: Sized;
    fn transpose(&self) -> Result<Self>
    where
        Self: Sized;
    fn squeeze(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;
    fn unsqueeze(&self, axis: usize) -> Result<Self>
    where
        Self: Sized;
}
