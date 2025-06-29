use super::{Backend, BackendType, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::{dtype::DType, shape::Shape};
use rayon::prelude::*;

#[derive(Debug)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn zeros(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        let size = shape.numel();
        Ok(Storage::Cpu(vec![0.0; size]))
    }

    fn ones(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        let size = shape.numel();
        Ok(Storage::Cpu(vec![1.0; size]))
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        Ok(Storage::Cpu(data.to_vec()))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cpu(a), Storage::Cpu(b)) => {
                if a.len() != b.len() {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len()],
                        got: vec![b.len()],
                    });
                }
                let result: Vec<f32> = a.par_iter().zip(b.par_iter()).map(|(x, y)| x + y).collect();
                Ok(Storage::Cpu(result))
            }
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cpu(a), Storage::Cpu(b)) => {
                if a.len() != b.len() {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len()],
                        got: vec![b.len()],
                    });
                }
                let result: Vec<f32> = a.par_iter().zip(b.par_iter()).map(|(x, y)| x - y).collect();
                Ok(Storage::Cpu(result))
            }
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cpu(a), Storage::Cpu(b)) => {
                if a.len() != b.len() {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len()],
                        got: vec![b.len()],
                    });
                }
                let result: Vec<f32> = a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).collect();
                Ok(Storage::Cpu(result))
            }
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cpu(a), Storage::Cpu(b)) => {
                if a.len() != b.len() {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len()],
                        got: vec![b.len()],
                    });
                }
                let result: Vec<f32> = a.par_iter().zip(b.par_iter()).map(|(x, y)| x / y).collect();
                Ok(Storage::Cpu(result))
            }
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }

    fn matmul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        // Simplified matmul - needs proper implementation
        match (lhs, rhs) {
            (Storage::Cpu(_a), Storage::Cpu(_b)) => Err(TensorError::BackendError(
                "Matmul not yet implemented".to_string(),
            )),
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }

    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        match storage {
            Storage::Cpu(data) => {
                if axis.is_some() {
                    return Err(TensorError::BackendError(
                        "Axis sum not yet implemented".to_string(),
                    ));
                }
                let sum: f32 = data.par_iter().sum();
                Ok(Storage::Cpu(vec![sum]))
            }
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }

    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        match storage {
            Storage::Cpu(data) => {
                if axis.is_some() {
                    return Err(TensorError::BackendError(
                        "Axis mean not yet implemented".to_string(),
                    ));
                }
                let sum: f32 = data.par_iter().sum();
                let mean = sum / data.len() as f32;
                Ok(Storage::Cpu(vec![mean]))
            }
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        match storage {
            Storage::Cpu(data) => Ok(data.clone()),
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            _ => Err(TensorError::BackendError(
                "CPU backend can only operate on CPU storage".to_string(),
            )),
        }
    }
}
