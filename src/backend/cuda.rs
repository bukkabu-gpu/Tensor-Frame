use super::{Backend, BackendType, CudaStorage, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::{dtype::DType, shape::Shape};

#[derive(Debug)]
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // Try to initialize CUDA device
            match cudarc::driver::CudaDevice::new(0) {
                Ok(device) => Ok(CudaBackend {
                    device: std::sync::Arc::new(device),
                }),
                Err(e) => Err(TensorError::BackendError(format!(
                    "Failed to initialize CUDA: {}",
                    e
                ))),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError(
                "CUDA support not compiled in".to_string(),
            ))
        }
    }
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Try to create a simple CUDA device to check availability
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

impl Backend for CudaBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    fn is_available(&self) -> bool {
        is_available()
    }

    fn zeros(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            let size = shape.numel();
            let byte_size = size * std::mem::size_of::<f32>();
            
            match self.device.alloc_zeros::<f32>(size) {
                Ok(buf) => {
                    let ptr = buf.device_ptr() as *mut f32;
                    // Store the buffer to keep it alive - in real implementation
                    // you'd want to store this properly
                    std::mem::forget(buf);
                    Ok(Storage::Cuda(CudaStorage {
                        ptr,
                        len: size,
                    }))
                }
                Err(e) => Err(TensorError::BackendError(format!(
                    "Failed to allocate CUDA memory: {}",
                    e
                ))),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError(
                "CUDA support not compiled in".to_string(),
            ))
        }
    }

    fn ones(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            // For now, allocate zeros and then fill with ones
            // In a real implementation, you'd use a CUDA kernel
            let zeros_storage = self.zeros(shape, _dtype)?;
            // TODO: Launch CUDA kernel to fill with ones
            // For now, return an error to indicate not implemented
            Err(TensorError::BackendError(
                "CUDA ones not yet implemented".to_string(),
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError(
                "CUDA support not compiled in".to_string(),
            ))
        }
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            if data.len() != shape.numel() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![shape.numel()],
                    got: vec![data.len()],
                });
            }

            match self.device.htod_copy(data.to_vec()) {
                Ok(buf) => {
                    let ptr = buf.device_ptr() as *mut f32;
                    std::mem::forget(buf); // Keep buffer alive
                    Ok(Storage::Cuda(CudaStorage {
                        ptr,
                        len: data.len(),
                    }))
                }
                Err(e) => Err(TensorError::BackendError(format!(
                    "Failed to copy data to CUDA: {}",
                    e
                ))),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError(
                "CUDA support not compiled in".to_string(),
            ))
        }
    }

    fn add(&self, _lhs: &Storage, _rhs: &Storage) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA operations not yet implemented".to_string(),
        ))
    }

    fn sub(&self, _lhs: &Storage, _rhs: &Storage) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA operations not yet implemented".to_string(),
        ))
    }

    fn mul(&self, _lhs: &Storage, _rhs: &Storage) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA operations not yet implemented".to_string(),
        ))
    }

    fn div(&self, _lhs: &Storage, _rhs: &Storage) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA operations not yet implemented".to_string(),
        ))
    }

    fn matmul(&self, _lhs: &Storage, _rhs: &Storage) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA matmul not yet implemented".to_string(),
        ))
    }

    fn sum(&self, _storage: &Storage, _axis: Option<usize>) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA sum not yet implemented".to_string(),
        ))
    }

    fn mean(&self, _storage: &Storage, _axis: Option<usize>) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA mean not yet implemented".to_string(),
        ))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    // For now, return an error since we need proper device buffer management
                    Err(TensorError::BackendError(
                        "CUDA to_vec not yet implemented - needs proper buffer management".to_string(),
                    ))
                }
                _ => Err(TensorError::BackendError(
                    "Invalid storage type for CUDA backend".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError(
                "CUDA support not compiled in".to_string(),
            ))
        }
    }
}