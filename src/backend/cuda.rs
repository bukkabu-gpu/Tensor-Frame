use super::{Backend, BackendType, Storage, CudaStorage};
use crate::error::{Result, TensorError};
use crate::tensor::{shape::Shape, dtype::DType};

#[derive(Debug)]
pub struct CudaBackend;

impl CudaBackend {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // Check if CUDA is available
            if !is_available() {
                return Err(TensorError::BackendError("CUDA not available".to_string()));
            }
            Ok(CudaBackend)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
        }
    }
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Try to create a simple CUDA context to check availability
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
            use cudarc::driver::CudaDevice;
            
            let device = CudaDevice::new(0)
                .map_err(|e| TensorError::BackendError(format!("Failed to create CUDA device: {:?}", e)))?;
            
            let size = shape.numel();
            let zeros = vec![0.0f32; size];
            let device_ptr = device.htod_copy(zeros)
                .map_err(|e| TensorError::BackendError(format!("Failed to copy to device: {:?}", e)))?;
            
            Ok(Storage::Cuda(CudaStorage {
                ptr: device_ptr.device_ptr() as *mut f32,
                len: size,
            }))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
        }
    }

    fn ones(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;
            
            let device = CudaDevice::new(0)
                .map_err(|e| TensorError::BackendError(format!("Failed to create CUDA device: {:?}", e)))?;
            
            let size = shape.numel();
            let ones = vec![1.0f32; size];
            let device_ptr = device.htod_copy(ones)
                .map_err(|e| TensorError::BackendError(format!("Failed to copy to device: {:?}", e)))?;
            
            Ok(Storage::Cuda(CudaStorage {
                ptr: device_ptr.device_ptr() as *mut f32,
                len: size,
            }))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
        }
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;
            
            let device = CudaDevice::new(0)
                .map_err(|e| TensorError::BackendError(format!("Failed to create CUDA device: {:?}", e)))?;
            
            let device_ptr = device.htod_copy(data.to_vec())
                .map_err(|e| TensorError::BackendError(format!("Failed to copy to device: {:?}", e)))?;
            
            Ok(Storage::Cuda(CudaStorage {
                ptr: device_ptr.device_ptr() as *mut f32,
                len: data.len(),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
        }
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cuda(a), Storage::Cuda(b)) => {
                if a.len != b.len {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len],
                        got: vec![b.len],
                    });
                }
                
                #[cfg(feature = "cuda")]
                {
                    // Simple CPU fallback for arithmetic operations
                    // In a real implementation, you'd use CUDA kernels
                    let a_data = self.cuda_to_vec(a)?;
                    let b_data = self.cuda_to_vec(b)?;
                    
                    let result_data: Vec<f32> = a_data.iter()
                        .zip(b_data.iter())
                        .map(|(x, y)| x + y)
                        .collect();
                    
                    self.vec_to_cuda(&result_data)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
                }
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cuda(a), Storage::Cuda(b)) => {
                if a.len != b.len {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len],
                        got: vec![b.len],
                    });
                }
                
                #[cfg(feature = "cuda")]
                {
                    let a_data = self.cuda_to_vec(a)?;
                    let b_data = self.cuda_to_vec(b)?;
                    
                    let result_data: Vec<f32> = a_data.iter()
                        .zip(b_data.iter())
                        .map(|(x, y)| x - y)
                        .collect();
                    
                    self.vec_to_cuda(&result_data)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
                }
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cuda(a), Storage::Cuda(b)) => {
                if a.len != b.len {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len],
                        got: vec![b.len],
                    });
                }
                
                #[cfg(feature = "cuda")]
                {
                    let a_data = self.cuda_to_vec(a)?;
                    let b_data = self.cuda_to_vec(b)?;
                    
                    let result_data: Vec<f32> = a_data.iter()
                        .zip(b_data.iter())
                        .map(|(x, y)| x * y)
                        .collect();
                    
                    self.vec_to_cuda(&result_data)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
                }
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Cuda(a), Storage::Cuda(b)) => {
                if a.len != b.len {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a.len],
                        got: vec![b.len],
                    });
                }
                
                #[cfg(feature = "cuda")]
                {
                    let a_data = self.cuda_to_vec(a)?;
                    let b_data = self.cuda_to_vec(b)?;
                    
                    let result_data: Vec<f32> = a_data.iter()
                        .zip(b_data.iter())
                        .map(|(x, y)| x / y)
                        .collect();
                    
                    self.vec_to_cuda(&result_data)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
                }
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn matmul(&self, _lhs: &Storage, _rhs: &Storage) -> Result<Storage> {
        Err(TensorError::BackendError("Matmul not yet implemented for CUDA".to_string()))
    }

    fn sum(&self, _storage: &Storage, _axis: Option<usize>) -> Result<Storage> {
        Err(TensorError::BackendError("Sum not yet implemented for CUDA".to_string()))
    }

    fn mean(&self, _storage: &Storage, _axis: Option<usize>) -> Result<Storage> {
        Err(TensorError::BackendError("Mean not yet implemented for CUDA".to_string()))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        match storage {
            Storage::Cuda(cuda_storage) => self.cuda_to_vec(cuda_storage),
            _ => Err(TensorError::BackendError("Invalid storage type".to_string())),
        }
    }
}

impl CudaBackend {
    #[cfg(feature = "cuda")]
    fn cuda_to_vec(&self, cuda_storage: &CudaStorage) -> Result<Vec<f32>> {
        use cudarc::driver::{CudaDevice, DevicePtr};
        
        let device = CudaDevice::new(0)
            .map_err(|e| TensorError::BackendError(format!("Failed to create CUDA device: {:?}", e)))?;
        
        // Create a DevicePtr from the raw pointer - this is unsafe but necessary
        let device_ptr = unsafe { 
            DevicePtr::<f32>::from_raw(cuda_storage.ptr as cudarc::driver::sys::CUdeviceptr)
        };
        
        let result = device.dtoh_sync_copy(&device_ptr)
            .map_err(|e| TensorError::BackendError(format!("Failed to copy from device: {:?}", e)))?;
        
        Ok(result)
    }
    
    #[cfg(not(feature = "cuda"))]
    fn cuda_to_vec(&self, _cuda_storage: &CudaStorage) -> Result<Vec<f32>> {
        Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
    }
    
    #[cfg(feature = "cuda")]
    fn vec_to_cuda(&self, data: &[f32]) -> Result<Storage> {
        use cudarc::driver::CudaDevice;
        
        let device = CudaDevice::new(0)
            .map_err(|e| TensorError::BackendError(format!("Failed to create CUDA device: {:?}", e)))?;
        
        let device_ptr = device.htod_copy(data.to_vec())
            .map_err(|e| TensorError::BackendError(format!("Failed to copy to device: {:?}", e)))?;
        
        Ok(Storage::Cuda(CudaStorage {
            ptr: device_ptr.device_ptr() as *mut f32,
            len: data.len(),
        }))
    }
    
    #[cfg(not(feature = "cuda"))]
    fn vec_to_cuda(&self, _data: &[f32]) -> Result<Storage> {
        Err(TensorError::BackendError("CUDA support not compiled in".to_string()))
    }
}