use crate::error::Result;
use crate::tensor::{shape::Shape, dtype::DType};
use std::fmt::Debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

pub mod cpu;
#[cfg(feature = "wgpu")]
pub mod wgpu;
#[cfg(feature = "cuda")]
pub mod cuda;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

pub trait Backend: Debug + Send + Sync {
    fn backend_type(&self) -> BackendType;
    fn is_available(&self) -> bool { true }
    
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage>;
    fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage>;
    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage>;
    
    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    
    fn matmul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage>;
    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage>;
    
    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(Vec<f32>),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaStorage {
    pub ptr: *mut f32,
    pub len: usize,
}

#[cfg(feature = "wgpu")]
#[derive(Debug, Clone)]
pub struct WgpuStorage {
    pub buffer: std::sync::Arc<::wgpu::Buffer>,
}

pub static BACKEND: Lazy<Arc<dyn Backend>> = Lazy::new(|| {
    get_backend_with_priority(&default_backend_priority())
});

fn default_backend_priority() -> Vec<BackendType> {
    let mut priority = Vec::new();
    
    #[cfg(feature = "cuda")]
    priority.push(BackendType::Cuda);
    
    #[cfg(feature = "wgpu")]
    priority.push(BackendType::Wgpu);
    
    priority.push(BackendType::Cpu);
    
    priority
}

pub fn get_backend_with_priority(priority: &[BackendType]) -> Arc<dyn Backend> {
    for backend_type in priority {
        match backend_type {
            BackendType::Cpu => {
                let backend = cpu::CpuBackend::new();
                if backend.is_available() {
                    return Arc::new(backend);
                }
            }
            #[cfg(feature = "wgpu")]
            BackendType::Wgpu => {
                if let Ok(backend) = wgpu::WgpuBackend::new() {
                    if backend.is_available() {
                        return Arc::new(backend);
                    }
                }
            }
            #[cfg(feature = "cuda")]
            BackendType::Cuda => {
                if cuda::is_available() {
                    if let Ok(backend) = cuda::CudaBackend::new() {
                        if backend.is_available() {
                            return Arc::new(backend);
                        }
                    }
                }
            }
        }
    }
    
    // Fallback to CPU backend
    Arc::new(cpu::CpuBackend::new())
}

pub fn set_backend_priority(priority: Vec<BackendType>) -> Arc<dyn Backend> {
    get_backend_with_priority(&priority)
}