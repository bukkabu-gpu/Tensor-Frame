# Backend System

Tensor Frame uses a pluggable backend system that allows tensors to run on different computational devices. This page documents the backend architecture and API.

## Backend Trait

All backends implement the `Backend` trait:

```rust
pub trait Backend: Debug + Send + Sync {
    fn backend_type(&self) -> BackendType;
    fn is_available(&self) -> bool;
    
    // Tensor creation
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage>;
    fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage>;
    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage>;
    
    // Arithmetic operations
    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    
    // Matrix operations
    fn matmul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    
    // Reduction operations
    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage>;
    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage>;
    
    // Data access
    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>>;
}
```

## Storage Types

Each backend uses a different storage mechanism:

```rust
pub enum Storage {
    Cpu(Vec<f32>),                    // CPU: simple Vec
    Wgpu(WgpuStorage),                // WGPU: GPU buffer
    Cuda(CudaStorage),                // CUDA: device pointer
}

pub struct WgpuStorage {
    pub buffer: Arc<wgpu::Buffer>,    // WGPU buffer handle
}

pub struct CudaStorage {
    pub ptr: *mut f32,                // Raw CUDA device pointer
    pub len: usize,                   // Buffer length
}
```

## Backend Selection

### Automatic Selection

By default, Tensor Frame automatically selects the best available backend:

1. **CUDA** (if available and feature enabled)
2. **WGPU** (if available and feature enabled)  
3. **CPU** (always available)

```rust
// Uses automatic backend selection
let tensor = Tensor::zeros(vec![1000, 1000])?;
println!("Selected backend: {:?}", tensor.backend_type());
```

### Manual Selection

You can also explicitly specify backend priority:

```rust
use tensor_frame::backend::{set_backend_priority, BackendType};

// Force CPU backend
let cpu_backend = set_backend_priority(vec![BackendType::Cpu]);

// Prefer WGPU over CUDA
let gpu_backend = set_backend_priority(vec![
    BackendType::Wgpu,
    BackendType::Cuda, 
    BackendType::Cpu
]);
```

### Backend Conversion

Convert tensors between backends:

```rust
let cpu_tensor = Tensor::ones(vec![100, 100])?;

// Convert to GPU backend (if available)
let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;

// Convert back to CPU
let back_to_cpu = gpu_tensor.to_backend(BackendType::Cpu)?;
```

## Performance Characteristics

### CPU Backend
- **Pros**: Always available, good for small tensors, excellent for development
- **Cons**: Limited parallelism, slower for large operations
- **Best for**: Tensors < 10K elements, prototyping, fallback option
- **Implementation**: Uses Rayon for parallel CPU operations

### WGPU Backend  
- **Pros**: Cross-platform GPU support, works on Metal/Vulkan/DX12/OpenGL
- **Cons**: Compute shader overhead, limited by GPU memory
- **Best for**: Large tensor operations, cross-platform deployment
- **Implementation**: Compute shaders with buffer storage

### CUDA Backend
- **Pros**: Highest performance on NVIDIA GPUs, mature ecosystem
- **Cons**: NVIDIA-only, requires CUDA toolkit installation
- **Best for**: Production workloads on NVIDIA hardware
- **Implementation**: cuBLAS and custom CUDA kernels

## Backend Availability

Check backend availability at runtime:

```rust
use tensor_frame::backend::{cpu, wgpu, cuda};

// CPU backend is always available
println!("CPU available: {}", cpu::CpuBackend::new().is_available());

// Check GPU backends
#[cfg(feature = "wgpu")]
if let Ok(wgpu_backend) = wgpu::WgpuBackend::new() {
    println!("WGPU available: {}", wgpu_backend.is_available());
}

#[cfg(feature = "cuda")]
println!("CUDA available: {}", cuda::is_available());
```

## Cross-Backend Operations

Operations between tensors on different backends automatically handle conversion:

```rust
let cpu_tensor = Tensor::ones(vec![100])?;
let gpu_tensor = Tensor::zeros(vec![100])?.to_backend(BackendType::Wgpu)?;

// Automatically converts gpu_tensor to CPU backend for the operation
let result = cpu_tensor + gpu_tensor;  
```

## Custom Backends

You can implement custom backends by implementing the `Backend` trait:

```rust
#[derive(Debug)]
struct MyCustomBackend;

impl Backend for MyCustomBackend {
    fn backend_type(&self) -> BackendType {
        // Would need to extend BackendType enum
        BackendType::Custom
    }
    
    fn is_available(&self) -> bool {
        true  // Your availability logic
    }
    
    // Implement all required methods...
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        // Your implementation
    }
    
    // ... more methods
}
```

## Memory Management

### Reference Counting
- Tensors use `Arc<dyn Backend>` for backend sharing
- Storage is reference counted within each backend
- Automatic cleanup when last reference is dropped

### Cross-Backend Memory
- Converting between backends allocates new memory
- Original data remains valid until all references dropped
- No automatic synchronization between backends

### GPU Memory Management
- WGPU backend uses WGPU's automatic memory management
- CUDA backend manually manages device memory with proper cleanup
- Out-of-memory errors are propagated as `TensorError::BackendError`