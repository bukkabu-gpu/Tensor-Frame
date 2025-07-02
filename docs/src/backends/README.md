# Backends Overview

Tensor Frame's backend system provides a pluggable architecture for running tensor operations on different computational devices. This allows the same high-level tensor API to transparently utilize CPU cores, integrated GPUs, discrete GPUs, and specialized accelerators.

## Available Backends

| Backend | Feature Flag | Availability | Best Use Cases |
|---------|-------------|--------------|----------------|
| **CPU** | `cpu` (default) | Always | Small tensors, development, fallback |
| **WGPU** | `wgpu` | Cross-platform GPU | Large tensors, cross-platform deployment |
| **CUDA** | `cuda` | NVIDIA GPUs | High-performance production workloads |

## Backend Selection Strategy

### Automatic Selection (Recommended)

By default, Tensor Frame automatically selects the best available backend using this priority order:

1. **CUDA** - Highest performance on NVIDIA hardware
2. **WGPU** - Cross-platform GPU acceleration  
3. **CPU** - Universal fallback

```rust
use tensor_frame::Tensor;

// Automatically uses best available backend
let tensor = Tensor::zeros(vec![1000, 1000])?;
println!("Using backend: {:?}", tensor.backend_type());
```

### Manual Backend Control

For specific requirements, you can control backend selection:

```rust
use tensor_frame::backend::{set_backend_priority, BackendType};

// Force CPU-only execution
let backend = set_backend_priority(vec![BackendType::Cpu]);

// Prefer WGPU over CUDA
let backend = set_backend_priority(vec![
    BackendType::Wgpu,
    BackendType::Cuda,
    BackendType::Cpu
]);
```

### Per-Tensor Backend Conversion

Convert individual tensors between backends:

```rust
let cpu_tensor = Tensor::ones(vec![100, 100])?;

// Move to GPU
let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;

// Move back to CPU  
let back_to_cpu = gpu_tensor.to_backend(BackendType::Cpu)?;
```

## Performance Characteristics

### CPU Backend
- **Latency**: Lowest for small operations (< 1ms)
- **Throughput**: Limited by CPU cores and memory bandwidth
- **Memory**: System RAM (typically abundant)
- **Parallelism**: Thread-level via Rayon
- **Overhead**: Minimal function call overhead

### WGPU Backend  
- **Latency**: Higher initialization cost (~1-10ms)
- **Throughput**: High for large, parallel operations
- **Memory**: GPU memory (limited but fast)
- **Parallelism**: Massive thread-level via compute shaders
- **Overhead**: GPU command submission and synchronization

### CUDA Backend
- **Latency**: Moderate initialization cost (~0.1-1ms)
- **Throughput**: Highest for supported operations
- **Memory**: GPU memory with CUDA optimizations
- **Parallelism**: Optimal GPU utilization via cuBLAS/cuDNN
- **Overhead**: Minimal with mature driver stack

## When to Use Each Backend

### CPU Backend
```rust
// Good for:
let small_tensor = Tensor::ones(vec![10, 10])?;        // Small tensors
let dev_tensor = Tensor::zeros(vec![100])?;            // Development/testing
let scalar_ops = tensor.sum(None)?;                    // Scalar results

// Avoid for:
// - Large matrix multiplications (> 1000x1000)
// - Batch operations on many tensors
// - Compute-intensive element-wise operations
```

### WGPU Backend
```rust
// Good for:
let large_tensor = Tensor::zeros(vec![2048, 2048])?;   // Large tensors
let batch_ops = tensors.iter().map(|t| t * 2.0);      // Batch operations
let element_wise = (a * b) + c;                       // Complex element-wise

// Consider for:
// - Cross-platform deployment
// - When CUDA is not available
// - Mixed CPU/GPU workloads
```

### CUDA Backend
```rust
// Excellent for:
let huge_tensor = Tensor::zeros(vec![4096, 4096])?;    // Very large tensors
let matrix_mul = a.matmul(&b)?;                        // Matrix operations
let ml_workload = model.forward(input)?;               // ML training/inference

// Best when:
// - NVIDIA GPU available
// - Performance is critical
// - Using alongside other CUDA libraries
```

## Cross-Backend Operations

Operations between tensors on different backends automatically handle conversion:

```rust
let cpu_a = Tensor::ones(vec![1000])?;
let gpu_b = Tensor::zeros(vec![1000])?.to_backend(BackendType::Wgpu)?;

// Automatically converts to common backend
let result = cpu_a + gpu_b;  // Runs on CPU backend
```

**Conversion Rules**:
1. If backends match, operation runs on that backend
2. If backends differ, converts to the "lower priority" backend
3. Priority order: CPU > WGPU > CUDA (CPU is most compatible)

## Memory Management

### Reference Counting
All backends use reference counting for efficient memory management:

```rust
let tensor1 = Tensor::ones(vec![1000, 1000])?;
let tensor2 = tensor1.clone();  // O(1) - just increments reference count

// Memory freed automatically when last reference dropped
```

### Cross-Backend Memory
Converting between backends allocates new memory:

```rust
let cpu_tensor = Tensor::ones(vec![1000])?;         // 4KB CPU memory
let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;  // +4KB GPU memory

// Both tensors exist independently until dropped
```

### Memory Usage Guidelines

- **Development**: Use CPU backend to avoid GPU memory pressure
- **Production**: Convert to GPU early, minimize cross-backend copies
- **Mixed workloads**: Keep frequently-accessed tensors on CPU
- **Large datasets**: Stream data through GPU backends

## Error Handling

Backend operations can fail for various reasons:

```rust
match Tensor::zeros(vec![100000, 100000]) {
    Ok(tensor) => println!("Created tensor on {:?}", tensor.backend_type()),
    Err(TensorError::BackendError(msg)) => {
        eprintln!("Backend error: {}", msg);
        // Fallback to smaller size or different backend
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

**Common Error Scenarios**:
- **GPU Out of Memory**: Try smaller tensors or CPU backend  
- **Backend Unavailable**: Fallback to CPU backend
- **Feature Not Implemented**: Some operations only available on certain backends
- **Cross-Backend Type Mismatch**: Ensure compatible data types

## Backend Implementation Status

| Operation | CPU | WGPU | CUDA |
|-----------|-----|------|------|
| Basic arithmetic (+, -, *, /) | ✅ | ✅ | ✅ |
| Reductions (sum, mean) | ✅ | ❌ | ✅ |
| Reshape, transpose | ✅ | ✅ | ✅ |
| Broadcasting | ✅ | ✅ | ✅ |

✅ = Fully implemented  
❌ = Not yet implemented  
⚠️ = Partially implemented