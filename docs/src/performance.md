# Performance Guide

This guide provides detailed information on optimizing Tensor Frame performance across different backends and use cases.

## Performance Overview

Tensor Frame's performance characteristics vary significantly based on:
- **Tensor size**: Small vs large tensors have different optimal backends
- **Operation type**: Element-wise vs reductions vs matrix operations
- **Backend selection**: CPU vs WGPU vs CUDA performance profiles
- **Memory patterns**: Data locality and transfer overhead

## Backend Performance Characteristics

### CPU Backend
- **Best for**: Small tensors (< 10K elements), development, guaranteed availability
- **Strengths**: Low latency, no setup overhead, excellent debugging
- **Limitations**: Limited parallelism, memory bandwidth bound for large operations

```rust
use tensor_frame::Tensor;
// CPU optimal: Small tensors and scalar operations
let small = Tensor::ones(vec![100, 100])?;
let result = small.sum(None)?;  // ~0.1ms on modern CPU
```

### WGPU Backend  
- **Best for**: Large element-wise operations (> 100K elements), cross-platform deployment
- **Strengths**: Massive parallelism, good memory bandwidth, portable
- **Limitations**: GPU setup overhead (~1-10ms), limited operation support

```rust
use tensor_frame::Tensor;
// WGPU optimal: Large parallel operations
let large = Tensor::ones(vec![2048, 2048])?
    .to_backend(BackendType::Wgpu)?;
let result = (large_a * large_b) + large_c;  // ~2ms on modern GPU
```

### CUDA Backend
- **Best for**: Very large operations (> 1M elements), production workloads
- **Strengths**: Peak performance, mature optimizations, cuBLAS integration
- **Limitations**: NVIDIA-only, CUDA toolkit requirement

```rust
use tensor_frame::Tensor;
// CUDA optimal: Matrix operations and very large tensors
let matrices = Tensor::ones(vec![4096, 4096])?
    .to_backend(BackendType::Cuda)?;
let result = matrix_a.matmul(&matrix_b)?;  // ~15ms with cuBLAS
```

## Operation-Specific Performance

### Element-wise Operations

**Performance Scaling**:
- CPU: O(n) with thread-level parallelism (8-32 threads)
- WGPU: O(n) with massive parallelism (1000+ threads)  
- CUDA: O(n) with optimal parallelism (10000+ threads)

```rust
use std::time::Instant;

fn benchmark_element_wise() -> Result<()> {
    let sizes = vec![1000, 5000, 10000, 50000];
    
    for size in sizes {
        let a = Tensor::ones(vec![size, size])?;
        let b = Tensor::ones(vec![size, size])?;
        
        // CPU timing
        let start = Instant::now();
        let cpu_result = &a + &b;
        let cpu_time = start.elapsed();
        
        // GPU timing (if available)
        #[cfg(feature = "wgpu")]
        {
            let gpu_a = a.to_backend(BackendType::Wgpu)?;
            let gpu_b = b.to_backend(BackendType::Wgpu)?;
            
            let start = Instant::now();
            let gpu_result = &gpu_a + &gpu_b;
            let _sync = gpu_result.to_vec()?;
            let gpu_time = start.elapsed();
            
            let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
            println!("Size {}x{}: CPU {:?}, GPU {:?}, Speedup: {:.1}x", 
                    size, size, cpu_time, gpu_time, speedup);
        }
    }
    
    Ok(())
}
```


### Reduction Operations

**Performance Notes**:
- CPU: Rayon parallel reduction, cache-efficient
- GPU: Requires multiple kernel launches for large reductions
- Memory-bound for large tensors

```rust
fn reduction_performance() -> Result<()> {
    let tensor = Tensor::ones(vec![10000, 10000])?;  // 100M elements
    
    // Sum reduction timing
    let start = Instant::now();
    let sum = tensor.sum(None)?;
    let cpu_time = start.elapsed();
    
    println!("CPU sum reduction (100M elements): {:?}", cpu_time);
    println!("Result: {}", sum.to_vec()?[0]);
    
    Ok(())
}
```

## Memory Performance

### Memory Transfer Costs

GPU operations include memory transfer overhead:

```rust
fn memory_transfer_analysis() -> Result<()> {
    let sizes = vec![1000, 5000, 10000];
    
    for size in sizes {
        let tensor = Tensor::ones(vec![size, size])?;
        let elements = tensor.numel();
        let bytes = elements * 4;  // f32 = 4 bytes
        
        #[cfg(feature = "wgpu")]
        {
            // Time conversion to GPU
            let start = Instant::now();
            let gpu_tensor = tensor.to_backend(BackendType::Wgpu)?;
            let upload_time = start.elapsed();
            
            // Time conversion back to CPU
            let start = Instant::now();
            let _data = gpu_tensor.to_vec()?;
            let download_time = start.elapsed();
            
            let upload_bw = bytes as f64 / upload_time.as_secs_f64() / 1e9;  // GB/s
            let download_bw = bytes as f64 / download_time.as_secs_f64() / 1e9;  // GB/s
            
            println!("Size {}x{} ({} MB):", size, size, bytes / 1024 / 1024);
            println!("  Upload: {:?} ({:.1} GB/s)", upload_time, upload_bw);
            println!("  Download: {:?} ({:.1} GB/s)", download_time, download_bw);
        }
    }
    
    Ok(())
}
```

### Memory Layout Optimization

```rust
// Efficient: Contiguous memory access
let matrix = Tensor::from_vec(data, vec![rows, cols])?;
let transposed = matrix.transpose()?;  // May require memory copy

// Efficient: Operations that preserve layout
let result = (&matrix_a + &matrix_b) * 2.0;  // All operations maintain layout

// Less efficient: Operations that break layout
let reshaped = matrix.reshape(vec![cols, rows])?;  // May require copy
```

## Optimization Strategies

### 1. Backend Selection Strategy

```rust
fn optimal_backend_for_workload(tensor_size: usize, operation: &str) -> BackendType {
    match (tensor_size, operation) {
        // Small tensors: CPU always optimal
        (0..=10_000, _) => BackendType::Cpu,
        
        // Large reductions: Prefer CUDA
        (_, "reduction") if tensor_size > 1_000_000 => {
            #[cfg(feature = "cuda")]
            { BackendType::Cuda }
            #[cfg(not(feature = "cuda"))]
            { BackendType::Cpu }
        }
        
        // Large element-wise: GPU beneficial
        (10_001..=1_000_000, "elementwise") => {
            #[cfg(feature = "wgpu")]
            { BackendType::Wgpu }
            #[cfg(not(feature = "wgpu"))]
            { BackendType::Cpu }
        }
        
        // Very large: Prefer CUDA > WGPU > CPU
        (1_000_001.., _) => {
            #[cfg(feature = "cuda")]
            { BackendType::Cuda }
            #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
            { BackendType::Wgpu }
            #[cfg(all(not(feature = "wgpu"), not(feature = "cuda")))]
            { BackendType::Cpu }
        }
        
        // Default: CPU
        _ => BackendType::Cpu,
    }
}
```

### 2. Operation Fusion

```rust
// Efficient: Fused operations
let result = ((a * b) + c) / d;  // Single expression, potential fusion

// Less efficient: Separate operations  
let temp1 = a * b;
let temp2 = temp1 + c;
let result = temp2 / d;  // Multiple temporary allocations
```

### 3. Batch Processing

```rust
fn efficient_batch_processing(batches: Vec<Tensor>) -> Result<Vec<Tensor>> {
    // Convert all to same backend once
    let backend = BackendType::Wgpu;
    let gpu_batches: Result<Vec<_>> = batches
        .into_iter()
        .map(|t| t.to_backend(backend))
        .collect();
    
    // Process on GPU
    gpu_batches?
        .into_iter()
        .map(|batch| {
            // Heavy computation on GPU
            (batch * 2.0) + 1.0
        })
        .collect()
}
```

### 4. Memory Pool Usage

```rust
// Efficient: Reuse similar-sized tensors
struct TensorPool {
    cached_tensors: HashMap<Vec<usize>, Vec<Tensor>>,
}

impl TensorPool {
    fn get_or_create(&mut self, shape: Vec<usize>) -> Result<Tensor> {
        if let Some(cached) = self.cached_tensors.get_mut(&shape) {
            if let Some(tensor) = cached.pop() {
                return Ok(tensor);
            }
        }
        
        // Create new tensor if no cached version
        Tensor::zeros(shape)
    }
    
    fn return_tensor(&mut self, tensor: Tensor) {
        let shape = tensor.shape().dims().to_vec();
        self.cached_tensors
            .entry(shape)
            .or_insert_with(Vec::new)
            .push(tensor);
    }
}
```

## Profiling and Debugging

### CPU Profiling

```rust
// Use built-in timing
use std::time::Instant;

let start = Instant::now();
let result = expensive_operation()?;
println!("Operation took: {:?}", start.elapsed());

// Use external profilers
// cargo install flamegraph
// cargo flamegraph --bin your_app
```

### GPU Profiling

**NVIDIA Tools** (for CUDA backend):
```bash
# Nsight Systems for timeline analysis
nsys profile --stats=true ./your_app

# Nsight Compute for kernel analysis  
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./your_app
```

**Platform Tools** (for WGPU backend):
- **Windows**: PIX for Windows, RenderDoc
- **macOS**: Xcode Instruments (GPU Timeline)
- **Linux**: RenderDoc, Vulkan Tools

### Memory Profiling

```rust
fn memory_usage_analysis() -> Result<()> {
    use std::alloc::{GlobalAlloc, Layout, System};
    
    // Monitor system memory usage
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status")?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                println!("Memory usage: {}", line);
            }
        }
    }
    
    // GPU memory monitoring (platform-specific)
    #[cfg(feature = "cuda")]
    {
        // CUDA memory info
        let (free, total) = cuda::memory_info()?;
        println!("GPU memory: {} MB free of {} MB total", 
                free / 1024 / 1024, total / 1024 / 1024);
    }
    
    Ok(())
}
```

## Performance Benchmarking

### Comprehensive Benchmark Suite

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_tensor_operations(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 2000];
    
    for size in sizes {
        let a = Tensor::ones(vec![size, size]).unwrap();
        let b = Tensor::ones(vec![size, size]).unwrap();
        
        // CPU benchmark
        c.bench_function(&format!("cpu_add_{}x{}", size, size), |bench| {
            bench.iter(|| {
                let _result = &a + &b;
            });
        });
        
        // GPU benchmark (if available)
        #[cfg(feature = "wgpu")]
        {
            let gpu_a = a.to_backend(BackendType::Wgpu).unwrap();
            let gpu_b = b.to_backend(BackendType::Wgpu).unwrap();
            
            c.bench_function(&format!("gpu_add_{}x{}", size, size), |bench| {
                bench.iter(|| {
                    let result = &gpu_a + &gpu_b;
                    let _sync = result.to_vec().unwrap();  // Force sync
                });
            });
        }
    }
}

criterion_group!(benches, bench_tensor_operations);
criterion_main!(benches);
```

## Performance Troubleshooting

### Common Performance Issues

1. **Small Tensors on GPU**
```rust
// Problem: GPU overhead for small operations
let small = Tensor::ones(vec![10, 10])?;
let slow = small.to_backend(BackendType::Wgpu)?;  // Overhead > computation

// Solution: Use CPU for small tensors
let fast = small;  // Stay on CPU
```

2. **Frequent Backend Conversions**
```rust
// Problem: Repeated conversions
for i in 0..1000 {
    let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;
    let result = gpu_tensor + 1.0;
    let back_to_cpu = result.to_backend(BackendType::Cpu)?;
}

// Solution: Convert once
let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;
for i in 0..1000 {
    gpu_tensor = gpu_tensor + 1.0;  // Stay on GPU
}
let final_result = gpu_tensor.to_backend(BackendType::Cpu)?;
```

3. **Memory Fragmentation**
```rust
// Problem: Large temporary allocations
let huge_temp = (huge_a * huge_b) + huge_c;  // 3 large tensors in memory

// Solution: In-place operations (when available)
let result = huge_a.mul_add(&huge_b, &huge_c)?;  // Hypothetical in-place op
```

### Performance Debugging Checklist

1. **Profile first**: Measure before optimizing
2. **Check backend selection**: Ensure optimal backend for workload
3. **Monitor memory transfers**: GPU transfer costs often dominate
4. **Verify operation fusion**: Combine operations when possible
5. **Consider batch size**: Larger batches amortize overhead
6. **Test different tensor sizes**: Performance characteristics vary by size
7. **Use appropriate data types**: f32 vs f64 performance difference
8. **Monitor memory usage**: Avoid memory pressure and swapping

## Hardware-Specific Optimization

### CPU Optimization
- Use all available cores (Rayon handles this automatically)
- Ensure sufficient memory bandwidth
- Consider NUMA topology for large systems
- Link with optimized BLAS (OpenBLAS, Intel MKL)

### GPU Optimization  
- Ensure sufficient GPU memory
- Consider tensor sizes that align with GPU architecture
- Use appropriate batch sizes for GPU utilization
- Monitor thermal throttling on mobile/laptop GPUs

### Memory Hierarchy
- L1/L2 cache: Small frequently-accessed tensors
- System RAM: Medium tensors and CPU operations
- GPU VRAM: Large tensors for GPU operations
- Storage: Streaming large datasets

## Conclusion

Tensor Frame performance optimization requires understanding:
1. **Workload characteristics**: Size, operations, access patterns
2. **Backend strengths**: CPU for small/mixed, GPU for large parallel
3. **Memory costs**: Transfer overhead, allocation patterns
4. **Platform specifics**: Hardware capabilities and limitations

Use profiling tools to guide optimization decisions and always measure performance improvements to ensure they provide real benefits for your specific use case.