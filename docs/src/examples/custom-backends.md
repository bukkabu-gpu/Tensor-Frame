# Custom Backend Examples

This guide demonstrates how to effectively use different computational backends in Tensor Frame, including when to switch backends, performance optimization strategies, and mixed backend workflows.

## Backend Selection Strategies

### Automatic vs Manual Selection

```rust
use tensor_frame::{Tensor, BackendType, Result};
use std::time::Instant;

fn backend_selection_demo() -> Result<()> {
    println!("=== Backend Selection Strategies ===\n");
    
    // Automatic selection (recommended for most cases)
    let auto_tensor = Tensor::zeros(vec![1000, 1000])?;
    println!("Automatic backend selected: {:?}", auto_tensor.backend_type());
    
    // Manual backend specification
    let cpu_tensor = auto_tensor.to_backend(BackendType::Cpu)?;
    println!("Forced CPU backend: {:?}", cpu_tensor.backend_type());
    
    #[cfg(feature = "wgpu")]
    {
        match auto_tensor.to_backend(BackendType::Wgpu) {
            Ok(wgpu_tensor) => {
                println!("WGPU backend available: {:?}", wgpu_tensor.backend_type());
            }
            Err(e) => {
                println!("WGPU backend not available: {}", e);
            }
        }
    }
    
    #[cfg(feature = "cuda")]
    {
        match auto_tensor.to_backend(BackendType::Cuda) {
            Ok(cuda_tensor) => {
                println!("CUDA backend available: {:?}", cuda_tensor.backend_type());
            }
            Err(e) => {
                println!("CUDA backend not available: {}", e);
            }
        }
    }
    
    Ok(())
}
```

### Size-Based Backend Selection

```rust
fn adaptive_backend_selection() -> Result<()> {
    println!("=== Adaptive Backend Selection ===\n");
    
    let sizes = vec![
        (vec![10, 10], "tiny"),
        (vec![100, 100], "small"), 
        (vec![1000, 1000], "medium"),
        (vec![3000, 3000], "large"),
    ];
    
    for (shape, description) in sizes {
        let elements = shape.iter().product::<usize>();
        
        // Choose backend based on tensor size
        let backend = if elements < 1000 {
            BackendType::Cpu  // CPU overhead minimal for small tensors
        } else if elements < 1_000_000 {
            // Try WGPU first, fallback to CPU
            #[cfg(feature = "wgpu")]
            { BackendType::Wgpu }
            #[cfg(not(feature = "wgpu"))]
            { BackendType::Cpu }
        } else {
            // Large tensors: prefer CUDA > WGPU > CPU
            #[cfg(feature = "cuda")]
            { BackendType::Cuda }
            #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
            { BackendType::Wgpu }
            #[cfg(all(not(feature = "wgpu"), not(feature = "cuda")))]
            { BackendType::Cpu }
        };
        
        let tensor = Tensor::zeros(shape.clone())?;
        let optimized_tensor = tensor.to_backend(backend)?;
        
        println!("{} tensor {:?}: {} elements -> {:?} backend", 
                description, shape, elements, optimized_tensor.backend_type());
    }
    
    Ok(())
}
```

## Performance Benchmarking

### Backend Performance Comparison

```rust
fn benchmark_backends() -> Result<()> {
    println!("=== Backend Performance Comparison ===\n");
    
    let sizes = vec![
        vec![100, 100],
        vec![500, 500], 
        vec![1000, 1000],
        vec![2000, 2000],
    ];
    
    for size in sizes {
        println!("Benchmarking {}x{} matrix addition:", size[0], size[1]);
        
        // Create test tensors
        let a = Tensor::ones(size.clone())?;
        let b = Tensor::ones(size.clone())?;
        
        // CPU benchmark
        let cpu_a = a.to_backend(BackendType::Cpu)?;
        let cpu_b = b.to_backend(BackendType::Cpu)?;
        
        let start = Instant::now();
        let cpu_result = &cpu_a + &cpu_b;
        let cpu_time = start.elapsed();
        
        println!("  CPU: {:?}", cpu_time);
        
        // WGPU benchmark (if available)
        #[cfg(feature = "wgpu")]
        {
            match (a.to_backend(BackendType::Wgpu), b.to_backend(BackendType::Wgpu)) {
                (Ok(wgpu_a), Ok(wgpu_b)) => {
                    let start = Instant::now();
                    let wgpu_result = &wgpu_a + &wgpu_b;
                    // Force synchronization by converting back
                    let _sync = wgpu_result.to_vec()?;
                    let wgpu_time = start.elapsed();
                    
                    let speedup = cpu_time.as_nanos() as f64 / wgpu_time.as_nanos() as f64;
                    println!("  WGPU: {:?} ({}x speedup)", wgpu_time, speedup);
                }
                _ => println!("  WGPU: Not available"),
            }
        }
        
        // CUDA benchmark (if available)
        #[cfg(feature = "cuda")]
        {
            match (a.to_backend(BackendType::Cuda), b.to_backend(BackendType::Cuda)) {
                (Ok(cuda_a), Ok(cuda_b)) => {
                    let start = Instant::now();
                    let cuda_result = &cuda_a + &cuda_b;
                    let _sync = cuda_result.to_vec()?;
                    let cuda_time = start.elapsed();
                    
                    let speedup = cpu_time.as_nanos() as f64 / cuda_time.as_nanos() as f64;
                    println!("  CUDA: {:?} ({}x speedup)", cuda_time, speedup);
                }
                _ => println!("  CUDA: Not available"),
            }
        }
        
        println!();
    }
    
    Ok(())
}
```

### Operation-Specific Benchmarks

```rust
fn operation_benchmarks() -> Result<()> {
    println!("=== Operation-Specific Benchmarks ===\n");
    
    let size = vec![1000, 1000];
    let a = Tensor::ones(size.clone())?;
    let b = Tensor::ones(size.clone())?;
    
    // Test different operations
    let operations = vec![
        ("Addition", |a: &Tensor, b: &Tensor| a + b),
        ("Multiplication", |a: &Tensor, b: &Tensor| a * b),
        ("Complex", |a: &Tensor, b: &Tensor| (a * 2.0) + b),
    ];
    
    for (op_name, operation) in operations {
        println!("Operation: {}", op_name);
        
        // CPU timing
        let cpu_a = a.to_backend(BackendType::Cpu)?;
        let cpu_b = b.to_backend(BackendType::Cpu)?;
        
        let start = Instant::now();
        let _cpu_result = operation(&cpu_a, &cpu_b)?;
        let cpu_time = start.elapsed();
        
        println!("  CPU: {:?}", cpu_time);
        
        // GPU timing (if available)
        #[cfg(feature = "wgpu")]
        {
            if let (Ok(gpu_a), Ok(gpu_b)) = (
                a.to_backend(BackendType::Wgpu),
                b.to_backend(BackendType::Wgpu)
            ) {
                let start = Instant::now();
                let gpu_result = operation(&gpu_a, &gpu_b)?;
                let _sync = gpu_result.to_vec()?;  // Force sync
                let gpu_time = start.elapsed();
                
                let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
                println!("  GPU: {:?} ({}x speedup)", gpu_time, speedup);
            }
        }
        
        println!();
    }
    
    Ok(())
}
```

## Mixed Backend Workflows

### Pipeline with Backend Transitions

```rust
fn mixed_backend_pipeline() -> Result<()> {
    println!("=== Mixed Backend Pipeline ===\n");
    
    // Stage 1: Data preparation on CPU (I/O intensive)
    println!("Stage 1: Data preparation on CPU");
    let raw_data = vec![1.0; 1_000_000];  // Simulate data loading
    let cpu_tensor = Tensor::from_vec(raw_data, vec![1000, 1000])?;
    println!("  Created tensor on CPU: {:?}", cpu_tensor.backend_type());
    
    // Stage 2: Heavy computation on GPU
    #[cfg(feature = "wgpu")]
    {
        println!("Stage 2: Moving to GPU for computation");
        let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;
        println!("  Moved to GPU: {:?}", gpu_tensor.backend_type());
        
        // Perform heavy computations on GPU
        let processed = (&gpu_tensor * 2.0) + 1.0;
        let normalized = &processed / processed.sum(None)?;
        
        println!("  Completed GPU computations");
        
        // Stage 3: Results back to CPU for output
        println!("Stage 3: Moving results back to CPU");
        let final_result = normalized.to_backend(BackendType::Cpu)?;
        println!("  Final result on CPU: {:?}", final_result.backend_type());
        
        // Stage 4: Extract specific values (CPU efficient)
        let summary = final_result.sum(None)?;
        println!("  Summary value: {}", summary.to_vec()?[0]);
    }
    
    #[cfg(not(feature = "wgpu"))]
    {
        println!("Stage 2-4: Processing on CPU (GPU not available)");
        let processed = (&cpu_tensor * 2.0) + 1.0;
        let summary = processed.sum(None)?;
        println!("  Summary value: {}", summary.to_vec()?[0]);
    }
    
    Ok(())
}
```

### Batch Processing Strategy

```rust
fn batch_processing_strategy() -> Result<()> {
    println!("=== Batch Processing Strategy ===\n");
    
    // Simulate multiple data batches
    let batch_sizes = vec![100, 500, 1000, 2000];
    
    for batch_size in batch_sizes {
        println!("Processing batch size: {}", batch_size);
        
        // Create multiple tensors (simulating data batches)
        let batches: Result<Vec<_>> = (0..5)
            .map(|i| {
                let data = vec![i as f32; batch_size * batch_size];
                Tensor::from_vec(data, vec![batch_size, batch_size])
            })
            .collect();
        
        let batches = batches?;
        
        // Choose optimal backend based on batch size
        let backend = if batch_size < 500 {
            BackendType::Cpu
        } else {
            #[cfg(feature = "wgpu")]
            { BackendType::Wgpu }
            #[cfg(not(feature = "wgpu"))]
            { BackendType::Cpu }
        };
        
        let start = Instant::now();
        
        // Convert all batches to optimal backend
        let gpu_batches: Result<Vec<_>> = batches
            .into_iter()
            .map(|batch| batch.to_backend(backend))
            .collect();
        
        let gpu_batches = gpu_batches?;
        
        // Process all batches
        let results: Result<Vec<_>> = gpu_batches
            .iter()
            .map(|batch| batch.sum(None))
            .collect();
        
        let results = results?;
        let processing_time = start.elapsed();
        
        println!("  Backend: {:?}", backend);
        println!("  Processing time: {:?}", processing_time);
        println!("  Results count: {}", results.len());
        println!();
    }
    
    Ok(())
}
```

## Error Handling and Fallback Strategies

### Robust Backend Selection

```rust
fn robust_backend_selection(tensor: Tensor) -> Result<Tensor> {
    // Try backends in order of preference
    let backends_to_try = vec![
        #[cfg(feature = "cuda")]
        BackendType::Cuda,
        #[cfg(feature = "wgpu")]
        BackendType::Wgpu,
        BackendType::Cpu,
    ];
    
    for backend in backends_to_try {
        match tensor.to_backend(backend) {
            Ok(converted_tensor) => {
                println!("Successfully using backend: {:?}", backend);
                return Ok(converted_tensor);
            }
            Err(e) => {
                println!("Backend {:?} failed: {}", backend, e);
                continue;
            }
        }
    }
    
    // This should never happen since CPU should always work
    Err(tensor_frame::TensorError::BackendError(
        "No backend available".to_string()
    ))
}

fn robust_operation_with_fallback() -> Result<()> {
    println!("=== Robust Operation with Fallback ===\n");
    
    let large_tensor = Tensor::ones(vec![2000, 2000])?;
    
    // Try GPU operation first
    let result = match large_tensor.to_backend(BackendType::Wgpu) {
        Ok(gpu_tensor) => {
            match gpu_tensor.sum(None) {
                Ok(result) => {
                    println!("GPU operation successful");
                    result
                }
                Err(e) => {
                    println!("GPU operation failed: {}, falling back to CPU", e);
                    large_tensor.to_backend(BackendType::Cpu)?.sum(None)?
                }
            }
        }
        Err(e) => {
            println!("GPU conversion failed: {}, using CPU", e);
            large_tensor.sum(None)?
        }
    };
    
    println!("Final result: {}", result.to_vec()?[0]);
    
    Ok(())
}
```

### Memory Management Across Backends

```rust
fn memory_management_demo() -> Result<()> {
    println!("=== Memory Management Across Backends ===\n");
    
    // Monitor memory usage pattern
    let tensor_size = vec![1000, 1000];  // 4MB tensor
    
    // Start with CPU
    let cpu_tensor = Tensor::ones(tensor_size.clone())?;
    println!("Created tensor on CPU");
    
    // Convert to GPU (allocates GPU memory)
    #[cfg(feature = "wgpu")]
    {
        let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;
        println!("Converted to GPU (both CPU and GPU memory used)");
        
        // Process on GPU
        let gpu_result = (&gpu_tensor * 2.0) + 1.0;
        println!("Processed on GPU");
        
        // Convert back to CPU (allocates new CPU memory)
        let final_result = gpu_result.to_backend(BackendType::Cpu)?;
        println!("Converted back to CPU");
        
        // At this point: original CPU tensor, GPU tensor, and final CPU tensor exist
        // Memory is automatically freed when variables go out of scope
        
        let summary = final_result.sum(None)?;
        println!("Final summary: {}", summary.to_vec()?[0]);
    }
    
    println!("Memory automatically freed when variables go out of scope");
    
    Ok(())
}
```

## Production Patterns

### Configuration-Driven Backend Selection

```rust
use std::env;

#[derive(Debug)]
struct TensorConfig {
    preferred_backend: BackendType,
    fallback_backends: Vec<BackendType>,
    small_tensor_threshold: usize,
}

impl TensorConfig {
    fn from_env() -> Self {
        let preferred = env::var("TENSOR_BACKEND")
            .unwrap_or_else(|_| "auto".to_string());
        
        let preferred_backend = match preferred.as_str() {
            "cpu" => BackendType::Cpu,
            #[cfg(feature = "wgpu")]
            "wgpu" => BackendType::Wgpu,
            #[cfg(feature = "cuda")]
            "cuda" => BackendType::Cuda,
            _ => {
                // Auto-select best available
                #[cfg(feature = "cuda")]
                { BackendType::Cuda }
                #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
                { BackendType::Wgpu }
                #[cfg(all(not(feature = "wgpu"), not(feature = "cuda")))]
                { BackendType::Cpu }
            }
        };
        
        let threshold = env::var("SMALL_TENSOR_THRESHOLD")
            .unwrap_or_else(|_| "10000".to_string())
            .parse()
            .unwrap_or(10000);
        
        TensorConfig {
            preferred_backend,
            fallback_backends: vec![BackendType::Cpu],  // Always fallback to CPU
            small_tensor_threshold: threshold,
        }
    }
    
    fn select_backend(&self, tensor_size: usize) -> BackendType {
        if tensor_size < self.small_tensor_threshold {
            BackendType::Cpu  // Always use CPU for small tensors
        } else {
            self.preferred_backend
        }
    }
}

fn production_backend_usage() -> Result<()> {
    println!("=== Production Backend Usage ===\n");
    
    let config = TensorConfig::from_env();
    println!("Configuration: {:?}", config);
    
    // Use configuration for tensor operations
    let sizes = vec![100, 1000, 10000, 100000];
    
    for size in sizes {
        let tensor = Tensor::ones(vec![size])?;
        let elements = tensor.numel();
        
        let backend = config.select_backend(elements);
        let optimized_tensor = tensor.to_backend(backend)?;
        
        println!("Tensor size {}: using {:?} backend", 
                elements, optimized_tensor.backend_type());
    }
    
    Ok(())
}
```

### Application-Level Backend Strategy

```rust
struct TensorApplication {
    config: TensorConfig,
}

impl TensorApplication {
    fn new() -> Self {
        Self {
            config: TensorConfig::from_env(),
        }
    }
    
    fn process_data(&self, data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor> {
        // Create tensor
        let tensor = Tensor::from_vec(data, shape)?;
        
        // Select optimal backend
        let backend = self.config.select_backend(tensor.numel());
        let optimized_tensor = tensor.to_backend(backend)?;
        
        // Perform operations
        let processed = (&optimized_tensor * 2.0) + 1.0;
        let normalized = &processed / processed.sum(None)?;
        
        Ok(normalized)
    }
    
    fn batch_process(&self, batches: Vec<Vec<f32>>, shape: Vec<usize>) -> Result<Vec<Tensor>> {
        batches
            .into_iter()
            .map(|batch| self.process_data(batch, shape.clone()))
            .collect()
    }
}
```

## Best Practices Summary

### 1. Size-Based Selection
- **Small tensors (< 10K elements)**: Use CPU backend
- **Medium tensors (10K - 1M elements)**: Consider WGPU
- **Large tensors (> 1M elements)**: Prefer CUDA > WGPU > CPU

### 2. Operation-Based Selection
- **I/O operations**: Use CPU backend
- **Element-wise operations**: Use GPU backends for large tensors
- **Reductions**: GPU effective for very large tensors
- **Large reductions**: CUDA > CPU > WGPU (until WGPU reductions implemented)

### 3. Memory Management
- Convert to target backend early in pipeline
- Avoid frequent backend conversions
- Use batch processing when possible
- Monitor memory usage in production

### 4. Error Handling
- Always provide CPU fallback
- Handle backend-specific errors gracefully
- Use configuration for backend preferences
- Test with all available backends

### 5. Performance Optimization
- Benchmark with your specific workload
- Consider warmup time for GPU backends
- Profile memory transfer overhead
- Use appropriate tensor sizes for each backend

## Next Steps

- [Performance Guide](../performance.md) - Advanced optimization techniques
- [API Reference](../api/backends.md) - Detailed backend API documentation
- [Backend-Specific Guides](../backends/) - Deep dives into each backend