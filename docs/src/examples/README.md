# Examples and Tutorials

This section provides practical examples and tutorials for using Tensor Frame effectively. Each example is designed to demonstrate specific features and common usage patterns.

## Getting Started Examples

Perfect for newcomers to Tensor Frame:

- **[Basic Operations](./basic.md)** - Tensor creation, arithmetic, and basic manipulation
- **[Broadcasting](./broadcasting.md)** - Understanding automatic shape broadcasting
- **[Custom Backends](./custom-backends.md)** - Working with different computational backends

## Example Categories

### Fundamental Operations
Learn the core tensor operations that form the foundation of all computational work:

```rust
// Tensor creation
let zeros = Tensor::zeros(vec![3, 4])?;
let ones = Tensor::ones(vec![2, 2])?;
let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

// Basic arithmetic
let sum = a + b;
let product = a * b;
let result = (a * 2.0) + b;
```

### Shape Manipulation
Master tensor reshaping and dimension manipulation:

```rust
// Reshaping and transposition
let reshaped = tensor.reshape(vec![4, 3])?;
let transposed = matrix.transpose()?;

// Dimension manipulation
let squeezed = tensor.squeeze(None)?;
let unsqueezed = squeezed.unsqueeze(1)?;
```

### Backend Optimization
Learn when and how to use different computational backends:

```rust
// Automatic backend selection
let tensor = Tensor::zeros(vec![1000, 1000])?;

// Manual backend control
let gpu_tensor = tensor.to_backend(BackendType::Wgpu)?;
let cuda_tensor = tensor.to_backend(BackendType::Cuda)?;
```

## Running Examples

All examples are located in the `examples/` directory of the repository:

```bash
# Run basic operations example
cargo run --example basic_operations

# Run with specific backend
cargo run --example basic_operations --features wgpu
cargo run --example basic_operations --features cuda

# Run with all features
cargo run --example basic_operations --features "wgpu,cuda"
```

## Example Structure

Each example follows a consistent structure:

1. **Setup**: Import necessary modules and create test data
2. **Demonstration**: Show the specific feature in action
3. **Explanation**: Detailed comments explaining what's happening
4. **Performance Notes**: Tips for optimal usage
5. **Error Handling**: Proper error handling patterns

## Performance Benchmarking

Many examples include performance comparisons:

```rust
use std::time::Instant;

// CPU benchmark
let start = Instant::now();
let cpu_result = &cpu_tensor + &cpu_other;
let cpu_time = start.elapsed();

// GPU benchmark  
let start = Instant::now();
let gpu_result = &gpu_tensor + &gpu_other;
let _sync = gpu_result.to_vec()?;  // Force synchronization
let gpu_time = start.elapsed();

println!("CPU: {:?}, GPU: {:?}, Speedup: {:.1}x", 
         cpu_time, gpu_time, cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
```

## Interactive Examples

Some examples are designed for interactive exploration:

```bash
# Interactive tensor exploration
cargo run --example interactive

# Performance testing with different sizes
cargo run --example benchmark -- --size 1000
cargo run --example benchmark -- --size 2000 --backend cuda
```

## Common Patterns

### Error Handling Pattern
```rust
use tensor_frame::{Tensor, Result, TensorError};

fn robust_operation() -> Result<Tensor> {
    let tensor = Tensor::zeros(vec![1000, 1000])?;
    
    // Try GPU backend first
    match tensor.to_backend(BackendType::Wgpu) {
        Ok(gpu_tensor) => {
            // GPU operations here
            Ok(expensive_gpu_operation(gpu_tensor)?)
        }
        Err(TensorError::BackendError(_)) => {
            // Fallback to CPU
            println!("GPU not available, using CPU");
            Ok(cpu_operation(tensor)?)
        }
        Err(e) => Err(e),
    }
}
```

### Memory Management Pattern
```rust
fn memory_efficient_batch_processing(batches: Vec<Vec<f32>>) -> Result<Vec<Tensor>> {
    let backend = BackendType::Wgpu; // Choose once
    
    batches
        .into_iter()
        .map(|batch| {
            let tensor = Tensor::from_vec(batch, vec![batch.len()])?;
            tensor.to_backend(backend)  // Convert once per batch
        })
        .collect()
}
```

### Broadcasting Pattern
```rust
fn demonstrate_broadcasting() -> Result<()> {
    // Scalar broadcast
    let tensor = Tensor::ones(vec![3, 4])?;
    let scaled = tensor * 2.0;  // Scalar broadcasts to all elements
    
    // Vector broadcast
    let matrix = Tensor::ones(vec![3, 4])?;
    let vector = Tensor::ones(vec![4])?;      // Shape: [4]
    let result = matrix + vector;             // Broadcasts to [3, 4]
    
    // Matrix broadcast
    let a = Tensor::ones(vec![3, 1])?;        // Shape: [3, 1]
    let b = Tensor::ones(vec![1, 4])?;        // Shape: [1, 4]
    let result = a + b;                       // Result: [3, 4]
    
    Ok(())
}
```

## Advanced Examples

For users comfortable with the basics:

### Custom Backend Selection
```rust
fn adaptive_backend_selection(tensor_size: usize) -> BackendType {
    match tensor_size {
        0..=1000 => BackendType::Cpu,           // Small: CPU overhead minimal
        1001..=100000 => BackendType::Wgpu,     // Medium: GPU beneficial
        _ => BackendType::Cuda,                 // Large: Maximum performance
    }
}
```

### Batched Operations
```rust
fn process_batch_efficiently(inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
    // Convert all inputs to same backend
    let backend = BackendType::Wgpu;
    let gpu_inputs: Result<Vec<_>> = inputs
        .into_iter()
        .map(|t| t.to_backend(backend))
        .collect();
    
    // Process on GPU
    let gpu_outputs: Result<Vec<_>> = gpu_inputs?
        .into_iter()
        .map(|input| expensive_operation(input))
        .collect();
    
    gpu_outputs
}
```

## Troubleshooting Common Issues

### Performance Problems
```rust
// Problem: Slow operations on small tensors
let small = Tensor::ones(vec![10, 10])?;
let slow_result = small.to_backend(BackendType::Wgpu)?; // GPU overhead

// Solution: Use CPU for small tensors
let fast_result = small; // Stay on CPU backend
```

### Memory Issues
```rust
// Problem: GPU out of memory
match Tensor::zeros(vec![10000, 10000]) {
    Err(TensorError::BackendError(msg)) if msg.contains("memory") => {
        // Solution: Use smaller chunks or CPU backend
        let chunks = create_smaller_chunks()?;
        process_chunks_individually(chunks)?;
    }
    Ok(tensor) => process_large_tensor(tensor)?,
    Err(e) => return Err(e),
}
```

### Backend Compatibility
```rust
// Problem: Operation not supported on backend
let result = match tensor.backend_type() {
    BackendType::Wgpu => {
        // Some operations not yet implemented on WGPU
        tensor.to_backend(BackendType::Cpu)?.complex_operation()?
    }
    _ => tensor.complex_operation()?,
};
```

## Contributing Examples

We welcome contributions of new examples! Please follow these guidelines:

1. **Clear Purpose**: Each example should demonstrate a specific concept
2. **Complete Code**: Include all necessary imports and error handling
3. **Documentation**: Add detailed comments explaining the concepts
4. **Performance Notes**: Include timing and backend recommendations
5. **Error Handling**: Show proper error handling patterns

See the [Contributing Guide](../contributing.md) for more details on submitting examples.