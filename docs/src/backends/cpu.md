# CPU Backend

The CPU backend is the default and most mature backend in Tensor Frame. It provides reliable tensor operations using system memory and CPU cores, with parallelization via the Rayon library.

## Features

- **Always Available**: No additional dependencies required
- **Parallel Processing**: Multi-threaded operations via Rayon
- **Full API Support**: All tensor operations implemented
- **Memory Efficient**: Direct Vec<f32> storage without additional overhead
- **Debugging Friendly**: Easy inspection with standard debugging tools

## Configuration

The CPU backend is enabled by default:

```toml
[dependencies]
tensor_frame = "0.0.1-alpha"  # CPU backend included
```

Or explicitly:

```toml
[dependencies]
tensor_frame = { version = "0.0.1-alpha", features = ["cpu"] }
```

## Implementation Details

### Storage
CPU tensors use standard Rust `Vec<f32>` for data storage:

```rust
pub enum Storage {
    Cpu(Vec<f32>),    // Direct vector storage
    // ...
}
```

This provides:
- **Memory Layout**: Contiguous, row-major (C-style) layout
- **Access**: Direct memory access without marshaling overhead
- **Debugging**: Easy inspection with standard Rust tools

### Parallelization

The CPU backend uses [Rayon](https://github.com/rayon-rs/rayon) for data-parallel operations:

```rust
// Element-wise operations are parallelized
a.par_iter()
    .zip(b.par_iter())
    .map(|(a, b)| a + b)
    .collect()
```

**Thread Pool**: Rayon automatically manages a global thread pool sized to the number of CPU cores.

**Granularity**: Operations are automatically chunked for optimal parallel efficiency.

## Performance Characteristics

### Strengths
- **Low Latency**: Minimal overhead for small operations
- **Predictable**: Performance scales linearly with data size and core count
- **Memory Bandwidth**: Efficiently utilizes system memory bandwidth
- **Cache Friendly**: Good locality for sequential operations

### Limitations  
- **Compute Bound**: Limited by CPU ALU throughput
- **Memory Bound**: Large operations limited by RAM bandwidth
- **Thread Overhead**: Parallel overhead dominates for small tensors

### Performance Guidelines

#### Optimal Use Cases
```rust
// Small to medium tensors (< 10K elements)
let small = Tensor::ones(vec![100, 100])?;

// Scalar reductions
let sum = large_tensor.sum(None)?;

// Development and prototyping
let test_tensor = Tensor::from_vec(test_data, shape)?;
```

#### Suboptimal Use Cases
```rust
// Very large tensor operations
let huge_op = a + b;  // Consider GPU for very large tensors

// Repeated large element-wise operations
for _ in 0..1000 {
    result = (a.clone() * b.clone())?;  // GPU would be faster
}
```

## Memory Management

### Allocation
CPU tensors allocate memory directly from the system heap:

```rust
let tensor = Tensor::zeros(vec![1000, 1000])?;  // Allocates 4MB
```

### Reference Counting
Tensors use `Arc<Vec<f32>>` internally for efficient cloning:

```rust
let tensor1 = Tensor::ones(vec![1000])?;
let tensor2 = tensor1.clone();  // O(1) reference count increment
// Memory shared until one tensor is modified (copy-on-write semantics)
```

### Memory Usage
Monitor memory usage with standard system tools:

```bash
# Linux
cat /proc/meminfo

# macOS  
vm_stat

# Windows
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory
```

## Debugging and Profiling

### Tensor Inspection
CPU tensors are easy to inspect:

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

// Direct access to underlying data
let data = tensor.to_vec()?;
println!("Raw data: {:?}", data);

// Shape information
println!("Shape: {:?}", tensor.shape().dims());
println!("Elements: {}", tensor.numel());
```

### Performance Profiling
Use standard Rust profiling tools:

```rust
// Add timing
use std::time::Instant;

let start = Instant::now();
let result = large_tensor.sum(None)?;
println!("CPU operation took: {:?}", start.elapsed());
```

For detailed profiling:

```bash
# Install flamegraph
cargo install flamegraph

# Profile your application  
cargo flamegraph --bin your_app
```

### Thread Analysis
Monitor Rayon thread usage:

```rust
// Check thread pool size
println!("Rayon threads: {}", rayon::current_num_threads());

// Custom thread pool
let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()?;

pool.install(|| {
    // Operations here use 4 threads max
    let result = tensor1 + tensor2;
});
```

## Error Handling

CPU backend errors are typically related to memory allocation:

```rust
use tensor_frame::{Tensor, TensorError};

match Tensor::zeros(vec![100000, 100000]) {
    Ok(tensor) => {
        // Success - 40GB allocated
    }
    Err(TensorError::BackendError(msg)) => {
        // Likely out of memory
        eprintln!("CPU backend error: {}", msg);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

**Common Error Conditions**:
- **Out of Memory**: Requesting more memory than available
- **Integer Overflow**: Tensor dimensions too large for address space
- **Thread Panic**: Rayon worker thread panics (rare)

## Optimization Tips

### Memory Layout Optimization
```rust
// Prefer contiguous operations
let result = (a + b) * c;  // Better than separate operations

// Avoid unnecessary allocations
let result = a.clone() + b;  // Creates temporary clone
let result = &a + &b;       // Better - uses references
```

### Parallel Operation Tuning
```rust
// For very small tensors, disable parallelism
let small_result = small_a + small_b;  // Rayon decides automatically

// For custom control
rayon::ThreadPoolBuilder::new()
    .num_threads(1)  // Force single-threaded
    .build_global()?;
```

### Cache Optimization
```rust
// Process data in blocks for better cache usage
for chunk in tensor.chunks(cache_friendly_size) {
    // Process chunk
}

// Transpose cache-friendly
let transposed = matrix.transpose()?;  // May benefit from blocking
```

## Integration with Other Libraries

### NumPy Compatibility
```rust
// Convert to/from Vec for NumPy interop
let tensor = Tensor::from_vec(numpy_data, shape)?;
let back_to_numpy = tensor.to_vec()?;
```

### ndarray Integration
```rust
use ndarray::Array2;

// Convert from ndarray
let nd_array = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
let tensor = Tensor::from_vec(nd_array.into_raw_vec(), vec![2, 2])?;

// Convert to ndarray
let data = tensor.to_vec()?;
let shape = tensor.shape().dims();
let nd_array = Array2::from_shape_vec((shape[0], shape[1]), data)?;
```

### BLAS Integration
For maximum performance, consider linking with optimized BLAS:

```toml
[dependencies]
tensor_frame = "0.0.1-alpha"
blas-src = { version = "0.8", features = ["openblas"] }
```

This can significantly speed up matrix operations on the CPU backend.