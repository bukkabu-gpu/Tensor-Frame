# Basic Operations

This example demonstrates the fundamental tensor operations in Tensor Frame. It covers tensor creation, basic arithmetic, shape manipulation, and data access patterns.

## Complete Example

```rust
use tensor_frame::{Tensor, Result, TensorOps};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Tensor Frame Basic Operations ===\n");

    // 1. Tensor Creation
    tensor_creation_examples()?;
    
    // 2. Basic Arithmetic
    arithmetic_examples()?;
    
    // 3. Shape Manipulation  
    shape_manipulation_examples()?;
    
    // 4. Data Access
    data_access_examples()?;
    
    // 5. Performance Comparison
    performance_comparison()?;

    Ok(())
}

/// Demonstrates various ways to create tensors
fn tensor_creation_examples() -> Result<()> {
    println!("=== Tensor Creation ===");
    
    // Create tensor filled with zeros
    let zeros = Tensor::zeros(vec![2, 3])?;
    println!("Zeros tensor (2x3):\n{}\n", zeros);
    
    // Create tensor filled with ones
    let ones = Tensor::ones(vec![3, 2])?;
    println!("Ones tensor (3x2):\n{}\n", ones);
    
    // Create tensor from existing data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let from_data = Tensor::from_vec(data, vec![2, 3])?;
    println!("From data (2x3):\n{}\n", from_data);
    
    // Check tensor properties
    println!("Tensor properties:");
    println!("  Shape: {:?}", from_data.shape().dims());
    println!("  Number of elements: {}", from_data.numel());
    println!("  Data type: {:?}", from_data.dtype());
    println!("  Backend: {:?}\n", from_data.backend_type());
    
    Ok(())
}

/// Demonstrates basic arithmetic operations
fn arithmetic_examples() -> Result<()> {
    println!("=== Arithmetic Operations ===");
    
    // Create test tensors
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    
    println!("Tensor A:\n{}\n", a);
    println!("Tensor B:\n{}\n", b);
    
    // Element-wise addition
    let sum = &a + &b;  // Use references to avoid moving tensors
    println!("A + B:\n{}\n", sum);
    
    // Element-wise subtraction
    let diff = &a - &b;
    println!("A - B:\n{}\n", diff);
    
    // Element-wise multiplication
    let product = &a * &b;
    println!("A * B (element-wise):\n{}\n", product);
    
    // Element-wise division
    let quotient = &a / &b;
    println!("A / B:\n{}\n", quotient);
    
    // Chained operations
    let complex = ((&a * 2.0) + &b) / 3.0;
    println!("(A * 2 + B) / 3:\n{}\n", complex);
    
    Ok(())
}

/// Demonstrates shape manipulation operations
fn shape_manipulation_examples() -> Result<()> {
    println!("=== Shape Manipulation ===");
    
    // Create a tensor to manipulate
    let tensor = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
        vec![2, 4]
    )?;
    println!("Original tensor (2x4):\n{}\n", tensor);
    
    // Reshape to different dimensions
    let reshaped = tensor.reshape(vec![4, 2])?;
    println!("Reshaped to (4x2):\n{}\n", reshaped);
    
    // Reshape to 1D
    let flattened = tensor.reshape(vec![8])?;
    println!("Flattened to (8,):\n{}\n", flattened);
    
    // Transpose (2D only)
    let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let transposed = matrix.transpose()?;
    println!("Original matrix:\n{}\n", matrix);
    println!("Transposed matrix:\n{}\n", transposed);
    
    // Squeeze and unsqueeze
    let with_ones = Tensor::ones(vec![1, 3, 1])?;
    println!("Tensor with size-1 dimensions (1x3x1):\n{}\n", with_ones);
    
    let squeezed = with_ones.squeeze(None)?;
    println!("Squeezed (removes all size-1 dims):\n{}\n", squeezed);
    
    let unsqueezed = squeezed.unsqueeze(0)?;
    println!("Unsqueezed at dimension 0:\n{}\n", unsqueezed);
    
    Ok(())
}

/// Demonstrates data access patterns
fn data_access_examples() -> Result<()> {
    println!("=== Data Access ===");
    
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    println!("Tensor:\n{}\n", tensor);
    
    // Convert to Vec for external use
    let data = tensor.to_vec()?;
    println!("As Vec<f32>: {:?}\n", data);
    
    // Reduction operations
    let sum_all = tensor.sum(None)?;
    println!("Sum of all elements: {}\n", sum_all);
    
    let mean_all = tensor.mean(None)?;
    println!("Mean of all elements: {}\n", mean_all);
    
    // Axis-specific reductions
    let row_sums = tensor.sum(Some(1))?;  // Sum along columns (axis 1)
    println!("Row sums (sum along axis 1): {}\n", row_sums);
    
    let col_sums = tensor.sum(Some(0))?;  // Sum along rows (axis 0)
    println!("Column sums (sum along axis 0): {}\n", col_sums);
    
    Ok(())
}

/// Demonstrates performance characteristics
fn performance_comparison() -> Result<()> {
    println!("=== Performance Comparison ===");
    
    // Small tensor operations (CPU should be faster)
    let small_a = Tensor::ones(vec![100, 100])?;
    let small_b = Tensor::ones(vec![100, 100])?;
    
    let start = Instant::now();
    let result = &small_a + &small_b;
    let small_time = start.elapsed();
    println!("Small tensor (100x100) addition: {:?}", small_time);
    
    // Large tensor operations (GPU might be faster if available)
    let large_a = Tensor::ones(vec![1000, 1000])?;
    let large_b = Tensor::ones(vec![1000, 1000])?;
    
    let start = Instant::now();
    let result = &large_a + &large_b;
    let large_time = start.elapsed();
    println!("Large tensor (1000x1000) addition: {:?}", large_time);
    
    // Show current backend
    println!("Current backend: {:?}", result.backend_type());
    
    // Demonstrate backend conversion (if other backends available)
    #[cfg(feature = "wgpu")]
    {
        println!("\n--- WGPU Backend Comparison ---");
        let start = Instant::now();
        let wgpu_a = large_a.to_backend(tensor_frame::BackendType::Wgpu)?;
        let wgpu_b = large_b.to_backend(tensor_frame::BackendType::Wgpu)?;
        let conversion_time = start.elapsed();
        
        let start = Instant::now();
        let wgpu_result = &wgpu_a + &wgpu_b;
        let _sync = wgpu_result.to_vec()?;  // Force synchronization
        let wgpu_time = start.elapsed();
        
        println!("WGPU conversion time: {:?}", conversion_time);
        println!("WGPU computation time: {:?}", wgpu_time);
        println!("Total WGPU time: {:?}", conversion_time + wgpu_time);
    }
    
    Ok(())
}

/// Advanced patterns demonstration
fn advanced_patterns() -> Result<()> {
    println!("=== Advanced Patterns ===");
    
    // Broadcasting example
    let matrix = Tensor::ones(vec![3, 4])?;     // Shape: [3, 4]
    let vector = Tensor::ones(vec![4])?;        // Shape: [4]
    let broadcasted = &matrix + &vector;        // Result: [3, 4]
    
    println!("Matrix (3x4):\n{}\n", matrix);
    println!("Vector (4,):\n{}\n", vector);
    println!("Matrix + Vector (broadcasted):\n{}\n", broadcasted);
    
    // Complex broadcasting
    let a = Tensor::ones(vec![2, 1, 3])?;       // Shape: [2, 1, 3]
    let b = Tensor::ones(vec![1, 4, 1])?;       // Shape: [1, 4, 1]
    let complex_broadcast = &a + &b;            // Result: [2, 4, 3]
    
    println!("Complex broadcasting:");
    println!("A shape: {:?}", a.shape().dims());
    println!("B shape: {:?}", b.shape().dims());
    println!("Result shape: {:?}", complex_broadcast.shape().dims());
    
    // Method chaining
    let result = Tensor::ones(vec![2, 3])?
        .reshape(vec![3, 2])?
        .transpose()?;
    
    println!("Method chaining result:\n{}\n", result);
    
    Ok(())
}

/// Error handling examples
fn error_handling_examples() -> Result<()> {
    println!("=== Error Handling ===");
    
    // Shape mismatch error
    let a = Tensor::ones(vec![2, 3])?;
    let b = Tensor::ones(vec![3, 2])?;
    
    match &a + &b {
        Ok(result) => println!("Addition succeeded: {}", result),
        Err(e) => println!("Expected error - shape mismatch: {}", e),
    }
    
    // Invalid reshape error
    let tensor = Tensor::ones(vec![2, 3])?;  // 6 elements
    match tensor.reshape(vec![2, 2]) {       // 4 elements - invalid!
        Ok(result) => println!("Reshape succeeded: {}", result),
        Err(e) => println!("Expected error - invalid reshape: {}", e),
    }
    
    // Out of bounds dimension error
    match tensor.squeeze(Some(5)) {  // Dimension 5 doesn't exist
        Ok(result) => println!("Squeeze succeeded: {}", result),
        Err(e) => println!("Expected error - invalid dimension: {}", e),
    }
    
    Ok(())
}
```

## Key Concepts Demonstrated

### 1. Tensor Creation
Three primary ways to create tensors:
- `Tensor::zeros(shape)` - Creates tensor filled with zeros
- `Tensor::ones(shape)` - Creates tensor filled with ones  
- `Tensor::from_vec(data, shape)` - Creates tensor from existing data

### 2. Reference vs. Owned Operations
```rust
// Moves tensors (can only use once)
let result = a + b;

// Uses references (can reuse tensors)
let result = &a + &b;
```

### 3. Shape Broadcasting
Tensor Frame automatically broadcasts compatible shapes:
```rust
let matrix = Tensor::ones(vec![3, 4])?;  // [3, 4]
let vector = Tensor::ones(vec![4])?;     // [4] broadcasts to [1, 4]
let result = matrix + vector;            // Result: [3, 4]
```

### 4. Method Chaining
Operations can be chained for concise code:
```rust
let result = tensor
    .reshape(vec![4, 2])?
    .transpose()?
    .squeeze(None)?;
```

### 5. Error Handling
All operations return `Result<T>` for proper error handling:
```rust
match risky_operation() {
    Ok(tensor) => process_tensor(tensor),
    Err(TensorError::ShapeMismatch { expected, got }) => {
        eprintln!("Shape error: expected {:?}, got {:?}", expected, got);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Performance Tips

1. **Use References**: Use `&a + &b` instead of `a + b` to avoid unnecessary clones
2. **Batch Operations**: Combine operations when possible: `(a * 2.0) + b` vs separate operations
3. **Choose Right Backend**: CPU for small tensors, GPU for large operations
4. **Avoid Frequent Conversions**: Stay on one backend when possible

## Common Pitfalls

1. **Shape Mismatches**: Ensure compatible shapes for operations
2. **Invalid Reshapes**: New shape must have same total elements
3. **Backend Overhead**: GPU operations have overhead for small tensors
4. **Memory Usage**: Large tensors consume significant memory

## Next Steps

After mastering basic operations, explore:
- [Broadcasting Examples](./broadcasting.md) - Advanced broadcasting patterns
- [Backend Selection](./custom-backends.md) - Optimizing backend usage
- [Performance Guide](../performance.md) - Advanced performance optimization