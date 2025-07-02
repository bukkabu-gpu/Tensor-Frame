# Getting Started

## Installation

Add Tensor Frame to your `Cargo.toml`:

```toml
[dependencies]
tensor_frame = "0.0.1-alpha"
```

### Feature Flags

Tensor Frame supports optional backends via feature flags:

```toml
[dependencies]
# CPU only (default)
tensor_frame = "0.0.1-alpha"

# With WGPU support
tensor_frame = { version = "0.0.1-alpha", features = ["wgpu"] }

# With CUDA support  
tensor_frame = { version = "0.0.1-alpha", features = ["cuda"] }

# All backends
tensor_frame = { version = "0.0.1-alpha", features = ["wgpu", "cuda"] }
```

## Basic Usage

### Creating Tensors

```rust
use tensor_frame::{Tensor, Result};

fn main() -> Result<()> {
    // Create tensors with different initialization
    let zeros = Tensor::zeros(vec![2, 3])?;
    let ones = Tensor::ones(vec![2, 3])?;
    let from_data = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0], 
        vec![2, 2]
    )?;
    
    // Inspect tensor properties
    println!("Shape: {:?}", zeros.shape().dims());
    println!("Number of elements: {}", zeros.numel());
    println!("Number of dimensions: {}", zeros.ndim());
    
    Ok(())
}
```

### Basic Operations

```rust
use tensor_frame::{Tensor, Result};

fn main() -> Result<()> {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    
    // Element-wise operations
    let sum = (a.clone() + b.clone())?;
    let diff = (a.clone() - b.clone())?;
    let product = (a.clone() * b.clone())?;
    let quotient = (a / b)?;
    
    // Reduction operations
    let total = sum.sum(None)?;
    let average = product.mean(None)?;
    
    println!("Sum result: {:?}", total.to_vec()?);
    
    Ok(())
}
```

### Broadcasting

Tensor Frame supports automatic broadcasting similar to NumPy and PyTorch:

```rust
use tensor_frame::{Tensor, Result};

fn main() -> Result<()> {
    let a = Tensor::ones(vec![2, 1])?;  // Shape: [2, 1]
    let b = Tensor::ones(vec![1, 3])?;  // Shape: [1, 3]
    
    // Broadcasting: [2, 1] + [1, 3] -> [2, 3]
    let c = (a + b)?;
    println!("Result shape: {:?}", c.shape().dims());
    
    Ok(())
}
```

### Tensor Manipulation

```rust
use tensor_frame::{Tensor, Result, TensorOps};

fn main() -> Result<()> {
    let tensor = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;
    
    // Reshape
    let reshaped = tensor.reshape(vec![3, 2])?;
    
    // Transpose (2D only for now)
    let transposed = reshaped.transpose()?;
    
    // Squeeze and unsqueeze
    let squeezed = tensor.squeeze(None)?;
    let unsqueezed = squeezed.unsqueeze(0)?;
    
    Ok(())
}
```