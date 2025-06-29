# Tensor Frame

Tensor Frame is a high-performance, PyTorch-like tensor library for Rust that supports multiple computational backends including CPU (with Rayon), WGPU (for GPU compute), and CUDA.

## Features

- **Multiple Backends**: Automatic backend selection with fallback support
  - CPU backend with Rayon for parallel processing
  - WGPU backend for cross-platform GPU computing
  - CUDA backend for NVIDIA GPU acceleration
- **PyTorch-like API**: Familiar tensor operations and broadcasting
- **Dynamic Tensors**: Runtime shape and type flexibility
- **Broadcasting Support**: Automatic shape broadcasting for operations
- **Zero-Copy Operations**: Efficient memory management where possible
- **Feature Flags**: Optional backends via Cargo features

## Quick Example

```rust
use tensor_frame::Tensor;

// Create tensors (automatically uses the best available backend)
let a = Tensor::ones(vec![2, 3])?;
let b = Tensor::zeros(vec![2, 3])?;

// Perform operations with automatic broadcasting
let c = (a + b)?;
let d = c.sum(None)?; // Sum all elements

// Convert back to Vec for inspection
let result = d.to_vec()?;
println!("Result: {:?}", result);
```

## Backend Priority

By default, Tensor Frame will attempt to use backends in this order:
1. CUDA (if available and feature enabled)
2. WGPU (if available and feature enabled)  
3. CPU (always available)

You can also explicitly specify a backend or create custom backend implementations.