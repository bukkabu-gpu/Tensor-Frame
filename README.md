# Tensor Frame

[![Crates.io](https://img.shields.io/crates/v/tensor-frame)](https://crates.io/crates/tensor-frame)
[![Documentation](https://docs.rs/tensor-frame/badge.svg)](https://docs.rs/tensor-frame)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A high-performance, PyTorch-like tensor library for Rust with support for multiple computational backends.

## Features

- 🚀 **Multiple Backends**: CPU (Rayon), WGPU, and CUDA support
- 🔄 **Automatic Backend Selection**: Falls back to best available backend
- 📐 **Broadcasting**: NumPy/PyTorch-style automatic broadcasting
- 🎯 **Type Safety**: Rust's type system for memory safety
- ⚡ **Zero-Copy Operations**: Efficient memory management
- 🎛️ **Feature Flags**: Optional dependencies for different backends

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tensor-frame = "0.1"

# For GPU support
tensor-frame = { version = "0.1", features = ["wgpu"] }
```

Basic usage:

```rust
use tensor_frame::Tensor;

// Create tensors (automatically uses best backend)
let a = Tensor::ones(vec![2, 3])?;
let b = Tensor::zeros(vec![2, 3])?;

// Operations with broadcasting
let c = (a + b)?;
let sum = c.sum(None)?;

println!("Result: {:?}", sum.to_vec()?);
```

## Backends

### CPU Backend (Default)
- Uses Rayon for parallel computation
- Always available
- Good for small to medium tensors

### WGPU Backend
- Cross-platform GPU compute
- Supports Metal, Vulkan, DX12, OpenGL
- Enable with `features = ["wgpu"]`

### CUDA Backend  
- NVIDIA GPU acceleration
- Enable with `features = ["cuda"]`
- Requires CUDA toolkit

## Documentation

- 📖 [**Complete Guide**](https://yourusername.github.io/tensor-frame/) - Comprehensive documentation with tutorials
- 🚀 [**Getting Started**](https://yourusername.github.io/tensor-frame/getting-started.html) - Quick start guide  
- 📚 [**API Reference**](https://docs.rs/tensor-frame) - Detailed API documentation
- 💡 [**Examples**](https://yourusername.github.io/tensor-frame/examples/) - Practical examples and tutorials
- ⚡ [**Performance Guide**](https://yourusername.github.io/tensor-frame/performance.html) - Optimization tips and benchmarks
- 🔧 [**Backend Guides**](https://yourusername.github.io/tensor-frame/backends/) - CPU, WGPU, and CUDA backend details

## Examples

See the [examples](examples/) directory for more detailed usage:

- [Basic Operations](examples/basic_operations.rs)
- [Broadcasting](examples/broadcasting.rs)
- [Backend Selection](examples/backends.rs)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.