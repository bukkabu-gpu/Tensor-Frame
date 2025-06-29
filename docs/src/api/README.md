# API Reference

This section provides detailed documentation for all public APIs in Tensor Frame.

## Core Types

- **[Tensor](./tensor.md)** - The main tensor type with all operations
- **[Backends](./backends.md)** - Backend trait and implementation details  
- **[Operations](./operations.md)** - Detailed operation specifications

## Key Traits and Enums

### TensorOps Trait

The `TensorOps` trait defines all tensor manipulation and computation operations:

```rust
pub trait TensorOps {
    fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor>;
    fn transpose(&self) -> Result<Tensor>;
    fn squeeze(&self, dim: Option<usize>) -> Result<Tensor>;
    fn unsqueeze(&self, dim: usize) -> Result<Tensor>;
    // ... more methods
}
```

### DType Enum

Supported data types:

```rust
pub enum DType {
    F32,    // 32-bit floating point (default)
    F64,    // 64-bit floating point  
    I32,    // 32-bit signed integer
    U32,    // 32-bit unsigned integer
}
```

### BackendType Enum

Available computational backends:

```rust
pub enum BackendType {
    Cpu,    // CPU backend with Rayon
    Wgpu,   // Cross-platform GPU backend
    Cuda,   // NVIDIA CUDA backend
}
```

## Error Handling

All operations return `Result<T>` with `TensorError` for comprehensive error handling:

```rust
pub enum TensorError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    BackendError(String),
    InvalidOperation(String),
    DimensionError(String),
}
```

## Memory Management

Tensor Frame uses smart pointers and reference counting for efficient memory management:

- Tensors are cheaply clonable (reference counted)
- Backend storage is automatically managed
- Cross-backend tensor conversion is supported
- Zero-copy operations where possible