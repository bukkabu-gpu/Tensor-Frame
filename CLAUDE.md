# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tensor Frame is a PyTorch-like tensor library for Rust with multiple computational backends. The architecture uses a trait-based backend system with automatic fallback selection (CUDA → WGPU → CPU).

## Commands

### Building and Testing
- `make test` - Run basic tests with CPU backend
- `make test-wgpu` - Test with WGPU backend enabled (currently disabled)
- `make test-cuda` - Test with CUDA backend (requires CUDA toolkit)
- `cargo run --example basic_operations` - Run the main example
- `make publish` - Test crates.io publication readiness

### Development
- `cargo check` - Quick compilation check
- `cargo test` - Run unit tests
- `cargo doc --open` - Generate and open documentation

## Architecture

### Backend System
The core architecture revolves around the `Backend` trait in `src/backend/mod.rs`:

1. **Backend Selection**: Automatic priority-based fallback using `BACKEND` static in `backend/mod.rs`
   - Priority order: CUDA → WGPU → CPU
   - Each backend checks `is_available()` before selection

2. **Storage Abstraction**: `Storage` enum handles different memory types:
   - `Storage::Cpu(Vec<f32>)` - CPU memory with Rayon parallelization
   - `Storage::Cuda(CudaStorage)` - GPU memory via cudarc
   - `Storage::Wgpu(WgpuStorage)` - GPU compute via WGPU (currently disabled)

3. **Tensor Operations**: Implemented via `std::ops` traits (Add, Sub, Mul, Div) in `tensor/mod.rs`
   - Operations return `Result<Tensor>` for error handling
   - Broadcasting handled in `tensor/broadcast.rs`

### Key Components
- **Shape System**: `tensor/shape.rs` handles n-dimensional shapes and broadcasting rules
- **Error Handling**: Comprehensive error types in `error.rs` for tensor operations
- **Display Formatting**: User-friendly tensor printing with proper matrix formatting

### Feature Flags
- `cpu` (default): Rayon-based CPU parallelization
- `wgpu`: Cross-platform GPU compute (currently disabled due to API changes)
- `cuda`: NVIDIA GPU acceleration via cudarc

## Known Issues
- WGPU backend is disabled due to `wgpu::Buffer` API changes in v23
- CUDA backend requires specific version features (`cuda-version-from-build-system`)
- Some unreachable pattern warnings in CPU backend when other backends are disabled

## Testing Strategy
Tests are in `src/lib.rs` and cover:
- Basic tensor creation and arithmetic
- Broadcasting behavior
- Backend selection and tensor conversion
- Error handling for invalid operations

The example in `examples/basic_operations.rs` demonstrates the full API including the Display trait implementation.