# Contributing to Tensor Frame

We welcome contributions to Tensor Frame! This guide will help you get started with contributing to the project.

## Getting Started

### Development Setup

1. **Clone the repository**:
```bash
git clone https://github.com/TrainPioneers/Tensor-Frame.git
cd Tensor-Frame
```

2. **Install Rust** (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

3. **Install development dependencies**:
```bash
# For documentation building
cargo install mdbook

# For benchmarking
cargo install criterion

# For code formatting
rustup component add rustfmt

# For linting
rustup component add clippy
```

4. **Build and test**:
```bash
# Build with all features
cargo build --all-features

# Run tests
cargo test

# Run with specific backend
cargo test --features wgpu
cargo test --features cuda
```

## Development Workflow

### Building the Project

```bash
# Quick compilation check
cargo check

# Build with specific backends
cargo build --features wgpu
cargo build --features cuda
cargo build --all-features

# Release build
cargo build --release --all-features
```

### Running Tests

```bash
# Run all tests
cargo test

# Test specific backend
make test-wgpu
make test-cuda

# Test with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_tensor_creation
```

### Code Formatting and Linting

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check

# Run clippy lints
cargo clippy

# Run clippy with all features
cargo clippy --all-features

# Fix clippy warnings
cargo clippy --fix
```

### Documentation

```bash
# Generate API documentation
cargo doc --open

# Build the book
cd docs
mdbook build

# Serve book locally
mdbook serve
```

## Contribution Guidelines

### Code Style

- **Formatting**: Use `cargo fmt` for consistent formatting
- **Linting**: Address all `cargo clippy` warnings
- **Naming**: Use descriptive names following Rust conventions
- **Comments**: Document public APIs and complex algorithms
- **Error Handling**: Use proper `Result` types and meaningful error messages

### Testing

All contributions must include appropriate tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_feature() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        let result = tensor.new_operation().unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_error_handling() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        let result = tensor.invalid_operation();
        assert!(result.is_err());
    }
}
```

### Documentation Requirements

- **Public APIs**: All public functions, structs, and traits must have documentation
- **Examples**: Include usage examples in documentation
- **Error Cases**: Document when functions return errors
- **Safety**: Document any unsafe code usage

```rust
/// Creates a new tensor filled with zeros.
///
/// # Arguments
/// * `shape` - The dimensions of the tensor
///
/// # Returns
/// A new tensor filled with zeros, or an error if the shape is invalid.
///
/// # Examples
/// ```
/// use tensor_frame::Tensor;
/// 
/// let tensor = Tensor::zeros(vec![2, 3])?;
/// assert_eq!(tensor.numel(), 6);
/// # Ok::<(), tensor_frame::TensorError>(())
/// ```
///
/// # Errors
/// Returns `TensorError::InvalidShape` if any dimension is zero.
pub fn zeros(shape: Vec<usize>) -> Result<Self> {
    // Implementation
}
```

## Types of Contributions

### Bug Fixes

1. **Report the issue**: Create a GitHub issue with:
   - Clear reproduction steps
   - Expected vs actual behavior
   - Environment details (OS, Rust version, GPU info)
   - Minimal code example

2. **Fix the bug**:
   - Create a focused fix addressing the specific issue
   - Add regression tests to prevent recurrence
   - Update documentation if the bug was in documented behavior

### New Features

Before implementing new features:

1. **Discuss the feature**: Open a GitHub issue to discuss:
   - Use case and motivation
   - Proposed API design
   - Implementation approach
   - Performance implications

2. **Implementation guidelines**:
   - Follow existing patterns and conventions
   - Implement for all relevant backends
   - Add comprehensive tests
   - Update documentation and examples

#### Backend Implementation

New operations should be implemented across all backends:

```rust
// src/backend/mod.rs
pub trait Backend {
    // Add new operation to trait
    fn new_operation(&self, input: &Storage) -> Result<Storage>;
}

// src/backend/cpu.rs
impl Backend for CpuBackend {
    fn new_operation(&self, input: &Storage) -> Result<Storage> {
        match input {
            Storage::Cpu(data) => {
                // CPU implementation using Rayon
                let result: Vec<f32> = data
                    .par_iter()
                    .map(|&x| compute_new_operation(x))
                    .collect();
                Ok(Storage::Cpu(result))
            }
            _ => Err(TensorError::BackendError("Invalid storage type".to_string())),
        }
    }
}

// src/backend/wgpu.rs
impl Backend for WgpuBackend {
    fn new_operation(&self, input: &Storage) -> Result<Storage> {
        match input {
            Storage::Wgpu(wgpu_storage) => {
                // WGPU implementation using compute shaders
                self.execute_compute_shader(
                    &wgpu_storage.buffer,
                    include_str!("../shaders/new_operation.wgsl")
                )
            }
            _ => Err(TensorError::BackendError("Invalid storage type".to_string())),
        }
    }
}
```

### Performance Improvements

1. **Benchmark first**: Establish baseline performance
2. **Profile the bottleneck**: Use profiling tools to identify issues
3. **Implement optimization**: Make targeted improvements
4. **Measure improvement**: Verify performance gains
5. **Add performance tests**: Prevent performance regressions

```rust
// Add benchmark for new optimization
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_optimized_operation(c: &mut Criterion) {
    let tensor = Tensor::ones(vec![1000, 1000]).unwrap();
    
    c.bench_function("optimized_operation", |b| {
        b.iter(|| {
            tensor.optimized_operation().unwrap()
        });
    });
}

criterion_group!(benches, bench_optimized_operation);
criterion_main!(benches);
```

### Documentation Improvements

- **API documentation**: Improve function/struct documentation
- **Examples**: Add or improve usage examples
- **Guides**: Write tutorials for specific use cases
- **Book**: Contribute to the mdbook documentation

## Backend-Specific Contributions

### CPU Backend

- **Optimization**: Improve Rayon parallelization
- **BLAS integration**: Better integration with optimized BLAS libraries
- **Memory layout**: Optimize for cache efficiency

### WGPU Backend

- **Shader optimization**: Improve WGSL compute shaders
- **New operations**: Implement missing operations (matmul, reductions)
- **Platform support**: Improve compatibility across graphics APIs

### CUDA Backend

- **Kernel optimization**: Improve CUDA kernel performance
- **cuBLAS integration**: Better integration with cuBLAS/cuDNN
- **Memory management**: Optimize GPU memory usage

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
```bash
cargo test --all-features
```

2. **Check formatting and lints**:
```bash
cargo fmt --check
cargo clippy --all-features
```

3. **Update documentation**:
```bash
cargo doc --all-features
cd docs && mdbook build
```

4. **Add changelog entry** (if applicable):
```markdown
## [Unreleased]
### Added
- New tensor operation `my_operation` (#123)
### Fixed  
- Fixed broadcasting bug in GPU backend (#124)
```

### Pull Request Template

```markdown
## Description
Brief description of the changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested with different backends (CPU/WGPU/CUDA)

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. **Automated checks**: CI will run tests, linting, and formatting checks
2. **Code review**: Maintainers will review for:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Performance implications
   - API design consistency
3. **Feedback**: Address review feedback and update the PR
4. **Approval**: Once approved, maintainers will merge the PR

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Create tensor with '...'
2. Call operation '....'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Code Example**
```rust
use tensor_frame::Tensor;

let tensor = Tensor::zeros(vec![2, 3])?;
let result = tensor.problematic_operation()?; // This fails
```

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Rust version: [e.g. 1.75.0]
- Tensor Frame version: [e.g. 0.1.0]
- GPU info: [if applicable]
- Backend: [CPU/WGPU/CUDA]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Use case**
Describe how this feature would be used in practice.

**API Design** (if applicable)
```rust
// Proposed API
let result = tensor.new_operation(parameters)?;
```

**Additional context**
Add any other context about the feature request here.
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Celebrate diverse perspectives and backgrounds

### Communication

- **GitHub Issues**: Bug reports, feature requests, design discussions
- **GitHub Discussions**: General questions, show and tell, ideas
- **Pull Requests**: Code contributions and reviews

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributor statistics

## Getting Help

If you need help contributing:

1. **Read existing code**: Look at similar implementations for patterns
2. **Check documentation**: API docs and this book contain guidance
3. **Ask questions**: Open a GitHub issue or discussion
4. **Start small**: Begin with bug fixes or documentation improvements

Thank you for contributing to Tensor Frame!