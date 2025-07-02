# Tensor API

The `Tensor` struct is the core data structure in Tensor Frame, representing multi-dimensional arrays with automatic backend selection.

## Constructor Methods

### Basic Constructors

```rust
// Create tensor filled with zeros
pub fn zeros(shape: Vec<usize>) -> Result<Tensor>

// Create tensor filled with ones  
pub fn ones(shape: Vec<usize>) -> Result<Tensor>

// Create tensor from Vec data
pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor>
```

#### Examples

```rust
use tensor_frame::Tensor;

// 2x3 matrix of zeros
let zeros = Tensor::zeros(vec![2, 3])?;

// 1D vector of ones
let ones = Tensor::ones(vec![5])?;

// Create from existing data
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_vec(data, vec![2, 2])?;
```

## Properties

### Shape Information

```rust
// Get tensor shape
pub fn shape(&self) -> &Shape

// Get number of elements
pub fn numel(&self) -> usize

// Get number of dimensions  
pub fn ndim(&self) -> usize
```

### Data Access

```rust
// Convert tensor to Vec<f32>
pub fn to_vec(&self) -> Result<Vec<f32>>
```

## Arithmetic Operations

Tensor Frame supports standard arithmetic operations through operator overloading:

### Binary Operations

```rust
// Addition (element-wise)
let c = a + b;
let c = &a + &b;  // Avoid cloning

// Subtraction (element-wise)  
let c = a - b;

// Multiplication (element-wise)
let c = a * b;

// Division (element-wise)
let c = a / b;
```

### Broadcasting Rules

Addition operations automatically broadcast tensors following NumPy/PyTorch rules. Note: Broadcasting is currently only implemented for addition; other operations require matching shapes.

1. Dimensions are aligned from the right
2. Missing dimensions are treated as size 1
3. Dimensions of size 1 are expanded to match

```rust
let a = Tensor::ones(vec![2, 1, 3])?;    // Shape: [2, 1, 3]
let b = Tensor::ones(vec![1, 4, 1])?;    // Shape: [1, 4, 1]
let c = a + b;                           // Result: [2, 4, 3]
```

## Tensor Manipulation

### Reshaping

```rust
impl TensorOps for Tensor {
    // Change tensor shape (must preserve total elements)
    fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor>;
}
```

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
let reshaped = tensor.reshape(vec![3, 2])?;  // 2x3 -> 3x2
```

### Transposition

```rust
// Transpose 2D tensor (swap dimensions)
fn transpose(&self) -> Result<Tensor>;
```

```rust
let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let transposed = matrix.transpose()?;  // [[1,2],[3,4]] -> [[1,3],[2,4]]
```

### Dimension Manipulation

```rust
// Remove dimensions of size 1
fn squeeze(&self, dim: Option<usize>) -> Result<Tensor>;

// Add dimension of size 1
fn unsqueeze(&self, dim: usize) -> Result<Tensor>;
```

```rust
let tensor = Tensor::ones(vec![1, 3, 1])?;     // Shape: [1, 3, 1]
let squeezed = tensor.squeeze(None)?;          // Shape: [3]
let unsqueezed = squeezed.unsqueeze(0)?;       // Shape: [1, 3]
```

## Reduction Operations

### Full Reductions

```rust
// Sum all elements
fn sum(&self, axis: Option<usize>) -> Result<Tensor>;

// Mean of all elements
fn mean(&self, axis: Option<usize>) -> Result<Tensor>;
```

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

// Sum all elements -> scalar tensor
let total = tensor.sum(None)?;              // Result: 10.0

// Mean of all elements -> scalar tensor  
let average = tensor.mean(None)?;           // Result: 2.5
```

### Axis-specific Reductions

Note: Axis-specific reductions are not yet implemented in the CPU backend. Currently, only full tensor reductions (with `axis=None`) are supported.


## Display and Debug

Tensors implement comprehensive display formatting:

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
println!("{}", tensor);
// Output:
// Tensor([[1.0000, 2.0000],
//        [3.0000, 4.0000]], dtype=f32)
```

## Type Conversions

```rust
// Convert to Vec for external use
let data: Vec<f32> = tensor.to_vec()?;

// Clone (cheap - reference counted)
let cloned = tensor.clone();
```

## Performance Notes

- **Cloning**: Tensors use reference counting, so cloning is O(1)
- **Backend Selection**: Operations stay on the same backend when possible
- **Memory Layout**: Tensors use row-major (C-style) memory layout
- **Broadcasting**: Zero-copy when possible, falls back to explicit expansion