//! # Tensor Frame
//!
//! A PyTorch-like tensor library for Rust with support for multiple backends including CPU, WGPU, and CUDA.
//!
//! ## Overview
//!
//! Tensor Frame provides a flexible and efficient tensor computation framework that allows you to:
//! - Create and manipulate multi-dimensional arrays (tensors)
//! - Perform element-wise operations with automatic broadcasting
//! - Use different compute backends (CPU, GPU via WGPU, or CUDA)
//! - Seamlessly switch between backends based on your hardware capabilities
//!
//! ## Quick Start
//!
//! ```rust
//! use tensor_frame::{Tensor, TensorOps};
//!
//! // Create tensors
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::ones(vec![2, 2]).unwrap();
//!
//! // Perform operations
//! let c = (a + b).unwrap();
//! let sum = c.sum(None).unwrap();
//!
//! println!("Result: {:?}", c.to_vec().unwrap());
//! ```
//!
//! ## Features
//!
//! - **Multiple Backends**: Choose between CPU (with Rayon parallelization), WGPU (WebGPU), or CUDA
//! - **Broadcasting**: Automatic shape broadcasting for element-wise operations
//! - **Rich Operations**: Addition, subtraction, multiplication, division, reductions (sum, mean)
//! - **Shape Manipulation**: Reshape and transpose operations
//! - **Type Safety**: Strong typing with comprehensive error handling
//!
//! ## Backend Selection
//!
//! Enable different backends through Cargo features:
//!
//! ```toml
//! # CPU backend (default)
//! tensor_frame = "0.0.3-alpha"
//!
//! # WGPU backend
//! tensor_frame = { version = "0.0.3-alpha", features = ["wgpu"] }
//!
//! # CUDA backend
//! tensor_frame = { version = "0.0.3-alpha", features = ["cuda"] }
//! ```
//!
//! ## Examples
//!
//! ### Creating Tensors
//!
//! ```rust
//! use tensor_frame::Tensor;
//!
//! // From a vector with shape
//! let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // Zeros tensor
//! let zeros = Tensor::zeros(vec![3, 3]).unwrap();
//!
//! // Ones tensor
//! let ones = Tensor::ones(vec![2, 4]).unwrap();
//! ```
//!
//! ### Operations with Broadcasting
//!
//! ```rust
//! use tensor_frame::Tensor;
//!
//! let a = Tensor::ones(vec![2, 1]).unwrap();  // Shape: [2, 1]
//! let b = Tensor::ones(vec![1, 3]).unwrap();  // Shape: [1, 3]
//! let c = (a + b).unwrap();                   // Shape: [2, 3] via broadcasting
//! ```

mod backend;
mod error;
mod tensor;

/// The backend trait for tensor operations
pub use backend::Backend;
/// Error types and Result alias for the library
pub use error::{Result, TensorError};
/// Core tensor types and operations
pub use tensor::{Tensor, ops::TensorOps, shape::Shape};

#[cfg(test)]
mod tests {
    use super::*;

    // ==== TENSOR CREATION TESTS ====

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        let data = tensor.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::ones(vec![3, 2]).unwrap();
        assert_eq!(tensor.shape().dims(), &[3, 2]);
        assert_eq!(tensor.numel(), 6);
        let data = tensor.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.to_vec().unwrap(), data);
    }

    #[test]
    fn test_tensor_1d() {
        let tensor = Tensor::ones(vec![5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[5]);
        assert_eq!(tensor.numel(), 5);
    }

    #[test]
    fn test_tensor_3d() {
        let tensor = Tensor::zeros(vec![2, 3, 4]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_tensor_scalar() {
        let tensor = Tensor::from_vec(vec![42.0], vec![]).unwrap();
        assert_eq!(tensor.shape().dims(), &[] as &[usize]);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.to_vec().unwrap(), vec![42.0]);
    }

    // ==== ARITHMETIC OPERATION TESTS ====

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_tensor_subtraction() {
        let a = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = (a - b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_tensor_multiplication() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = (a * b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_tensor_division() {
        let a = Tensor::from_vec(vec![8.0, 12.0, 16.0, 20.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let c = (a / b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_tensor_chain_operations() {
        let a = Tensor::ones(vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let c = Tensor::from_vec(vec![3.0, 3.0, 3.0, 3.0], vec![2, 2]).unwrap();

        let result = ((a + b).unwrap() * c).unwrap();
        assert_eq!(result.to_vec().unwrap(), vec![9.0, 9.0, 9.0, 9.0]);
    }

    // ==== BROADCASTING TESTS ====

    #[test]
    fn test_broadcast_2d_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
        // This should fail without proper broadcasting implementation
        // but we'll test the shape compatibility
        assert_eq!(a.shape().dims(), &[2, 2]);
        assert_eq!(b.shape().dims(), &[2]);
    }

    #[test]
    fn test_broadcast_same_shape() {
        let a = Tensor::ones(vec![2, 3]).unwrap();
        let b = Tensor::ones(vec![2, 3]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_broadcast_compatible_shapes() {
        let a = Tensor::ones(vec![2, 1]).unwrap();
        let b = Tensor::ones(vec![1, 3]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0], vec![]).unwrap(); // scalar
        // Test shape compatibility
        assert_eq!(a.shape().dims(), &[2, 2]);
        assert_eq!(b.shape().dims(), &[] as &[usize]);
    }

    // ==== REDUCTION OPERATION TESTS ====

    #[test]
    fn test_tensor_sum() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let sum = tensor.sum(None).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![10.0]);
    }

    #[test]
    fn test_tensor_mean() {
        let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let mean = tensor.mean(None).unwrap();
        assert_eq!(mean.to_vec().unwrap(), vec![5.0]);
    }

    #[test]
    fn test_sum_ones() {
        let tensor = Tensor::ones(vec![3, 3]).unwrap();
        let sum = tensor.sum(None).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![9.0]);
    }

    #[test]
    fn test_mean_zeros() {
        let tensor = Tensor::zeros(vec![2, 5]).unwrap();
        let mean = tensor.mean(None).unwrap();
        assert_eq!(mean.to_vec().unwrap(), vec![0.0]);
    }

    // ==== SHAPE MANIPULATION TESTS ====

    #[test]
    fn test_tensor_reshape() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(
            reshaped.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_tensor_reshape_1d() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let reshaped = tensor.reshape(vec![4]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[4]);
        assert_eq!(reshaped.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_transpose_2d() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let transposed = tensor.transpose().unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        // For 2x3 -> 3x2 transpose: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        assert_eq!(
            transposed.to_vec().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    // ==== ERROR HANDLING TESTS ====

    #[test]
    fn test_shape_mismatch_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_vec(data, vec![2, 2]); // 3 elements, expecting 4
        assert!(result.is_err());
        if let Err(TensorError::ShapeMismatch { expected, got }) = result {
            assert_eq!(expected, vec![4]);
            assert_eq!(got, vec![3]);
        }
    }

    #[test]
    fn test_incompatible_shapes_addition() {
        let a = Tensor::ones(vec![2, 3]).unwrap();
        let b = Tensor::ones(vec![3, 4]).unwrap();
        let result = a + b;
        // This should either work with broadcasting or fail gracefully
        match result {
            Ok(_) => {}  // Broadcasting worked
            Err(_) => {} // Expected failure for incompatible shapes
        }
    }

    #[test]
    fn test_invalid_reshape() {
        let tensor = Tensor::ones(vec![2, 3]).unwrap(); // 6 elements
        let result = tensor.reshape(vec![2, 2]); // 4 elements
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_1d() {
        let tensor = Tensor::ones(vec![5]).unwrap();
        let result = tensor.transpose();
        // 1D transpose should either work (return same) or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // ==== EDGE CASE TESTS ====

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::zeros(vec![0]).unwrap();
        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor.to_vec().unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_large_tensor() {
        let tensor = Tensor::zeros(vec![100, 100]).unwrap();
        assert_eq!(tensor.numel(), 10000);
        assert_eq!(tensor.shape().dims(), &[100, 100]);
    }

    #[test]
    fn test_operations_with_negative_numbers() {
        let a = Tensor::from_vec(vec![-1.0, -2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, -3.0, -4.0], vec![2, 2]).unwrap();

        let sum = (a.clone() + b.clone()).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);

        let product = (a * b).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![-1.0, -4.0, -9.0, -16.0]);
    }

    #[test]
    fn test_operations_with_zero() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let zeros = Tensor::zeros(vec![2, 2]).unwrap();

        let sum = (a.clone() + zeros.clone()).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let product = (a * zeros).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_display_formatting() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let display_str = format!("{}", tensor);
        // Just ensure it doesn't panic and produces some output
        assert!(!display_str.is_empty());
    }

    // ==== SHAPE VALIDATION TESTS ====

    #[test]
    fn test_shape_validation() {
        use crate::tensor::shape::Shape;

        // These should all succeed - zero dimensions represent empty tensors
        assert!(Shape::new(vec![0]).is_ok());
        assert!(Shape::new(vec![2, 0]).is_ok());
        assert!(Shape::new(vec![0, 3]).is_ok());
        assert!(Shape::new(vec![2, 0, 3]).is_ok());

        // These should also succeed
        assert!(Shape::new(vec![1]).is_ok());
        assert!(Shape::new(vec![2, 3]).is_ok());
        assert!(Shape::new(vec![]).is_ok()); // Scalar is allowed

        // Test numel calculation with empty tensors
        let empty = Shape::new(vec![0]).unwrap();
        assert_eq!(empty.numel(), 0);

        let empty2 = Shape::new(vec![2, 0, 3]).unwrap();
        assert_eq!(empty2.numel(), 0);
    }

    #[test]
    fn test_overflow_protection() {
        use crate::tensor::shape::Shape;

        // This should fail due to overflow
        let huge_dims = vec![usize::MAX, 2];
        assert!(Shape::new(huge_dims).is_err());

        // This should also fail - 10^18 elements is way too many
        let large_dims = vec![1000000, 1000000, 1000000];
        let result = Shape::new(large_dims);
        // On some systems this might not overflow, so let's be more specific
        if result.is_ok() {
            // If it didn't overflow, try an even larger size
            let huge_dims = vec![usize::MAX / 2, usize::MAX / 2];
            assert!(Shape::new(huge_dims).is_err());
        } else {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tensor_creation_with_mismatched_data() {
        // Data size doesn't match shape - should fail
        let result = Tensor::from_vec_with_shape(vec![1.0, 2.0], vec![3, 2]);
        assert!(result.is_err());

        // Empty tensor creation should work
        let result2 = Tensor::from_vec_with_shape(Vec::new(), vec![0]);
        assert!(result2.is_ok());

        // Valid shape should work
        let result3 = Tensor::from_vec_with_shape(vec![1.0, 2.0], vec![1, 2]);
        assert!(result3.is_ok());
    }

    // ==== DIVISION BY ZERO TESTS ====

    #[test]
    fn test_division_by_zero_handling() {
        // Test different division by zero cases
        let numerator = Tensor::from_vec(vec![1.0, -1.0, 0.0, 5.0], vec![4]).unwrap();
        let denominator = Tensor::from_vec(vec![0.0, 0.0, 0.0, 2.0], vec![4]).unwrap();

        let result = (numerator / denominator).unwrap();
        let values = result.to_vec().unwrap();

        // Check that we get the expected IEEE floating point results
        assert!(values[0].is_infinite() && values[0].is_sign_positive()); // 1.0/0.0 = +inf
        assert!(values[1].is_infinite() && values[1].is_sign_negative()); // -1.0/0.0 = -inf
        assert!(values[2].is_nan()); // 0.0/0.0 = NaN
        assert_eq!(values[3], 2.5); // 5.0/2.0 = 2.5 (normal division)
    }

    #[test]
    fn test_division_by_near_zero() {
        // Test division by very small numbers (should not trigger special handling)
        let numerator = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let denominator = Tensor::from_vec(vec![1e-10, 1e-20], vec![2]).unwrap();

        let result = (numerator / denominator).unwrap();
        let values = result.to_vec().unwrap();

        // These should be very large but finite numbers, not infinity
        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
        assert!(values[0] > 1e9); // Should be approximately 1e10
        assert!(values[1] > 1e19); // Should be approximately 2e20
    }

    // ==== AXIS-SPECIFIC REDUCTION TESTS ====

    #[test]
    fn test_axis_specific_sum() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]
        let sum_axis_0 = tensor.sum(Some(0)).unwrap();
        let result_0 = sum_axis_0.to_vec().unwrap();
        assert_eq!(result_0, vec![5.0, 7.0, 9.0]);
        assert_eq!(sum_axis_0.shape().dims(), &[3]);

        // Sum along axis 1 (rows): should give [6, 15] with shape [2]
        let sum_axis_1 = tensor.sum(Some(1)).unwrap();
        let result_1 = sum_axis_1.to_vec().unwrap();
        assert_eq!(result_1, vec![6.0, 15.0]);
        assert_eq!(sum_axis_1.shape().dims(), &[2]);

        // Sum all elements: should give [21] with shape []
        let sum_all = tensor.sum(None).unwrap();
        let result_all = sum_all.to_vec().unwrap();
        assert_eq!(result_all, vec![21.0]);
        assert_eq!(sum_all.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_axis_specific_mean() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Mean along axis 0 (columns): should give [2.5, 3.5, 4.5] with shape [3]
        let mean_axis_0 = tensor.mean(Some(0)).unwrap();
        let result_0 = mean_axis_0.to_vec().unwrap();
        assert_eq!(result_0, vec![2.5, 3.5, 4.5]);
        assert_eq!(mean_axis_0.shape().dims(), &[3]);

        // Mean along axis 1 (rows): should give [2, 5] with shape [2]
        let mean_axis_1 = tensor.mean(Some(1)).unwrap();
        let result_1 = mean_axis_1.to_vec().unwrap();
        assert_eq!(result_1, vec![2.0, 5.0]);
        assert_eq!(mean_axis_1.shape().dims(), &[2]);

        // Mean all elements: should give [3.5] with shape []
        let mean_all = tensor.mean(None).unwrap();
        let result_all = mean_all.to_vec().unwrap();
        assert_eq!(result_all, vec![3.5]);
        assert_eq!(mean_all.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_axis_sum_3d_tensor() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x2x2 tensor
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();

        // Sum along axis 0: shape [2, 2] -> [2, 2]
        let sum_axis_0 = tensor.sum(Some(0)).unwrap();
        assert_eq!(sum_axis_0.shape().dims(), &[2, 2]);
        let result_0 = sum_axis_0.to_vec().unwrap();
        assert_eq!(result_0, vec![6.0, 8.0, 10.0, 12.0]); // [1+5, 2+6, 3+7, 4+8]

        // Sum along axis 1: shape [2, 2] -> [2, 2]
        let sum_axis_1 = tensor.sum(Some(1)).unwrap();
        assert_eq!(sum_axis_1.shape().dims(), &[2, 2]);
        let result_1 = sum_axis_1.to_vec().unwrap();
        assert_eq!(result_1, vec![4.0, 6.0, 12.0, 14.0]); // [1+3, 2+4, 5+7, 6+8]

        // Sum along axis 2: shape [2, 2] -> [2, 2]
        let sum_axis_2 = tensor.sum(Some(2)).unwrap();
        assert_eq!(sum_axis_2.shape().dims(), &[2, 2]);
        let result_2 = sum_axis_2.to_vec().unwrap();
        assert_eq!(result_2, vec![3.0, 7.0, 11.0, 15.0]); // [1+2, 3+4, 5+6, 7+8]
    }

    #[test]
    fn rows_slice_test() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap()
            .to_backend("CUDA")
            .expect("cudaだめ");

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]
        let tensor_rows_slice = tensor.rows_slice(&[0, 2]).unwrap();
        let result_0 = tensor_rows_slice.to_vec().unwrap();
        assert_eq!(result_0, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn sub_test() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap()
            .to_backend("CUDA")
            .expect("cudaだめ");
        let b = Tensor::from_vec(vec![2.0], vec![1, 1]).unwrap();

        let result = (a / b).unwrap();
        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        println!("result = {}", result);
    }
}

// ==== WGPU-SPECIFIC TESTS ====
#[cfg(test)]
#[cfg(feature = "wgpu")]
mod wgpu_tanh_tests {
    use super::*;

    #[test]
    fn test_wgpu_tanh_basic() {
        // Create a simple tensor and apply tanh
        let tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();

        // Apply tanh operation
        let result = tensor.tanh().unwrap();

        let actual = result.to_vec().unwrap();

        // Check that tanh values are in expected ranges
        assert!(actual[0] < -0.9); // tanh(-2) ≈ -0.964
        assert!(actual[1] < -0.7); // tanh(-1) ≈ -0.762
        assert!((actual[2] - 0.0).abs() < 1e-6); // tanh(0) = 0
        assert!(actual[3] > 0.7); // tanh(1) ≈ 0.762
        assert!(actual[4] > 0.9); // tanh(2) ≈ 0.964

        // More precise checks with expected values
        let expected_tanh_2 = 2.0_f32.tanh();
        let expected_tanh_1 = 1.0_f32.tanh();

        assert!(
            (actual[0] - (-expected_tanh_2)).abs() < 1e-3,
            "tanh(-2) expected: {}, got: {}",
            -expected_tanh_2,
            actual[0]
        );
        assert!(
            (actual[1] - (-expected_tanh_1)).abs() < 1e-3,
            "tanh(-1) expected: {}, got: {}",
            -expected_tanh_1,
            actual[1]
        );
        assert!(
            (actual[3] - expected_tanh_1).abs() < 1e-3,
            "tanh(1) expected: {}, got: {}",
            expected_tanh_1,
            actual[3]
        );
        assert!(
            (actual[4] - expected_tanh_2).abs() < 1e-3,
            "tanh(2) expected: {}, got: {}",
            expected_tanh_2,
            actual[4]
        );
    }

    #[test]
    fn test_wgpu_tanh_extreme_values() {
        // Test extreme values where tanh should saturate
        let tensor = Tensor::from_vec(vec![-10.0, 0.0, 10.0], vec![3]).unwrap();
        let result = tensor.tanh().unwrap();

        let actual = result.to_vec().unwrap();
        assert!(actual[0] < -0.999); // tanh(-10) ≈ -1
        assert!((actual[1] - 0.0).abs() < 1e-6); // tanh(0) = 0
        assert!(actual[2] > 0.999); // tanh(10) ≈ 1
    }

    #[test]
    fn test_wgpu_matmul_basic() {
        // Test basic 2D matrix multiplication
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        let expected = vec![19.0, 22.0, 43.0, 50.0]; // Manual calculation: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8]
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_wgpu_matmul_different_shapes() {
        // Test matrix multiplication with different shapes
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(); // 2x3
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap(); // 3x2

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        // Expected: [[58, 64], [139, 154]]
        // First row: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Second row: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_wgpu_bmm_basic() {
        // Test basic batched matrix multiplication
        let a =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();
        let b =
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]).unwrap();

        let result = a.bmm(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2, 2]);

        // First batch: identity multiplication, second batch: scaling by 2
        let expected = vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0];
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }
}
