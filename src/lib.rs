pub mod backend;
pub mod error;
pub mod tensor;

pub use backend::{Backend, BackendType};
pub use error::{Result, TensorError};
pub use tensor::{dtype::DType, ops::TensorOps, shape::Shape, Tensor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        println!("Created tensor: {}", tensor);
    }

    #[test]
    fn test_tensor_arithmetic() {
        let a = Tensor::ones(vec![2, 3]).unwrap();
        let b = Tensor::ones(vec![2, 3]).unwrap();
        let c = (a + b).unwrap();
        println!("Addition result: {}", c);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.to_vec().unwrap(), data);
    }

    #[test]
    fn test_broadcasting() {
        let a = Tensor::ones(vec![2, 1]).unwrap();
        let b = Tensor::ones(vec![1, 3]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }
}
