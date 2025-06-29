use crate::error::{Result, TensorError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Result<Self> {
        if dims.is_empty() {
            return Err(TensorError::InvalidShape(
                "Shape cannot be empty".to_string(),
            ));
        }
        Ok(Shape { dims })
    }

    pub fn scalar() -> Self {
        Shape { dims: vec![1] }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn stride(&self) -> Vec<usize> {
        let mut stride = vec![1; self.ndim()];
        for i in (0..self.ndim() - 1).rev() {
            stride[i] = stride[i + 1] * self.dims[i + 1];
        }
        stride
    }

    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        let ndim = self.ndim().max(other.ndim());
        let mut result_dims = vec![1; ndim];

        for i in 0..ndim {
            let self_dim = if i < ndim - self.ndim() {
                1
            } else {
                self.dims[i - (ndim - self.ndim())]
            };

            let other_dim = if i < ndim - other.ndim() {
                1
            } else {
                other.dims[i - (ndim - other.ndim())]
            };

            if self_dim == other_dim {
                result_dims[i] = self_dim;
            } else if self_dim == 1 {
                result_dims[i] = other_dim;
            } else if other_dim == 1 {
                result_dims[i] = self_dim;
            } else {
                return Err(TensorError::BroadcastError(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    self.dims, other.dims
                )));
            }
        }

        Ok(Shape { dims: result_dims })
    }

    pub fn can_broadcast_to(&self, target: &Shape) -> bool {
        if self.ndim() > target.ndim() {
            return false;
        }

        let offset = target.ndim() - self.ndim();
        for i in 0..self.ndim() {
            let self_dim = self.dims[i];
            let target_dim = target.dims[i + offset];
            if self_dim != target_dim && self_dim != 1 {
                return false;
            }
        }
        true
    }

    pub fn flatten_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim(),
                got: indices.len(),
            });
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.dims[i] {
                return Err(TensorError::InvalidIndex {
                    index: indices.to_vec(),
                    shape: self.dims.clone(),
                });
            }
        }

        let stride = self.stride();
        Ok(indices.iter().zip(stride.iter()).map(|(i, s)| i * s).sum())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims).expect("Invalid shape")
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec()).expect("Invalid shape")
    }
}
