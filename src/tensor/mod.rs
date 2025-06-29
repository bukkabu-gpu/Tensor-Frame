mod broadcast;
pub mod dtype;
pub mod ops;
pub mod shape;

use crate::backend::{Backend, Storage, BACKEND};
use crate::error::{Result, TensorError};
use broadcast::broadcast_data;
use dtype::DType;
use ops::TensorOps;
use shape::Shape;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    dtype: DType,
    backend: Arc<dyn Backend>,
}

impl Tensor {
    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        Self::zeros_with_backend(shape, BACKEND.clone())
    }

    pub fn zeros_with_backend(shape: impl Into<Shape>, backend: Arc<dyn Backend>) -> Result<Self> {
        let shape = shape.into();
        let dtype = DType::F32;
        let storage = backend.zeros(&shape, dtype)?;
        Ok(Tensor {
            storage,
            shape,
            dtype,
            backend,
        })
    }

    pub fn ones(shape: impl Into<Shape>) -> Result<Self> {
        Self::ones_with_backend(shape, BACKEND.clone())
    }

    pub fn ones_with_backend(shape: impl Into<Shape>, backend: Arc<dyn Backend>) -> Result<Self> {
        let shape = shape.into();
        let dtype = DType::F32;
        let storage = backend.ones(&shape, dtype)?;
        Ok(Tensor {
            storage,
            shape,
            dtype,
            backend,
        })
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        Self::from_vec_with_backend(data, shape, BACKEND.clone())
    }

    pub fn from_vec_with_backend(
        data: Vec<f32>,
        shape: impl Into<Shape>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = backend.from_slice(&data, &shape)?;
        Ok(Tensor {
            storage,
            shape,
            dtype: DType::F32,
            backend,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    pub fn backend_type(&self) -> crate::backend::BackendType {
        self.backend.backend_type()
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        self.backend.to_vec_f32(&self.storage)
    }

    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        let flat_idx = self.shape.flatten_index(indices)?;
        let data = self.to_vec()?;
        Ok(data[flat_idx])
    }

    pub fn to_backend(&self, backend: Arc<dyn Backend>) -> Result<Self> {
        if self.backend.backend_type() == backend.backend_type() {
            return Ok(self.clone());
        }

        let data = self.to_vec()?;
        Self::from_vec_with_backend(data, self.shape.dims().to_vec(), backend)
    }

    fn ensure_same_backend(&self, other: &Self) -> Result<()> {
        if self.backend.backend_type() != other.backend.backend_type() {
            return Err(TensorError::BackendError(
                "Tensors must be on the same backend".to_string(),
            ));
        }
        Ok(())
    }

    fn broadcast_shapes(&self, other: &Self) -> Result<Shape> {
        self.shape.broadcast_with(&other.shape)
    }
}

impl Add for Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: Self) -> Self::Output {
        self.ensure_same_backend(&other)?;
        let result_shape = self.broadcast_shapes(&other)?;

        // Get data and broadcast if necessary
        let lhs_data = self.to_vec()?;
        let rhs_data = other.to_vec()?;

        let (lhs_broadcasted, rhs_broadcasted) = broadcast_data(
            &lhs_data,
            &self.shape,
            &rhs_data,
            &other.shape,
            &result_shape,
        )?;

        let lhs_storage = self.backend.from_slice(&lhs_broadcasted, &result_shape)?;
        let rhs_storage = self.backend.from_slice(&rhs_broadcasted, &result_shape)?;
        let storage = self.backend.add(&lhs_storage, &rhs_storage)?;

        Ok(Tensor {
            storage,
            shape: result_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl Sub for Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: Self) -> Self::Output {
        self.ensure_same_backend(&other)?;
        let result_shape = self.broadcast_shapes(&other)?;
        let storage = self.backend.sub(&self.storage, &other.storage)?;
        Ok(Tensor {
            storage,
            shape: result_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl Mul for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: Self) -> Self::Output {
        self.ensure_same_backend(&other)?;
        let result_shape = self.broadcast_shapes(&other)?;
        let storage = self.backend.mul(&self.storage, &other.storage)?;
        Ok(Tensor {
            storage,
            shape: result_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl Div for Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: Self) -> Self::Output {
        self.ensure_same_backend(&other)?;
        let result_shape = self.broadcast_shapes(&other)?;
        let storage = self.backend.div(&self.storage, &other.storage)?;
        Ok(Tensor {
            storage,
            shape: result_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl TensorOps for Tensor {
    fn matmul(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let storage = self.backend.matmul(&self.storage, &other.storage)?;
        // TODO: Calculate proper output shape for matmul
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    fn sum(&self, axis: Option<usize>) -> Result<Self> {
        let storage = self.backend.sum(&self.storage, axis)?;
        let shape = if axis.is_none() {
            Shape::scalar()
        } else {
            // TODO: Calculate reduced shape
            self.shape.clone()
        };
        Ok(Tensor {
            storage,
            shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    fn mean(&self, axis: Option<usize>) -> Result<Self> {
        let storage = self.backend.mean(&self.storage, axis)?;
        let shape = if axis.is_none() {
            Shape::scalar()
        } else {
            // TODO: Calculate reduced shape
            self.shape.clone()
        };
        Ok(Tensor {
            storage,
            shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_shape = Shape::new(new_shape)?;
        if self.shape.numel() != new_shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.numel()],
                got: vec![new_shape.numel()],
            });
        }
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    fn transpose(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidShape(
                "Transpose only supports 2D tensors".to_string(),
            ));
        }
        let dims = self.shape.dims();
        let new_shape = Shape::new(vec![dims[1], dims[0]])?;
        // TODO: Implement actual transpose logic
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    fn squeeze(&self, axis: Option<usize>) -> Result<Self> {
        let dims = self.shape.dims();
        let new_dims = if let Some(axis) = axis {
            if axis >= self.ndim() || dims[axis] != 1 {
                return Err(TensorError::InvalidShape(format!(
                    "Cannot squeeze axis {} with size {}",
                    axis, dims[axis]
                )));
            }
            dims.iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &d)| d)
                .collect()
        } else {
            dims.iter().filter(|&&d| d != 1).copied().collect()
        };

        let new_shape = Shape::new(new_dims)?;
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    fn unsqueeze(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(TensorError::InvalidShape(format!(
                "Axis {} out of range for {}D tensor",
                axis,
                self.ndim()
            )));
        }
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.insert(axis, 1);
        let new_shape = Shape::new(new_dims)?;
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.to_vec().map_err(|_| fmt::Error)?;
        let shape = self.shape().dims();

        write!(f, "Tensor(")?;

        if shape.len() == 1 {
            // 1D tensor: [1, 2, 3, 4]
            write!(f, "[")?;
            for (i, &val) in data.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", val)?;
            }
            write!(f, "]")?;
        } else if shape.len() == 2 {
            // 2D tensor: [[1, 2], [3, 4]]
            write!(f, "[")?;
            for row in 0..shape[0] {
                if row > 0 {
                    write!(f, ",\n       ")?;
                }
                write!(f, "[")?;
                for col in 0..shape[1] {
                    if col > 0 {
                        write!(f, ", ")?;
                    }
                    let idx = row * shape[1] + col;
                    write!(f, "{:.4}", data[idx])?;
                }
                write!(f, "]")?;
            }
            write!(f, "]")?;
        } else {
            // Higher dimensional tensors: show shape and first few elements
            write!(f, "shape={:?}, data=[", shape)?;
            let max_display = 8.min(data.len());
            for (i, &val) in data.iter().take(max_display).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", val)?;
            }
            if data.len() > max_display {
                write!(f, ", ...")?;
            }
            write!(f, "]")?;
        }

        write!(
            f,
            ", dtype={}, backend={:?})",
            self.dtype,
            self.backend_type()
        )
    }
}
