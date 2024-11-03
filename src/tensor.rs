
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
}

impl Tensor {
    // Constructor methods, e.g., zeros and ones
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![1.0; size],
            shape,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        Tensor {
            data,
            shape
        }
    }

    // Get an element at the specified multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> f32 {
        assert_eq!(indices.len(), self.shape.len(), "Incorrect number of indices");
        let index = self.flatten_index(indices);
        self.data[index]
    }

    // Set an element at the specified multi-dimensional index
    pub fn set(&mut self, indices: &[usize], value: f32) {
        assert_eq!(indices.len(), self.shape.len(), "Incorrect number of indices");
        let index = self.flatten_index(indices);
        self.data[index] = value;
    }

    // Internal method to check if shapes match
    pub fn shapes_match(&self, other: &Tensor) -> bool {
        self.shape == other.shape
    }
    // Helper method to calculate the flat index from multi-dimensional indices
    fn flatten_index(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(&self.shape)
            .fold(0, |acc, (idx, dim)| acc * dim + idx)
    }

    pub fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.clone()
        }
    }
}


