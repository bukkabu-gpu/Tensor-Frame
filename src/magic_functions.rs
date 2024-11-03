use std::ops::{Add,Sub,Div,Mul};
use crate::tensor::Tensor;
use crate::gpu_acel::{
    run::run_on_gpu,
    shader_cache::{
        SHADER_CACHE,
        Operation
    }
};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        // Ensure both lists have the same length
        if self.data.len() != other.data.len() {
            panic!("Lists must have the same length");
        }

        // Create a new List with the element-wise sum
        let summed_elements: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor::from_vec(summed_elements, self.shape)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        // Ensure both lists have the same length
        if self.data.len() != other.data.len() {
            panic!("Lists must have the same length");
        }

        // Create a new List with the element-wise sum
        let summed_elements: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Tensor::from_vec(summed_elements, self.shape)
    }
}
impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        // Ensure both lists have the same length
        if self.data.len() != other.data.len() {
            panic!("Lists must have the same length");
        }

        // Create a new List with the element-wise sum
        let summed_elements: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Tensor::from_vec(summed_elements, self.shape)
    }
}
impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        // Ensure both lists have the same length
        if self.data.len() != other.data.len() {
            panic!("Lists must have the same length");
        }

        // Create a new List with the element-wise sum
        let summed_elements: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect();

        Tensor::from_vec(summed_elements, self.shape)
    }
}

impl Tensor {

    pub async fn add(self, other: Tensor) -> Tensor {
        let shader = SHADER_CACHE.get(&Operation::Add).expect("Shader code not found");
        run_on_gpu(&self, &other, shader).await
    }
    pub async fn div(self, other: Tensor) -> Tensor {
        let shader = SHADER_CACHE.get(&Operation::Div).expect("Shader code not found");
        run_on_gpu(&self, &other, shader).await
    }
    pub async fn mul(self, other: Tensor) -> Tensor {
        let shader = SHADER_CACHE.get(&Operation::Mul).expect("Shader code not found");
        run_on_gpu(&self, &other, shader).await
    }
    pub async fn sub(self, other: Tensor) -> Tensor {
        let shader = SHADER_CACHE.get(&Operation::Sub).expect("Shader code not found");
        run_on_gpu(&self, &other, shader).await
    }
}