use tensor_frame::{Tensor, Result, TensorOps};

fn main() -> Result<()> {
    println!("=== Tensor Frame Examples ===\n");

    // Create some tensors  
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

    println!("Tensor A:\n{}\n", a);
    println!("Tensor B:\n{}\n", b);

    // Arithmetic operations
    println!("=== Arithmetic Operations ===");
    
    let sum = (a.clone() + b.clone())?;
    println!("A + B:\n{}\n", sum);

    let product = (a.clone() * b.clone())?;
    println!("A * B (element-wise):\n{}\n", product);

    // Reduction operations
    println!("=== Reduction Operations ===");
    
    let total = sum.sum(None)?;
    println!("Sum of all elements: {}\n", total);

    let average = product.mean(None)?;
    println!("Mean of product: {}\n", average);

    // Broadcasting example
    println!("=== Broadcasting ===");
    let c = Tensor::ones(vec![2, 1])?;  // Shape: [2, 1]
    let d = Tensor::ones(vec![1, 3])?;  // Shape: [1, 3]
    
    println!("Tensor C (2x1):\n{}\n", c);
    println!("Tensor D (1x3):\n{}\n", d);
    
    let broadcasted = (c + d)?;  // Should broadcast to [2, 3]
    println!("C + D (broadcasted to 2x3):\n{}\n", broadcasted);

    // Tensor manipulation
    println!("=== Tensor Manipulation ===");
    
    let reshaped = a.reshape(vec![1, 4])?;
    println!("A reshaped to [1, 4]:\n{}\n", reshaped);

    let transposed = a.transpose()?;
    println!("A transposed:\n{}\n", transposed);

    Ok(())
}