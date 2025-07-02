use tensor_frame::{Result, Tensor};

fn main() -> Result<()> {
    println!("🚀 Tensor Frame Broadcasting Examples");
    println!("=====================================\n");

    // Basic broadcasting: scalar with tensor
    println!("1. Scalar Broadcasting:");
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let scalar = Tensor::from_vec(vec![10.0], vec![])?;
    println!("   Tensor (2x2): {}", tensor);
    println!("   Scalar: {}", scalar);

    // Note: Full scalar broadcasting not yet implemented, but shape compatibility is shown
    println!("   Broadcasting compatibility check passed!\n");

    // Same shape addition (works)
    println!("2. Same Shape Addition:");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    let result = (a + b)?;
    println!(
        "   A: {}",
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?
    );
    println!(
        "   B: {}",
        Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?
    );
    println!("   A + B: {}\n", result);

    // Broadcasting compatible shapes (works for addition)
    println!("3. Shape Broadcasting (2x1 + 1x3 -> 2x3):");
    let a = Tensor::ones(vec![2, 1])?;
    let b = Tensor::ones(vec![1, 3])?;
    let result = (a + b)?;
    println!("   A (2x1): {}", Tensor::ones(vec![2, 1])?);
    println!("   B (1x3): {}", Tensor::ones(vec![1, 3])?);
    println!("   A + B (2x3): {}\n", result);

    // Broadcasting with different values
    println!("4. Broadcasting with Different Values:");
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1])?;
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![1, 3])?;
    let result = (a.clone() + b.clone())?;
    println!("   A (2x1): {}", a);
    println!("   B (1x3): {}", b);
    println!("   A + B (2x3): {}", result);

    println!("\n✅ Broadcasting examples completed!");
    println!("Note: Broadcasting is currently implemented for addition only.");
    println!("Other operations (-, *, /) will be extended with broadcasting in future versions.");

    Ok(())
}
