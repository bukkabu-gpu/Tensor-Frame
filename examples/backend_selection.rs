use tensor_frame::{Tensor, Result, TensorOps};

fn main() -> Result<()> {
    println!("🔧 Tensor Frame Backend Selection Examples");
    println!("==========================================\n");

    println!("Tensor Frame automatically selects the best available backend:");
    println!("Priority order: CUDA > WGPU > CPU\n");

    // Check which backends are compiled in
    println!("🔍 Compiled Backend Support:");
    #[cfg(feature = "cuda")]
    println!("  ✅ CUDA backend compiled in");
    #[cfg(not(feature = "cuda"))]
    println!("  ❌ CUDA backend not compiled in");
    
    #[cfg(feature = "wgpu")]
    println!("  ✅ WGPU backend compiled in");
    #[cfg(not(feature = "wgpu"))]
    println!("  ❌ WGPU backend not compiled in");
    
    println!("  ✅ CPU backend always available\n");

    // Demonstrate backend usage through tensor operations
    println!("🚀 Creating tensors (automatic backend selection):");
    
    let a = Tensor::ones(vec![1000, 1000])?;
    println!("  Created 1000x1000 tensor of ones");
    
    let b = Tensor::zeros(vec![1000, 1000])?;
    println!("  Created 1000x1000 tensor of zeros");
    
    println!("\n⚡ Performing operations:");
    let sum_result = a.sum(None)?;
    println!("  Sum of ones tensor: {}", sum_result);
    
    let mean_result = b.mean(None)?;
    println!("  Mean of zeros tensor: {}", mean_result);

    // Test backend fallback behavior
    println!("\n🔄 Backend Fallback Behavior:");
    println!("  - If CUDA is available and tensor operations work, CUDA is used");
    println!("  - If CUDA fails, automatically falls back to WGPU");
    println!("  - If WGPU fails, automatically falls back to CPU");
    println!("  - CPU backend is always available as final fallback");

    // Demonstrate tensor creation and operations
    println!("\n🧮 Backend Performance Comparison:");
    let size = vec![100, 100];
    
    let start = std::time::Instant::now();
    let tensor1 = Tensor::ones(size.clone())?;
    let tensor2 = Tensor::ones(size)?;
    let _result = (tensor1 + tensor2)?;
    let duration = start.elapsed();
    
    println!("  Addition of 100x100 tensors took: {:?}", duration);
    println!("  (Actual backend used depends on availability and compilation flags)");

    println!("\n💡 Backend Selection Tips:");
    println!("  - For CPU-only usage: Default compilation (no features)");
    println!("  - For GPU usage: Compile with --features=\"wgpu\" or --features=\"cuda\"");
    println!("  - For maximum performance: Compile with --features=\"cuda\" on NVIDIA systems");
    println!("  - For cross-platform GPU: Use --features=\"wgpu\"");

    println!("\n✅ Backend selection examples completed!");

    Ok(())
}