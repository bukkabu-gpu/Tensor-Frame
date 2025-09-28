use std::process::Command;

fn main() {
    let status = Command::new("nvcc")
        .args(&[
            "src/kernel.cu",
            "-c",
            "-o",
            "target/kernel.o",
            "--compiler-options",
            "-fPIC",
            "--generate-code=arch=compute_89,code=sm_89",
        ])
        .status()
        .expect("Failed to run nvcc");

    if !status.success() {
        panic!("CUDA kernel compilation failed");
    }

    // リンク処理など...
}
