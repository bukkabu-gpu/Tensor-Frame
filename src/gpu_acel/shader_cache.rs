
use std::collections::HashMap;
use std::fs;
use once_cell::sync::Lazy;

static SHADER_CACHE: Lazy<HashMap<&str, String>> = Lazy::new(|| {
    let mut cache = HashMap::new();

    // Preload shaders
    let operations = ["add", "subtract", "multiply", "divide"];
    for &op in &operations {
        let shader_path = format!("gpu_accel/shader_code/{}.wgsl", op);
        let shader_code = fs::read_to_string(&shader_path).expect(&format!("Failed to read shader file: {}", shader_path));
        cache.insert(op, shader_code);
    }

    cache
});

pub enum Operations {
    Add,
    Mul,
    Div,
    Sub
}