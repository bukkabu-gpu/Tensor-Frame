
use std::collections::HashMap;
use std::fs;
use once_cell::sync::Lazy;

pub(crate) static SHADER_CACHE: Lazy<HashMap<Operation, String>> = Lazy::new(|| {
    let mut cache = HashMap::new();

    // Preload shaders
    for op in [Operation::Add, Operation::Sub, Operation::Mul, Operation::Div] {
        let shader_path = format!("src/gpu_acel/shader_code/{:?}.wgsl", op);
        let shader_code = fs::read_to_string(&shader_path)
            .expect(&format!("Failed to read shader file: {:?}", shader_path));
        cache.insert(op, shader_code);
    }

    cache
});

#[derive(Hash, Eq, PartialEq, Debug)]
pub enum Operation {
    Add,
    Mul,
    Div,
    Sub
}
