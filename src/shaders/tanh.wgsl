@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    let x = input[index];
    
    // Compute tanh(x) using the identity: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    // For numerical stability, we use different formulations for different ranges
    
    if (abs(x) > 15.0) {
        // For very large |x|, tanh(x) ≈ sign(x)
        output[index] = sign(x);
    } else if (abs(x) > 1.0) {
        // For moderate |x|, use the standard formula
        let exp_2x = exp(2.0 * x);
        output[index] = (exp_2x - 1.0) / (exp_2x + 1.0);
    } else {
        // For small |x|, use Taylor series expansion for better precision
        // tanh(x) ≈ x - x³/3 + 2x⁵/15 - 17x⁷/315 + ...
        let x2 = x * x;
        let x3 = x * x2;
        let x5 = x3 * x2;
        output[index] = x - x3 / 3.0 + 2.0 * x5 / 15.0;
    }
}