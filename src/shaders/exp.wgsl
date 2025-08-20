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
    
    // Compute exp(x) with overflow protection
    // For very large x, clamp to prevent infinity
    if (x > 88.0) {
        // exp(88) ≈ 1.65e38, close to f32 max
        output[index] = 3.4028235e38; // f32::MAX
    } else if (x < -87.0) {
        // For very negative x, exp(x) ≈ 0
        output[index] = 0.0;
    } else {
        // Use built-in exp function for normal range
        output[index] = exp(x);
    }
}