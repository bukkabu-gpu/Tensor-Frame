// Matrix multiplication shader for 2D matrices
// Computes C = A * B where A is M x K and B is K x N, resulting in C being M x N

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1) 
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> result: array<f32>;

@group(0) @binding(3)
var<uniform> dimensions: vec4<u32>;  // [M, K, K, N] for validation and indexing

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    let M = dimensions.x;
    let K = dimensions.y; 
    let N = dimensions.w;
    
    // Check bounds
    if (row >= M || col >= N) {
        return;
    }
    
    var sum = 0.0;
    
    // Compute dot product of row from A with column from B
    for (var k = 0u; k < K; k = k + 1u) {
        let a_val = matrix_a[row * K + k];        // A[row, k]
        let b_val = matrix_b[k * N + col];        // B[k, col] 
        sum = sum + a_val * b_val;
    }
    
    // Store result
    result[row * N + col] = sum;
}