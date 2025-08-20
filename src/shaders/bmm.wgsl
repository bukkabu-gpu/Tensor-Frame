// Batched matrix multiplication shader for 3D tensors
// Computes C = A * B where A is B x M x K and B is B x K x N, resulting in C being B x M x N

@group(0) @binding(0)
var<storage, read> batch_a: array<f32>;

@group(0) @binding(1)
var<storage, read> batch_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> result: array<f32>;

@group(0) @binding(3)
var<uniform> dimensions: vec4<u32>;  // [batch_size, M, K, N]

@compute @workgroup_size(8, 8, 4) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let row = global_id.x;
    let col = global_id.y;
    
    let batch_size = dimensions.x;
    let M = dimensions.y;
    let K = dimensions.z;
    let N = dimensions.w;
    
    // Check bounds
    if (batch >= batch_size || row >= M || col >= N) {
        return;
    }
    
    var sum = 0.0;
    
    // Calculate base offsets for this batch
    let a_batch_offset = batch * M * K;
    let b_batch_offset = batch * K * N;
    let result_batch_offset = batch * M * N;
    
    // Compute dot product of row from A with column from B for this batch
    for (var k = 0u; k < K; k = k + 1u) {
        let a_val = batch_a[a_batch_offset + row * K + k];        // A[batch, row, k]
        let b_val = batch_b[b_batch_offset + k * N + col];        // B[batch, k, col]
        sum = sum + a_val * b_val;
    }
    
    // Store result
    result[result_batch_offset + row * N + col] = sum;
}