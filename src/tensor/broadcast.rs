use crate::error::{Result, TensorError};
use crate::tensor::shape::Shape;

pub fn broadcast_data(
    lhs_data: &[f32],
    lhs_shape: &Shape,
    rhs_data: &[f32], 
    rhs_shape: &Shape,
    result_shape: &Shape,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let lhs_broadcasted = broadcast_single(lhs_data, lhs_shape, result_shape)?;
    let rhs_broadcasted = broadcast_single(rhs_data, rhs_shape, result_shape)?;
    Ok((lhs_broadcasted, rhs_broadcasted))
}

fn broadcast_single(data: &[f32], from_shape: &Shape, to_shape: &Shape) -> Result<Vec<f32>> {
    if from_shape == to_shape {
        return Ok(data.to_vec());
    }

    if !from_shape.can_broadcast_to(to_shape) {
        return Err(TensorError::BroadcastError(format!(
            "Cannot broadcast shape {:?} to {:?}",
            from_shape.dims(),
            to_shape.dims()
        )));
    }

    let result_size = to_shape.numel();
    let mut result = vec![0.0; result_size];
    
    let from_dims = from_shape.dims();
    let to_dims = to_shape.dims();
    let _to_stride = to_shape.stride();
    let from_stride = from_shape.stride();
    
    // Handle broadcasting by expanding dimensions
    let dim_offset = to_dims.len() - from_dims.len();
    
    for i in 0..result_size {
        let mut from_idx = 0;
        let mut temp_i = i;
        
        for (dim_idx, &to_dim) in to_dims.iter().enumerate().rev() {
            let coord = temp_i % to_dim;
            temp_i /= to_dim;
            
            if dim_idx >= dim_offset {
                let from_dim_idx = dim_idx - dim_offset;
                let from_dim = from_dims[from_dim_idx];
                
                if from_dim == 1 {
                    // Broadcasting: use index 0 for this dimension
                } else {
                    from_idx += coord * from_stride[from_dim_idx];
                }
            }
        }
        
        result[i] = data[from_idx];
    }
    
    Ok(result)
}