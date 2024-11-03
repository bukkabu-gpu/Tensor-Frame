use wgpu::util::DeviceExt;
use crate::tensor::Tensor;
use std::collections::HashMap;
use crate::gpu_acel::setup_wgpu;

pub(crate) async fn run_on_gpu(
    t1: &Tensor,
    t2: &Tensor,
    shader_code: &str,
) -> Tensor {
    assert!(t1.shapes_match(t2), "Shape mismatch for addition");
    let (device, queue) = setup_wgpu().await;
    let size = (t1.data.len() * std::mem::size_of::<f32>()) as u64;
    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer A"),
        contents: bytemuck::cast_slice(&t1.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer B"),
        contents: bytemuck::cast_slice(&t2.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Compile the shader and create a compute pipeline
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Operation Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let compilation_options = wgpu::PipelineCompilationOptions {
        constants: &HashMap::new(),
        zero_initialize_workgroup_memory: false,
    };

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options,
        cache: None
    });

    // Bind groups to send data to the shader
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_result.as_entire_binding(),
            },
        ],
        label: Some("Bind Group"),
    });

    // Set up a command encoder to run the shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Operation Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = ((t1.data.len() as u32) + 63) / 64;
        cpass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Copy results back to a buffer we can read from the CPU
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buffer_result, 0, &staging_buffer, 0, size);

    // Submit the work to the queue and wait for the results
    queue.submit(Some(encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    // Retrieve the data and construct the result tensor
    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Tensor::from_vec(result_data,t1.shape.clone())
    } else {
        panic!("Failed to run compute on GPU!")
    }
}