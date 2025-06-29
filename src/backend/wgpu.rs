use super::{Backend, BackendType, Storage, WgpuStorage};
use crate::error::{Result, TensorError};
use crate::tensor::{shape::Shape, dtype::DType};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| TensorError::BackendError(format!("Failed to create async runtime: {}", e)))?;
        
        rt.block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|e| TensorError::BackendError(format!("Failed to find WGPU adapter: {}", e)))?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Tensor Frame Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                        trace: wgpu::Trace::Off,
                    }
                )
                .await
                .map_err(|e| TensorError::BackendError(format!("Failed to create device: {}", e)))?;

            Ok(WgpuBackend { device, queue })
        })
    }
}

impl Backend for WgpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Wgpu
    }

    fn is_available(&self) -> bool {
        true
    }

    fn zeros(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        let size = shape.numel();
        let data = vec![0.0f32; size];
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Zeros Buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Storage::Wgpu(WgpuStorage { buffer: std::sync::Arc::new(buffer) }))
    }

    fn ones(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        let size = shape.numel();
        let data = vec![1.0f32; size];
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ones Buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Storage::Wgpu(WgpuStorage { buffer: std::sync::Arc::new(buffer) }))
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Storage::Wgpu(WgpuStorage { buffer: std::sync::Arc::new(buffer) }))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Wgpu(a), Storage::Wgpu(b)) => {
                self.execute_binary_op(&a.buffer, &b.buffer, include_str!("../shaders/add.wgsl"))
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Wgpu(a), Storage::Wgpu(b)) => {
                self.execute_binary_op(&a.buffer, &b.buffer, include_str!("../shaders/sub.wgsl"))
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Wgpu(a), Storage::Wgpu(b)) => {
                self.execute_binary_op(&a.buffer, &b.buffer, include_str!("../shaders/mul.wgsl"))
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::Wgpu(a), Storage::Wgpu(b)) => {
                self.execute_binary_op(&a.buffer, &b.buffer, include_str!("../shaders/div.wgsl"))
            }
            _ => Err(TensorError::BackendError("Storage type mismatch".to_string())),
        }
    }

    fn matmul(&self, _lhs: &Storage, _rhs: &Storage) -> Result<Storage> {
        Err(TensorError::BackendError("Matmul not yet implemented for WGPU".to_string()))
    }

    fn sum(&self, _storage: &Storage, _axis: Option<usize>) -> Result<Storage> {
        Err(TensorError::BackendError("Sum not yet implemented for WGPU".to_string()))
    }

    fn mean(&self, _storage: &Storage, _axis: Option<usize>) -> Result<Storage> {
        Err(TensorError::BackendError("Mean not yet implemented for WGPU".to_string()))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        match storage {
            Storage::Wgpu(wgpu_storage) => {
                let buffer = &wgpu_storage.buffer;
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| TensorError::BackendError(format!("Failed to create async runtime: {}", e)))?;
                
                rt.block_on(async {
                    let buffer_size = buffer.size();
                    let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Staging Buffer"),
                        size: buffer_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Copy Encoder"),
                    });

                    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
                    self.queue.submit(Some(encoder.finish()));

                    let buffer_slice = staging_buffer.slice(..);
                    let (sender, receiver) = futures::channel::oneshot::channel();
                    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                        sender.send(result).unwrap();
                    });

                    let _ = self.device.poll(wgpu::MaintainBase::Wait);
                    receiver.await.unwrap().unwrap();

                    let data = buffer_slice.get_mapped_range();
                    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                    drop(data);
                    staging_buffer.unmap();

                    Ok(result)
                })
            }
            _ => Err(TensorError::BackendError("Invalid storage type".to_string())),
        }
    }
}

impl WgpuBackend {
    fn execute_binary_op(&self, lhs: &wgpu::Buffer, rhs: &wgpu::Buffer, shader_source: &str) -> Result<Storage> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| TensorError::BackendError(format!("Failed to create async runtime: {}", e)))?;
        
        rt.block_on(async {
            let buffer_size = lhs.size();
            let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Result Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Binary Op Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Binary Op Pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lhs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: result_buffer.as_entire_binding(),
                    },
                ],
                label: Some("Binary Op Bind Group"),
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Binary Op Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Binary Op Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let workgroup_count = ((buffer_size / 4) as u32 + 63) / 64; // Assuming f32 (4 bytes)
                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));
            Ok(Storage::Wgpu(WgpuStorage { buffer: std::sync::Arc::new(result_buffer) }))
        })
    }
}