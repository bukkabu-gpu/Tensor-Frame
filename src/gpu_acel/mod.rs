pub(crate) mod run;
pub(crate) mod shader_cache;

use wgpu;
pub async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    // Create an instance of wgpu
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor{
        backends:wgpu::Backends::all(),
        flags: Default::default(),
        dx12_shader_compiler: Default::default(),
        gles_minor_version: Default::default(),
    });

    // Request an adapter that supports the features you need
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            ..Default::default()
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Request a device and queue
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: Default::default(),
            required_limits: Default::default(),
            // You can provide a shader compiler if needed
            // shader_compiler: Some(wgpu::ShaderCompiler::default()),
            memory_hints: Default::default(),
        }, None)
        .await
        .expect("Failed to create device");

    (device, queue)
}