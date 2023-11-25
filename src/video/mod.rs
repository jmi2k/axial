pub mod render;

use wgpu::{
    Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, PresentMode, Queue,
    RequestAdapterOptions, Surface, SurfaceCapabilities, SurfaceConfiguration, TextureUsages,
};
use winit::{dpi::PhysicalSize, window::Window};

pub struct Gfx {
    pub surface: Surface,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
}

impl Gfx {
    pub async fn new(window: &Window) -> Self {
        let instance = Instance::default();
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let options = RequestAdapterOptions {
            //power_preference: PowerPreference::HighPerformance,
            power_preference: PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };

        let adapter = instance.request_adapter(&options).await.unwrap();

        let limits = Limits {
            max_push_constant_size: 256,
            ..Limits::default()
        };

        let descriptor = DeviceDescriptor {
            label: None,
            features: Features::PUSH_CONSTANTS | Features::POLYGON_MODE_LINE,
            limits,
        };

        let (device, queue) = adapter.request_device(&descriptor, None).await.unwrap();

        let SurfaceCapabilities {
            formats,
            alpha_modes,
            ..
        } = surface.get_capabilities(&adapter);

        let PhysicalSize { width, height } = window.inner_size();

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: formats[0],
            width,
            height,
            present_mode: PresentMode::AutoVsync,
            alpha_mode: alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
        }
    }

    pub fn resize_viewport(&mut self, new_size: PhysicalSize<u32>) {
        let PhysicalSize { width, height } = new_size;

        if width * height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
