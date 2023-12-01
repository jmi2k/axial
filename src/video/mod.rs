pub mod render;

use std::sync::Arc;

use wgpu::{
    Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, PresentMode, Queue,
    RequestAdapterOptions, Surface, SurfaceCapabilities, SurfaceConfiguration, TextureUsages,
};
use winit::{dpi::PhysicalSize, window::Window};

pub struct Gfx<'win> {
    pub surface: Surface<'win>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
}

impl<'win> Gfx<'win> {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let options = RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };

        let adapter = instance.request_adapter(&options).await.unwrap();

        let limits = Limits {
            max_push_constant_size: 128,
            ..Limits::default()
        };

        let descriptor = DeviceDescriptor {
            label: None,
            features: Features::PUSH_CONSTANTS | Features::POLYGON_MODE_LINE | Features::MULTI_DRAW_INDIRECT | Features::INDIRECT_FIRST_INSTANCE,
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

    pub fn toggle_vsync(&mut self) {
        self.config.present_mode = match self.config.present_mode {
            PresentMode::AutoVsync => PresentMode::AutoNoVsync,
            PresentMode::AutoNoVsync => PresentMode::AutoVsync,
            _ => unreachable!(),
        };

        self.surface.configure(&self.device, &self.config);
    }
}
