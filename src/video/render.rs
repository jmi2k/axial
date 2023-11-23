use std::{array, collections::HashMap, f32::consts::PI, iter, mem, time::Instant, num::NonZeroU64};

use glam::{IVec3, Mat4};
use image::RgbaImage;
use wgpu::{
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferBinding, BufferBindingType, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandEncoder, CommandEncoderDescriptor, CompareFunction, DepthBiasState, DepthStencilState,
    Extent3d, Face, FragmentState, FrontFace, ImageCopyTexture, ImageDataLayout,
    LoadOp, MultisampleState, Operations, Origin3d, PipelineLayoutDescriptor,
    PolygonMode, PrimitiveState, PrimitiveTopology, PushConstantRange, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerDescriptor, ShaderStages, StencilState, StoreOp,
    SurfaceConfiguration, SurfaceTexture, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor,
    TextureViewDimension, VertexBufferLayout, VertexState, VertexStepMode, AddressMode,
};

use crate::{asset::{Pack, MIP_LEVELS, TILE_LENGTH}, chunk, pov::Pov, world::World, types::SideMap};

use super::Gfx;

const FOV: f32 = 90. * PI / 180.;
const ZNEAR: f32 = 1e-2;
const ZFAR: f32 = 1e4;

const VERTEX_LAYOUT: VertexBufferLayout<'static> = {
    let attributes = &vertex_attr_array! {
        0 => Uint32,
        1 => Uint32,
    };

    VertexBufferLayout {
        array_stride: mem::size_of::<VertexRef>() as _,
        step_mode: VertexStepMode::Vertex,
        attributes,
    }
};

const PACK_SLOTS: &[BindGroupLayoutEntry; 4] = &[
    BindGroupLayoutEntry {
        binding: 0,
        count: None,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    },
    BindGroupLayoutEntry {
        binding: 1,
        count: None,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
    },
    BindGroupLayoutEntry {
        binding: 2,
        count: None,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
    },
    BindGroupLayoutEntry {
        binding: 3,
        count: None,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    },
];

const FRAME_SLOTS: &[BindGroupLayoutEntry; 1] = &[BindGroupLayoutEntry {
    binding: 0,
    count: None,
    visibility: ShaderStages::FRAGMENT,
    ty: BindingType::Texture {
        sample_type: TextureSampleType::Float { filterable: true },
        view_dimension: TextureViewDimension::D2,
        multisampled: false,
    },
}];

const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

pub type VertexRef = u64;
pub type QuadRef = [VertexRef; 6];

struct GpuMesh {
    nonces: SideMap<Option<NonZeroU64>>,
    vertex_buf: Buffer,
    alpha_vertex_buf: Buffer,
}

pub struct Renderer {
    epoch: Instant,
    size: [u32; 2],
    depth: Texture,
    frame: Texture,
    sampler: Sampler,
    pack_layout: BindGroupLayout,
    pack_group: BindGroup,
    frame_layout: BindGroupLayout,
    frame_group: BindGroup,
    poly_chunk_pipeline: RenderPipeline,
    alpha_chunk_pipeline: RenderPipeline,
    wire_chunk_pipeline: RenderPipeline,
    post_pipeline: RenderPipeline,
    loaded_meshes: HashMap<IVec3, GpuMesh>,
}

impl Renderer {
    pub fn new(ctx: &Gfx, pack: &Pack) -> Self {
        let pack_descriptor = BindGroupLayoutDescriptor {
            label: None,
            entries: PACK_SLOTS,
        };

        let frame_descriptor = BindGroupLayoutDescriptor {
            label: None,
            entries: FRAME_SLOTS,
        };

        let pack_layout = ctx.device.create_bind_group_layout(&pack_descriptor);
        let frame_layout = ctx.device.create_bind_group_layout(&frame_descriptor);

        let (poly_chunk_pipeline, alpha_chunk_pipeline, wire_chunk_pipeline) = {
            let shader = ctx
                .device
                .create_shader_module(include_wgsl!("wgsl/chunk.wgsl"));

            #[rustfmt::skip]
            let push_constant_ranges = &[
                PushConstantRange { stages: ShaderStages::VERTEX, range: 0..144 },
                PushConstantRange { stages: ShaderStages::FRAGMENT, range: 140..144 },
            ];

            let layout_desc = PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&pack_layout],
                push_constant_ranges,
            };

            let layout = ctx.device.create_pipeline_layout(&layout_desc);

            let poly_primitive = PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..PrimitiveState::default()
            };

            let wire_primitive = PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Line,
                ..PrimitiveState::default()
            };

            let vertex = VertexState {
                module: &shader,
                entry_point: "vert_main",
                buffers: &[VERTEX_LAYOUT],
            };

            let target = ColorTargetState {
                format: ctx.config.format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            };

            let fragment = FragmentState {
                module: &shader,
                entry_point: "frag_main",
                targets: &[Some(target)],
            };

            let solid_depth_stencil = DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            };

            let alpha_depth_stencil = DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            };

            let poly_pipeline_desc = RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                primitive: poly_primitive,
                vertex: vertex.clone(),
                fragment: Some(fragment.clone()),
                depth_stencil: Some(solid_depth_stencil.clone()),
                multisample: MultisampleState::default(),
                multiview: None,
            };

            let alpha_pipeline_desc = RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                primitive: poly_primitive,
                vertex: vertex.clone(),
                fragment: Some(fragment.clone()),
                depth_stencil: Some(alpha_depth_stencil.clone()),
                multisample: MultisampleState::default(),
                multiview: None,
            };

            let wire_pipeline_desc = RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                primitive: wire_primitive,
                vertex,
                fragment: Some(fragment),
                depth_stencil: Some(solid_depth_stencil),
                multisample: MultisampleState::default(),
                multiview: None,
            };

            let poly_pipeline = ctx.device.create_render_pipeline(&poly_pipeline_desc);
            let alpha_pipeline = ctx.device.create_render_pipeline(&alpha_pipeline_desc);
            let wire_pipeline = ctx.device.create_render_pipeline(&wire_pipeline_desc);

            (poly_pipeline, alpha_pipeline, wire_pipeline)
        };

        let post_pipeline = {
            let shader = ctx
                .device
                .create_shader_module(include_wgsl!("wgsl/post.wgsl"));

            #[rustfmt::skip]
            let push_constant_ranges = &[
                PushConstantRange { stages: ShaderStages::FRAGMENT, range: 0..16 },
            ];

            let layout_desc = PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&frame_layout],
                push_constant_ranges,
            };

            let layout = ctx.device.create_pipeline_layout(&layout_desc);

            let vertex = VertexState {
                module: &shader,
                entry_point: "vert_main",
                buffers: &[],
            };

            let target = ColorTargetState {
                format: ctx.config.format,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            };

            let fragment = FragmentState {
                module: &shader,
                entry_point: "frag_main",
                targets: &[Some(target)],
            };

            let pipeline_desc = RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                primitive: PrimitiveState::default(),
                vertex,
                fragment: Some(fragment),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview: None,
            };

            ctx.device.create_render_pipeline(&pipeline_desc)
        };

        let sampler_descriptor = SamplerDescriptor {
            //min_filter: FilterMode::Linear,
            //mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            ..SamplerDescriptor::default()
        };

        let (depth, frame) = build_frames(ctx);
        let sampler = ctx.device.create_sampler(&sampler_descriptor);
        let pack_group = build_pack_group(ctx, &pack_layout, pack, &sampler);
        let frame_group = build_frame_group(ctx, &frame_layout, &frame);

        Self {
            epoch: Instant::now(),
            size: [ctx.config.width, ctx.config.height],
            depth,
            frame,
            sampler,
            pack_layout,
            pack_group,
            frame_layout,
            frame_group,
            poly_chunk_pipeline,
            alpha_chunk_pipeline,
            wire_chunk_pipeline,
            post_pipeline,
            loaded_meshes: HashMap::default(),
        }
    }

    pub fn update_pack(&mut self, ctx: &Gfx, pack: &Pack) {
        self.pack_group = build_pack_group(ctx, &self.pack_layout, pack, &self.sampler);
        self.loaded_meshes.clear();
    }

    // jmi2k: accept a closure instead?
    pub fn load_mesh(&mut self, ctx: &Gfx, location: IVec3, nonces: &SideMap<Option<NonZeroU64>>, quads: &[QuadRef], alpha_quads: &[QuadRef]) {
        match self.loaded_meshes.get(&location) {
            Some(entry) if &entry.nonces == nonces => return,
            _ => {}
        };

        let vertex_buf = {
            let vertices = quads.flatten();

            let descriptor = BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vertices),
                usage: BufferUsages::VERTEX,
            };

            ctx.device.create_buffer_init(&descriptor)
        };

        let alpha_vertex_buf = {
            let vertices = alpha_quads.flatten();

            let descriptor = BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vertices),
                usage: BufferUsages::VERTEX,
            };

            ctx.device.create_buffer_init(&descriptor)
        };

        let entry = GpuMesh {
            nonces: nonces.clone(),
            vertex_buf,
            alpha_vertex_buf,
        };

        self.loaded_meshes.insert(location, entry);
    }

    pub fn unload_mesh(&mut self, location: IVec3) {
        self.loaded_meshes.remove(&location);
    }

    pub fn has_mesh(&self, location: IVec3, nonces: &SideMap<Option<NonZeroU64>>) -> bool {
        self.loaded_meshes
            .get(&location)
            .map(|entry| &entry.nonces == nonces)
            .unwrap_or_default()
    }

    fn render_chunks(&mut self, encoder: &mut CommandEncoder, world: &World, pov: &Pov, max_distance: usize, wireframe: bool) {
        let frame_view = self.frame.create_view(&TextureViewDescriptor::default());
        let depth_view = self.depth.create_view(&TextureViewDescriptor::default());

        let color = Color {
            r: 0.266,
            g: 0.514,
            b: 1.,
            a: 1.,
        };

        let color_att = RenderPassColorAttachment {
            view: &frame_view,
            resolve_target: None,

            #[rustfmt::skip]
            ops: Operations { load: LoadOp::Clear(color), store: StoreOp::Store },
        };

        let depth_att = RenderPassDepthStencilAttachment {
            view: &depth_view,
            stencil_ops: None,

            #[rustfmt::skip]
            depth_ops: Some(Operations { load: LoadOp::Clear(1.), store: StoreOp::Store }),
        };

        let descriptor = RenderPassDescriptor {
            color_attachments: &[Some(color_att)],
            depth_stencil_attachment: Some(depth_att),
            ..RenderPassDescriptor::default()
        };

        let Extent3d { width, height, .. } = self.frame.size();
        let aspect = width as f32 / height as f32;
        let projection = Mat4::perspective_rh(FOV, aspect, ZNEAR, ZFAR);
        let xform = projection * Mat4::from(pov);
        let unxform = projection.inverse();
        let time = /* jmi2k: (world.time % TICKS_PER_DAY) as u32 */ 0u32;

        let solid_pipeline = if wireframe { &self.wire_chunk_pipeline } else { &self.poly_chunk_pipeline };
        let alpha_pipeline = if wireframe { &self.wire_chunk_pipeline } else { &self.alpha_chunk_pipeline };

        let mut pass = encoder.begin_render_pass(&descriptor);
        pass.set_bind_group(0, &self.pack_group, &[]);
        pass.set_pipeline(solid_pipeline);
        pass.set_push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&xform));
        pass.set_push_constants(ShaderStages::VERTEX, 64, bytemuck::bytes_of(&unxform));
        pass.set_push_constants(ShaderStages::VERTEX_FRAGMENT, 140, &time.to_ne_bytes());

        pass.set_pipeline(solid_pipeline);

        for (location, mesh) in &self.loaded_meshes {
            let GpuMesh { vertex_buf, .. } = mesh;
            let num_vertices = vertex_buf.size() / VERTEX_LAYOUT.array_stride;

            if num_vertices == 0 {
                continue;
            }

            let location = chunk::merge_loc(*location, IVec3::ZERO);
            pass.set_push_constants(ShaderStages::VERTEX, 128, bytemuck::bytes_of(&location));
            pass.set_vertex_buffer(0, vertex_buf.slice(..));
            pass.draw(0..num_vertices as _, 0..1);
        }

        pass.set_pipeline(alpha_pipeline);

        for (location, mesh) in &self.loaded_meshes {
            let GpuMesh { alpha_vertex_buf, .. } = mesh;
            let num_vertices = alpha_vertex_buf.size() / VERTEX_LAYOUT.array_stride;
            let location = chunk::merge_loc(*location, IVec3::ZERO);

            if num_vertices == 0 {
                continue;
            }

            pass.set_push_constants(ShaderStages::VERTEX, 128, bytemuck::bytes_of(&location));
            pass.set_vertex_buffer(0, alpha_vertex_buf.slice(..));
            pass.draw(0..num_vertices as _, 0..1);
        }
    }

    fn render_post(&self, encoder: &mut CommandEncoder, output: &SurfaceTexture) {
        #[rustfmt::skip]
        let output_view = output.texture.create_view(&TextureViewDescriptor::default());

        let color_att = RenderPassColorAttachment {
            view: &output_view,
            resolve_target: None,

            #[rustfmt::skip]
            ops: Operations { load: LoadOp::Clear(Color::default()), store: StoreOp::Store },
        };

        let descriptor = RenderPassDescriptor {
            color_attachments: &[Some(color_att)],
            depth_stencil_attachment: None,
            ..RenderPassDescriptor::default()
        };

        let Extent3d { width, height, .. } = self.frame.size();
        let viewport = [width, height];
        let time = self.epoch.elapsed().as_secs_f32();

        let mut pass = encoder.begin_render_pass(&descriptor);
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &self.frame_group, &[]);
        pass.set_push_constants(ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(&viewport));
        pass.set_push_constants(ShaderStages::FRAGMENT, 12, &time.to_ne_bytes());
        pass.draw(0..3, 0..1);
    }

    pub fn render(&mut self, ctx: &Gfx, world: &World, pov: &Pov, max_distance: usize, wireframe: bool) {
        let SurfaceConfiguration { width, height, .. } = ctx.config;
        let output = ctx.surface.get_current_texture().unwrap();

        if self.size != [width, height] {
            let (depth, frame) = build_frames(ctx);

            self.frame_group = build_frame_group(ctx, &self.frame_layout, &frame);
            self.depth = depth;
            self.frame = frame;
            self.size = [width, height];
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        self.render_chunks(&mut encoder, world, pov, max_distance, wireframe);
        self.render_post(&mut encoder, &output);

        ctx.queue.submit(iter::once(encoder.finish()));
        output.present();

        self.loaded_meshes.retain(|location, _| {
            let location = chunk::merge_loc(*location, IVec3::ZERO);
            let max_distance = (max_distance * 32) as f32;
            let distance = location.as_vec3().distance(pov.position);
            distance <= max_distance
        });
    }
}

pub fn quad_ref(base: usize, location: IVec3, translucent: bool, sky_exposure: u8) -> QuadRef {
    let offsets = [0, 1, 2, 3, 2, 1];
    array::from_fn(|idx| vertex_ref(4 * base + offsets[idx], location, translucent, sky_exposure))
}

pub fn extend_quad_ref(quad_ref: &mut QuadRef, increment: IVec3) {
    for (idx, vertex_ref) in quad_ref.iter_mut().enumerate() {
        if idx & 1 == 1 {
            *vertex_ref += 1 << 52;
            *vertex_ref +=
                (increment.x as u64) << 32
                | (increment.y as u64) << 37
                | (increment.z as u64) << 42;
        }
    }
}

fn vertex_ref(offset: usize, location: IVec3, translucent: bool, sky_exposure: u8) -> VertexRef {
    debug_assert!(sky_exposure < 16, "sky exposure out of bounds");

    let location = chunk::mask_block_loc(location);
    let sky_exposure = sky_exposure & 15;

    offset as u64
        | (location.x as u64) << 32
        | (location.y as u64) << 37
        | (location.z as u64) << 42
        | (translucent as u64) << 47
        | (sky_exposure as u64) << 48
}

fn build_frames(ctx: &Gfx) -> (Texture, Texture) {
    let SurfaceConfiguration { width, height, .. } = ctx.config;

    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let depth_descriptor = TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };

    let frame_descriptor = TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: ctx.config.format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };

    let depth = ctx.device.create_texture(&depth_descriptor);
    let frame = ctx.device.create_texture(&frame_descriptor);

    (depth, frame)
}

fn build_tile_texture(ctx: &Gfx, tiles: &[[RgbaImage; MIP_LEVELS]]) -> Texture {
    let size = Extent3d {
        width: TILE_LENGTH as _,
        height: TILE_LENGTH as _,
        depth_or_array_layers: tiles.len() as _,
    };

    let descriptor = TextureDescriptor {
        label: None,
        size,
        mip_level_count: MIP_LEVELS as _,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    };

    let atlas = ctx.device.create_texture(&descriptor);

    for (idx, tile) in tiles.iter().enumerate() {
        for (jdx, image) in tile.iter().enumerate() {
            let (width, height) = image.dimensions();

            let size = Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };

            let origin = Origin3d {
                x: 0,
                y: 0,
                z: idx as _,
            };

            let copy = ImageCopyTexture {
                texture: &atlas,
                mip_level: jdx as u32,
                origin,
                aspect: TextureAspect::All,
            };

            let layout = ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            };

            ctx.queue.write_texture(copy, image, layout, size);
        }
    }

    atlas
}

fn build_mask_texture(ctx: &Gfx, masks: &[RgbaImage]) -> Texture {
    let size = Extent3d {
        width: TILE_LENGTH as _,
        height: TILE_LENGTH as _,
        depth_or_array_layers: masks.len() as _,
    };

    let descriptor = TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    };

    let atlas = ctx.device.create_texture(&descriptor);

    for (idx, mask) in masks.iter().enumerate() {
        let (width, height) = mask.dimensions();

        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let origin = Origin3d {
            x: 0,
            y: 0,
            z: idx as _,
        };

        let copy = ImageCopyTexture {
            texture: &atlas,
            mip_level: 0,
            origin,
            aspect: TextureAspect::All,
        };

        let layout = ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        };

        ctx.queue.write_texture(copy, mask, layout, size);
    }

    atlas
}

fn build_pack_group(
    ctx: &Gfx,
    layout: &BindGroupLayout,
    pack: &Pack,
    sampler: &Sampler,
) -> BindGroup {
    let vertices = pack.vertex_atlas();
    let atlas = build_tile_texture(ctx, pack.tiles());
    let masks = build_mask_texture(ctx, pack.masks());
    let tile_view = atlas.create_view(&TextureViewDescriptor::default());
    let mask_view = masks.create_view(&TextureViewDescriptor::default());

    let ref_descriptor = BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertices),
        usage: BufferUsages::STORAGE,
    };

    let ref_buffer = ctx.device.create_buffer_init(&ref_descriptor);

    let ref_binding = BufferBinding {
        buffer: &ref_buffer,
        offset: 0,
        size: None,
    };

    #[rustfmt::skip]
    let entries = &[
        BindGroupEntry { binding: 0, resource: BindingResource::Buffer(ref_binding) },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&tile_view) },
        BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&mask_view) },
        BindGroupEntry { binding: 3, resource: BindingResource::Sampler(sampler)},
    ];

    let descriptor = BindGroupDescriptor {
        label: None,
        layout,
        entries,
    };

    ctx.device.create_bind_group(&descriptor)
}

fn build_frame_group(ctx: &Gfx, layout: &BindGroupLayout, frame: &Texture) -> BindGroup {
    let view = frame.create_view(&TextureViewDescriptor::default());

    #[rustfmt::skip]
    let entries = &[
        BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
    ];

    let descriptor = BindGroupDescriptor {
        label: None,
        layout,
        entries,
    };

    ctx.device.create_bind_group(&descriptor)
}
