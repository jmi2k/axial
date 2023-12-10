use std::{array, collections::HashMap, f32::consts::PI, iter, mem, time::Instant, num::NonZeroU64};

use glam::{IVec3, Mat4, Vec4, Vec3};
use image::RgbaImage;
use wgpu::{
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt, DrawIndexedIndirectArgs, DrawIndirectArgs},
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
    TextureViewDimension, VertexBufferLayout, VertexState, VertexStepMode, AddressMode, FilterMode, IndexFormat, BufferDescriptor,
};

use crate::{asset::{Pack, MIP_LEVELS, TILE_LENGTH}, chunk, pov::Pov, world::World, types::{SideMap, Direction, Cube}};

use super::Gfx;

const REGION_LEN: i32 = 4;
const FOV: f32 = 90. * PI / 180.;
const ZNEAR: f32 = 1e-2;
const ZFAR: f32 = 1e4;

const REGION_SLOTS: &[BindGroupLayoutEntry; 1] = &[
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
];

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

pub type QuadRef = u64;

#[derive(Default)]
struct MeshRef {
    nonces: SideMap<Option<NonZeroU64>>,
    offset: u32,
    num_quads: u32,
}

impl MeshRef {
    pub fn len(&self) -> u32 {
        self.num_quads
    }

    pub fn end(&self) -> u32 {
        self.offset + self.num_quads
    }
}

struct Region {
    meshes: Cube<MeshRef, {REGION_LEN as usize}>,
    vertex_buf: Buffer,
    indirect_buf: Buffer,
    bind_group: BindGroup,
    num_indirects: u32,
}

impl Region {
    fn new(ctx: &Gfx, layout: &BindGroupLayout) -> Self {
        let vertex_buf = alloc_vertex_buf(ctx, 16);
        let bind_group = build_region_group(ctx, layout, &vertex_buf);

        Self {
            meshes: Cube::default(),
            vertex_buf,
            indirect_buf: alloc_indirect_buf(ctx),
            bind_group,
            num_indirects: 0,
        }
    }

    fn len(&self) -> u32 {
        let last_mesh = unsafe {
            self.meshes
                .get_unchecked(REGION_LEN as usize - 1)
                .get_unchecked(REGION_LEN as usize - 1)
                .get_unchecked(REGION_LEN as usize - 1)
        };

        last_mesh.end()
    }

    fn capacity(&self) -> u32 {
        (self.vertex_buf.size() / mem::size_of::<QuadRef>() as u64) as u32
    }

    fn mesh(&self, location: IVec3) -> &MeshRef {
        let IVec3 { x, y, z } = mask_chunk_loc(location);

        unsafe {
            self.meshes
                .get_unchecked(z as usize)
                .get_unchecked(y as usize)
                .get_unchecked(x as usize)
        }
    }

    fn mesh_mut(&mut self, location: IVec3) -> &mut MeshRef {
        let IVec3 { x, y, z } = mask_chunk_loc(location);

        unsafe {
            self.meshes
                .get_unchecked_mut(z as usize)
                .get_unchecked_mut(y as usize)
                .get_unchecked_mut(x as usize)
        }
    }

    fn write_indirects(&mut self, ctx: &Gfx) {
        self.num_indirects = 0;

        for z in 0..REGION_LEN {
        for y in 0..REGION_LEN {
        for x in 0..REGION_LEN {
            let mesh = self.mesh(IVec3 { x, y, z });

            if mesh.num_quads == 0 {
                continue;
            }

            let instance = u32::from_ne_bytes([
                x as u8,
                y as u8,
                z as u8,
                0,
            ]);

            let indirect = DrawIndirectArgs {
                vertex_count: 6 * mesh.num_quads,
                instance_count: 1,
                first_vertex: 6 * mesh.offset,
                first_instance: instance,
            };

            ctx.queue.write_buffer(&self.indirect_buf, (self.num_indirects as usize * mem::size_of::<DrawIndirectArgs>()) as u64, indirect.as_bytes());
            self.num_indirects += 1;
        }
        }
        }
    }

    fn load(&mut self, ctx: &Gfx, layout: &BindGroupLayout, location: IVec3, nonces: &SideMap<Option<NonZeroU64>>, quads: &[QuadRef], transient_buf: &mut Buffer) {
        const Q: u64 = mem::size_of::<QuadRef>() as u64;

        let location = mask_chunk_loc(location);
        let old_len = self.len();
        let mesh = self.mesh_mut(location);
        let Δlen = quads.len() as u32 - mesh.len();
        let old_mesh_end = mesh.end();
        let old_mesh_offset = mesh.offset;
        let num_quads_left = mesh.offset;
        let num_quads_right = old_len - mesh.end();

        // Adjust mesh reference
        mesh.num_quads = quads.len() as _;
        mesh.nonces = nonces.clone();

        // Ignore no-ops on empty meshes
        if quads.is_empty() && Δlen == 0 {
            return;
        }

        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor::default());

        if self.len() + Δlen > self.capacity() {
            // allocate new buffer and move there
            let new_vertex_buf = alloc_vertex_buf(ctx, self.len() + Δlen);

            // Copy left remaining of buffer
            encoder.copy_buffer_to_buffer(
                &self.vertex_buf,
                0,
                &new_vertex_buf,
                0,
                num_quads_left as u64 * Q);

            // Copy right remaining of buffer
            encoder.copy_buffer_to_buffer(
                &self.vertex_buf,
                old_mesh_end as u64 * Q,
                &new_vertex_buf,
                (old_mesh_end + Δlen) as u64 * Q,
                num_quads_right as u64 * Q);

            let commands = encoder.finish();
            ctx.queue.submit(iter::once(commands));

            // Replace region vertex buffer
            self.bind_group = build_region_group(ctx, layout, &new_vertex_buf);
            self.vertex_buf = new_vertex_buf;
        } else {
            // reuse current buffer and use the transient buffer
            let transient_len = transient_buf.size() as u32 / Q as u32;

            if num_quads_right > transient_len {
                *transient_buf = alloc_vertex_buf(ctx, num_quads_right.next_power_of_two());
            }

            // Copy right remaining to transient
            encoder.copy_buffer_to_buffer(
                &self.vertex_buf,
                old_mesh_end as u64 * Q,
                &transient_buf,
                0,
                num_quads_right as u64 * Q);

            // ...and back to the current buffer
            encoder.copy_buffer_to_buffer(
                &transient_buf,
                0,
                &self.vertex_buf,
                (old_mesh_end + Δlen) as u64 * Q,
                num_quads_right as u64 * Q);

            let commands = encoder.finish();
            ctx.queue.submit(iter::once(commands));
        }

        // Write quads to mesh gap
        ctx.queue.write_buffer(
            &self.vertex_buf,
            old_mesh_offset as u64 * Q,
            bytemuck::cast_slice(quads));

        let flattened_meshes = self.meshes.flatten_mut().flatten_mut();
        let start = location.z * REGION_LEN * REGION_LEN + location.y * REGION_LEN + location.x + 1;

        // Adjust right offsets
        for mesh in &mut flattened_meshes[start as usize..] {
            mesh.offset += Δlen;
        }

        self.write_indirects(ctx);
    }
}

fn mask_chunk_loc(location: IVec3) -> IVec3 {
    debug_assert!(
        location.min_element() >= -REGION_LEN && location.max_element() < REGION_LEN,
        "chunk location out of region bounds",
    );

    location & (REGION_LEN - 1)
}

fn mask_region_loc(location: IVec3) -> IVec3 {
    debug_assert!(
        location.min_element() >= i32::MIN / 32 && location.max_element() <= i32::MAX / 32,
        "region location out of world bounds",
    );

    location << REGION_LEN.trailing_zeros() >> REGION_LEN.trailing_zeros()
}

fn split_loc(location: IVec3) -> (IVec3, IVec3) {
    let chunk_loc = location >> REGION_LEN.trailing_zeros();
    let block_loc = location & (REGION_LEN - 1);

    (chunk_loc, block_loc)
}

fn merge_loc(region_loc: IVec3, chunk_loc: IVec3) -> IVec3 {
    let region_loc = mask_region_loc(region_loc);
    let chunk_loc = mask_chunk_loc(chunk_loc);

    region_loc << REGION_LEN.trailing_zeros() | chunk_loc
}

fn alloc_vertex_buf(ctx: &Gfx, num_quads: u32) -> Buffer {
    let descriptor = BufferDescriptor {
        label: None,
        size: num_quads as u64 * mem::size_of::<QuadRef>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };

    ctx.device.create_buffer(&descriptor)
}

fn alloc_indirect_buf(ctx: &Gfx) -> Buffer {
    let descriptor = BufferDescriptor {
        label: None,
        size: (REGION_LEN * REGION_LEN * REGION_LEN) as u64 * mem::size_of::<DrawIndirectArgs>() as u64,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };

    ctx.device.create_buffer(&descriptor)
}

pub struct Renderer {
    epoch: Instant,
    size: [u32; 2],
    depth: Texture,
    frame: Texture,
    sampler: Sampler,
    region_layout: BindGroupLayout,
    pack_layout: BindGroupLayout,
    pack_group: BindGroup,
    frame_layout: BindGroupLayout,
    frame_group: BindGroup,
    poly_chunk_pipeline: RenderPipeline,
    alpha_chunk_pipeline: RenderPipeline,
    wire_chunk_pipeline: RenderPipeline,
    reach_pipeline: RenderPipeline,
    post_pipeline: RenderPipeline,
    loaded_regions: HashMap<IVec3, Region, ahash::RandomState>,
    transient_buf: Buffer,
}

impl Renderer {
    pub fn new(ctx: &Gfx, pack: &Pack) -> Self {
        let region_descriptor = BindGroupLayoutDescriptor {
            label: None,
            entries: REGION_SLOTS,
        };

        let pack_descriptor = BindGroupLayoutDescriptor {
            label: None,
            entries: PACK_SLOTS,
        };

        let frame_descriptor = BindGroupLayoutDescriptor {
            label: None,
            entries: FRAME_SLOTS,
        };

        let region_layout = ctx.device.create_bind_group_layout(&region_descriptor);
        let pack_layout = ctx.device.create_bind_group_layout(&pack_descriptor);
        let frame_layout = ctx.device.create_bind_group_layout(&frame_descriptor);

        let (poly_chunk_pipeline, alpha_chunk_pipeline, wire_chunk_pipeline) = {
            let shader = ctx
                .device
                .create_shader_module(include_wgsl!("wgsl/chunk.wgsl"));

            #[rustfmt::skip]
            let push_constant_ranges = &[
                PushConstantRange { stages: ShaderStages::VERTEX, range: 0..80 },
                PushConstantRange { stages: ShaderStages::FRAGMENT, range: 76..84 },
            ];

            let layout_desc = PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&pack_layout, &region_layout],
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
                buffers: &[],
            };

            let solid_target = ColorTargetState {
                format: ctx.config.format,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            };

            let alpha_target = ColorTargetState {
                format: ctx.config.format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            };

            let solid_fragment = FragmentState {
                module: &shader,
                entry_point: "frag_main",
                targets: &[Some(solid_target)],
            };

            let alpha_fragment = FragmentState {
                module: &shader,
                entry_point: "frag_main",
                targets: &[Some(alpha_target)],
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
                fragment: Some(solid_fragment.clone()),
                depth_stencil: Some(solid_depth_stencil.clone()),
                multisample: MultisampleState::default(),
                multiview: None,
            };

            let alpha_pipeline_desc = RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                primitive: poly_primitive,
                vertex: vertex.clone(),
                fragment: Some(alpha_fragment),
                depth_stencil: Some(alpha_depth_stencil.clone()),
                multisample: MultisampleState::default(),
                multiview: None,
            };

            let wire_pipeline_desc = RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                primitive: wire_primitive,
                vertex,
                fragment: Some(solid_fragment),
                depth_stencil: Some(solid_depth_stencil),
                multisample: MultisampleState::default(),
                multiview: None,
            };

            let poly_pipeline = ctx.device.create_render_pipeline(&poly_pipeline_desc);
            let alpha_pipeline = ctx.device.create_render_pipeline(&alpha_pipeline_desc);
            let wire_pipeline = ctx.device.create_render_pipeline(&wire_pipeline_desc);

            (poly_pipeline, alpha_pipeline, wire_pipeline)
        };

        let reach_pipeline = {
            let shader = ctx
                .device
                .create_shader_module(include_wgsl!("wgsl/reach.wgsl"));

            #[rustfmt::skip]
            let push_constant_ranges = &[
                PushConstantRange { stages: ShaderStages::VERTEX, range: 0..84 },
                PushConstantRange { stages: ShaderStages::FRAGMENT, range: 80..84 },
            ];

            let layout_desc = PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges,
            };

            let layout = ctx.device.create_pipeline_layout(&layout_desc);

            let primitive = PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..PrimitiveState::default()
            };

            let vertex = VertexState {
                module: &shader,
                entry_point: "vert_main",
                buffers: &[],
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

            let pipeline_desc = RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                primitive,
                vertex: vertex.clone(),
                fragment: Some(fragment.clone()),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview: None,
            };

            ctx.device.create_render_pipeline(&pipeline_desc)
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
            mipmap_filter: FilterMode::Linear,
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
            region_layout,
            pack_layout,
            pack_group,
            frame_layout,
            frame_group,
            poly_chunk_pipeline,
            alpha_chunk_pipeline,
            wire_chunk_pipeline,
            reach_pipeline,
            post_pipeline,
            loaded_regions: HashMap::default(),
            transient_buf: alloc_vertex_buf(ctx, 1_024),
        }
    }

    pub fn update_pack(&mut self, ctx: &Gfx, pack: &Pack) {
        self.pack_group = build_pack_group(ctx, &self.pack_layout, pack, &self.sampler);
        self.invalidate_meshes();
    }

    pub fn invalidate_meshes(&mut self) {
        self.loaded_regions.clear();
    }

    // jmi2k: accept a closure instead?
    pub fn load_mesh(&mut self, ctx: &Gfx, location: IVec3, nonces: &SideMap<Option<NonZeroU64>>, quads: &[QuadRef], alpha_quads: &[QuadRef]) {
        let (region_loc, chunk_loc) = split_loc(location);
        let region = self.loaded_regions.entry(region_loc).or_insert_with(|| Region::new(ctx, &self.region_layout));
        region.load(ctx, &self.region_layout, chunk_loc, nonces, quads, &mut self.transient_buf);
    }

    pub fn unload_mesh(&mut self, location: IVec3) {
        // self.loaded_meshes.remove(&location);
    }

    pub fn has_mesh(&self, location: IVec3, nonces: &SideMap<Option<NonZeroU64>>) -> bool {
        let (region_loc, chunk_loc) = split_loc(location);
        let Some(region) = self.loaded_regions.get(&region_loc) else { return false; };
        &region.mesh(chunk_loc).nonces == nonces
    }

    fn render_chunks(&mut self, encoder: &mut CommandEncoder, /*world: &World,*/ pov: &Pov, max_distance: usize, wireframe: bool) {
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
        let xform = projection * pov.xform();
        let camera_loc = pov.location();
        let time = self.epoch.elapsed().as_secs_f32();

        let planes = [
            xform.row(3) + xform.row(0),
            xform.row(3) - xform.row(0),
            xform.row(3) + xform.row(1),
            xform.row(3) - xform.row(1),
            xform.row(2),
        ];

        let radius = f32::sqrt(3. * (32. * REGION_LEN as f32) * (32. * REGION_LEN as f32));

        let solid_pipeline = if wireframe { &self.wire_chunk_pipeline } else { &self.poly_chunk_pipeline };
        let alpha_pipeline = if wireframe { &self.wire_chunk_pipeline } else { &self.alpha_chunk_pipeline };

        let mut pass = encoder.begin_render_pass(&descriptor);
        pass.set_pipeline(solid_pipeline);
        pass.set_bind_group(0, &self.pack_group, &[]);
        pass.set_push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&xform));
        pass.set_push_constants(ShaderStages::VERTEX_FRAGMENT, 76, &time.to_ne_bytes());
        pass.set_push_constants(ShaderStages::FRAGMENT, 80, &0u32.to_ne_bytes());

        'render:
        for (location, mesh) in &self.loaded_regions {
            let Region { vertex_buf, bind_group, .. } = mesh;
            let location = camera_loc + merge_loc(chunk::merge_loc(*location, IVec3::ZERO), IVec3::ZERO);

            if vertex_buf.size() == 0 {
                continue;
            }

            let center = (location + 16 * REGION_LEN).as_vec3();

            for plane in planes {
                let spherical_distance = center.dot(plane.truncate()) + plane.w;
                if spherical_distance < -radius { continue 'render; }
            }

            pass.set_push_constants(ShaderStages::VERTEX, 64, bytemuck::bytes_of(&location));
            pass.set_bind_group(1, bind_group, &[]);
            pass.multi_draw_indirect(&mesh.indirect_buf, 0, mesh.num_indirects);
        }

        pass.set_pipeline(alpha_pipeline);
    }

    fn render_reach(&mut self, encoder: &mut CommandEncoder, pov: &Pov, location: IVec3, direction: Direction) {
        let frame_view = self.frame.create_view(&TextureViewDescriptor::default());

        let color_att = RenderPassColorAttachment {
            view: &frame_view,
            resolve_target: None,

            #[rustfmt::skip]
            ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
        };

        let descriptor = RenderPassDescriptor {
            color_attachments: &[Some(color_att)],
            depth_stencil_attachment: None,
            ..RenderPassDescriptor::default()
        };

        let Extent3d { width, height, .. } = self.frame.size();
        let aspect = width as f32 / height as f32;
        let projection = Mat4::perspective_rh(FOV, aspect, ZNEAR, ZFAR);
        let xform = projection * pov.xform();
        let location = location + pov.location();
        let time = self.epoch.elapsed().as_secs_f32();

        let mut pass = encoder.begin_render_pass(&descriptor);
        pass.set_pipeline(&self.reach_pipeline);
        pass.set_push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&xform));
        pass.set_push_constants(ShaderStages::VERTEX, 64, bytemuck::bytes_of(&location));
        pass.set_push_constants(ShaderStages::VERTEX, 76, &(direction as u32).to_ne_bytes());
        pass.set_push_constants(ShaderStages::VERTEX_FRAGMENT, 80, &time.to_ne_bytes());
        pass.draw(0..6, 0..1);
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

    pub fn render(&mut self, ctx: &Gfx, /*world: &World,*/ pov: &Pov, reached_face: Option<(IVec3, Direction)>, max_distance: usize, wireframe: bool) {
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

        self.render_chunks(&mut encoder, /*world,*/ pov, max_distance, wireframe);
        if let Some((location, direction)) = reached_face {
            self.render_reach(&mut encoder, pov, location, direction);
        }
        self.render_post(&mut encoder, &output);

        ctx.queue.submit(iter::once(encoder.finish()));
        output.present();

        // self.loaded_meshes.retain(|location, _| {
        //     let location = chunk::merge_loc(*location, IVec3::ZERO);
        //     let max_distance = (max_distance * 32) as f32;
        //     let distance = location.as_vec3().distance(pov.position);
        //     distance <= max_distance
        // });
    }
}

pub fn quad_ref(offset: usize, location: IVec3, sky_exposure: u8) -> QuadRef {
    debug_assert!(sky_exposure < 16, "sky exposure out of bounds");

    let location = chunk::mask_block_loc(location);
    let sky_exposure = sky_exposure & 15;

    offset as u64
        | (location.x as u64) << 32
        | (location.y as u64) << 37
        | (location.z as u64) << 42
        | (sky_exposure as u64) << 47
}

pub fn extend_quad_ref_w(quad_ref: &mut QuadRef) {
    *quad_ref += 1 << 51;
}

pub fn extend_quad_ref_h(quad_ref: &mut QuadRef) {
    *quad_ref += 1 << 56;
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

fn build_region_group(
    ctx: &Gfx,
    layout: &BindGroupLayout,
    vertex_buf: &Buffer,
) -> BindGroup {
    let quad_ref_binding = BufferBinding {
        buffer: &vertex_buf,
        offset: 0,
        size: None,
    };

    #[rustfmt::skip]
    let entries = &[
        BindGroupEntry { binding: 0, resource: BindingResource::Buffer(quad_ref_binding) },
    ];

    let descriptor = BindGroupDescriptor {
        label: None,
        layout,
        entries,
    };

    ctx.device.create_bind_group(&descriptor)
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
