struct Push {
    xform: mat4x4f,
    location: vec3i,
    time: f32,
};

struct V2F {
    @builtin(position)
    position: vec4f,

    @location(0)
    mapping: vec2f,

    @location(1)
    translucent: u32,

    @location(2)
    shade: f32,

    @location(3)
    sky_light: f32,

    @location(4)
    tile: i32,

    @location(5)
    tint_mask: i32,

    @location(6)
    location: vec3u,

    @location(7)
    normal: vec3i,
};

struct Vertex {
    x: f32,
    y: f32,
    z: f32,
    u: f32,
    v: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    tile: i32,
    tint_mask: i32,
};

struct QuadRef {
    offset: u32,
    blob: u32,
};

@group(0) @binding(0)
var<storage> vertex_atlas: array<Vertex>;

@group(0) @binding(1)
var tiles: texture_2d_array<f32>;

@group(0) @binding(2)
var masks: texture_2d_array<f32>;

@group(0) @binding(3)
var sanpler: sampler;

@group(1) @binding(0)
var<storage> quad_buf: array<QuadRef>;

var<push_constant> p: Push;

@vertex
fn vert_main(
    @builtin(instance_index)
    instance_index: u32,

    @builtin(vertex_index)
    vertex_index: u32,
) -> V2F {
    let qr = quad_buf[vertex_index / 6u];

    var index = vertex_index % 6u;

    if index == 4u {
        index = 2u;
    } else if index == 5u {
        index = 1u;
    }

    let va = vertex_atlas[4u * qr.offset + index];

    var x = extractBits(qr.blob, 0u, 5u);
    var y = extractBits(qr.blob, 5u, 5u);
    var z = extractBits(qr.blob, 10u, 5u);
    let translucent = extractBits(qr.blob, 15u, 1u);
    let sky_exposure = extractBits(qr.blob, 16u, 4u);
    let num_copies = extractBits(qr.blob, 20u, 5u);
    var block_loc = vec3f(vec3u(x, y, z));

    var mapping = vec2f(va.u, va.v);

    // jmi2k: HACK!!!!
    if index == 1u || index == 3u || index == 5u {
        if va.nx == -1. {
            block_loc.z += f32(num_copies);
        } else if va.nx == 1. {
            block_loc.y += f32(num_copies);
        } else if va.ny == -1. {
            block_loc.x += f32(num_copies);
        } else if va.ny == 1. {
            block_loc.z += f32(num_copies);
        } else if va.nz == -1. {
            block_loc.y += f32(num_copies);
        } else if va.nz == 1. {
            block_loc.x += f32(num_copies);
        }

        mapping.x += f32(num_copies);
    }

    let cx = extractBits(instance_index, 0u, 8u);
    let cy = extractBits(instance_index, 8u, 8u);
    let cz = extractBits(instance_index, 16u, 8u);
    // jmi2k: suspicious of precision problems
    let position = vec3f(32u * vec3u(cx, cy, cz)) + vec3f(va.x, va.y, va.z) + block_loc + vec3f(p.location);
    let shade = max(0., dot(vec3f(abs(va.nx), abs(va.ny), va.nz), vec3f(0.5, 0.3, -0.6)));

    return V2F(
        p.xform * vec4f(position, 1.),
        mapping,
        translucent,
        shade,
        light(sky_exposure, 1.75),
        va.tile,
        va.tint_mask,
        vec3u(x, y, z),
        vec3i(vec3f(va.nx, va.ny, va.nz)),
    );
}

@fragment
fn frag_main(v: V2F) -> @location(0) vec4f {
    let tile = u32(abs(v.tile)) - 1u;
    let randomize = v.tile < 0;
    let mapping = fract(v.mapping);
    let copy_idx = u32(abs(floor(v.mapping.x)));
    var x = v.location.x;
    var y = v.location.y;
    var z = v.location.z;

    // jmi2k: HACK!!!!
    if v.normal.x == -1 {
        z += copy_idx;
    } else if v.normal.x == 1 {
        y += copy_idx;
    } else if v.normal.y == -1 {
        x += copy_idx;
    } else if v.normal.y == 1 {
        z += copy_idx;
    } else if v.normal.z == -1 {
        y += copy_idx;
    } else if v.normal.z == 1 {
        x += copy_idx;
    }

    let random = u32(fract(sin(dot(vec2f(vec2u(x, y)), vec2f(12.9898, 78.233))) * 43758.5453) * 4.) % 8u;
    var randomized_mapping = mapping;

    if randomize {
        if random == 1u {
            randomized_mapping *= vec2f(-1., 1.);
            randomized_mapping -= vec2f(-1., 0.);
        } else if random == 2u {
            randomized_mapping *= vec2f(1., -1.);
            randomized_mapping -= vec2f(0., -1.);
        } else if random == 3u {
            randomized_mapping *= vec2f(-1., -1.);
            randomized_mapping -= vec2f(-1., -1.);
        } else if random == 4u {
            randomized_mapping = vec2f(randomized_mapping.y, randomized_mapping.x);
            randomized_mapping *= vec2f(-1., 1.);
            randomized_mapping -= vec2f(-1., 0.);
        } else if random == 5u {
            randomized_mapping = vec2f(randomized_mapping.y, randomized_mapping.x);
            randomized_mapping *= vec2f(1., -1.);
            randomized_mapping -= vec2f(0., -1.);
        } else if random == 6u {
            randomized_mapping = vec2f(randomized_mapping.y, randomized_mapping.x);
            randomized_mapping *= vec2f(-1., -1.);
            randomized_mapping -= vec2f(-1., -1.);
        } else if random == 7u {
            randomized_mapping = vec2f(randomized_mapping.y, randomized_mapping.x);
            randomized_mapping *= vec2f(1., -1.);
            randomized_mapping -= vec2f(0., -1.);
        }
    }

    //let randomized_mappinggg = v.mapping;
    //var sample = textureSample(tiles, sanpler, randomized_mapping, tile);
    // jmi2k: we gain texture randomization, but we lost mipmapping.
    //var color_sample = textureLoad(tiles, vec2i(16.0 * randomized_mapping) % 16, tile, 0);
    var color_sample = textureSampleGrad(tiles, sanpler, randomized_mapping, tile, dpdx(v.mapping), dpdy(v.mapping));
    var mask_sample = textureLoad(masks, vec2i(16.0 * randomized_mapping) % 16, v.tint_mask, 0);

    if !bool(v.translucent) {
        color_sample.a = select(0., 1., color_sample.a >= 0.5);
        if color_sample.a == 0. { discard; }
    }

    if mask_sample.r == 1. {
        color_sample *= vec4f(0.529, 0.741, 0.341, 1.);
    }

    let shaded = color_sample - vec4f(color_sample.xyz * v.shade, 0.);
    let lit = shaded * vec4(vec3(v.sky_light), 1.);
    let fogged = lit; // jmi2k: todo
    return fogged;
}

fn light(sky_exposure: u32, gamma: f32) -> f32 {
    let ratio = f32(sky_exposure + 1u) / 16.;
    return pow(ratio, gamma);
}
