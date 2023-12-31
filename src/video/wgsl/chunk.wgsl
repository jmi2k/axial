struct Push {
    xform: mat4x4f,
    location: vec3i,
    time: f32,
    translucent: u32,
};

struct V2F {
    @builtin(position)
    position: vec4f,

    @location(0)
    mapping: vec2f,

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
var<storage> qrefs: array<QuadRef>;

var<push_constant> p: Push;

@vertex
fn vert_main(
    @builtin(instance_index)
    instance_index: u32,

    @builtin(vertex_index)
    vref_index: u32,
) -> V2F {
    // Indices to access buffers
    let qref_index = vref_index / 6u;
    let vertex_index_6 = vref_index % 6u;
    // [0, 1, 2, 3, 2, 1] according to atlas' vertex order
    let vertex_index = vertex_index_6
        - 2u * u32(vertex_index_6 > 3u)
        - 2u * u32(vertex_index_6 > 4u);

    // Read face and vertex from buffers
    let qref = qrefs[qref_index];
    let vertex = vertex_atlas[4u * qref.offset + vertex_index];

    // Decompress qref blob
    // Chunk-relative coordinates
    let bx = extractBits(qref.blob, 0u, 5u);
    let by = extractBits(qref.blob, 5u, 5u);
    let bz = extractBits(qref.blob, 10u, 5u);
    // Face attributes
    let sky_exposure = extractBits(qref.blob, 15u, 4u);
    // Greedy mesh dimensions
    let width = extractBits(qref.blob, 19u, 5u);
    let height = extractBits(qref.blob, 24u, 5u);

    // Decompress instance index
    // Region-relative chunk coordinates
    let cx = extractBits(instance_index, 0u, 8u);
    let cy = extractBits(instance_index, 8u, 8u);
    let cz = extractBits(instance_index, 16u, 8u);

    // Decompress vertex
    var position = vec3f(vertex.x, vertex.y, vertex.z);
    var mapping = vec2f(vertex.u, vertex.v);
    let normal = vec3f(vertex.nx, vertex.ny, vertex.nz);

    // Block location given by combining all gathered data
    var location = (
        p.location                       // Premultiplied region location
        + 32 * vec3i(vec3u(cx, cy, cz))  // Chunk location
        + vec3i(vec3u(bx, by, bz))       // Block location
    );

    // Tweak vertices [1, 3, 5] to apply greedy meshing horizontally
    if bool(vertex_index & 1u) {
        if vertex.nx == -1. {
            location.z += i32(width);
        } else if vertex.nx == 1. {
            location.y += i32(width);
        } else if vertex.ny == -1. {
            location.x += i32(width);
        } else if vertex.ny == 1. {
            location.z += i32(width);
        } else if vertex.nz == -1. {
            location.y += i32(width);
        } else if vertex.nz == 1. {
            location.x += i32(width);
        }

        mapping.x += f32(width);
    }

    // Tweak vertices [2, 3, 4] to apply greedy meshing horizontally
    if vertex_index >= 2u && vertex_index <= 4u {
        if vertex.nx == -1. {
            location.y += i32(height);
        } else if vertex.nx == 1. {
            location.z += i32(height);
        } else if vertex.ny == -1. {
            location.z += i32(height);
        } else if vertex.ny == 1. {
            location.x += i32(height);
        } else if vertex.nz == -1. {
            location.x += i32(height);
        } else if vertex.nz == 1. {
            location.y += i32(height);
        }

        mapping.y -= f32(height);
    }

    position += vec3f(location);

    return V2F(
        p.xform * vec4f(position, 1.),
        mapping,
        shade(normal),
        light(sky_exposure, 1.75),
        vertex.tile,
        vertex.tint_mask,
        vec3u(bx, by, bz),
        vec3i(normal),
    );
}

@fragment
fn frag_main(v: V2F) -> @location(0) vec4f {
    let tile = u32(abs(v.tile)) - 1u;
    let mapping = fract(v.mapping);
    let copy_idx_w = u32(abs(floor(v.mapping.x)));
    let copy_idx_h = u32(abs(floor(v.mapping.y)));
    let randomize = v.tile < 0;
    var x = v.location.x;
    var y = v.location.y;
    var z = v.location.z;

    if v.normal.x == -1 {
        z += copy_idx_w;
        y += copy_idx_h;
    } else if v.normal.x == 1 {
        y += copy_idx_w;
        z += copy_idx_h;
    } else if v.normal.y == -1 {
        x += copy_idx_w;
        z += copy_idx_h;
    } else if v.normal.y == 1 {
        z += copy_idx_w;
        x += copy_idx_h;
    } else if v.normal.z == -1 {
        y += copy_idx_w;
        x += copy_idx_h;
    } else if v.normal.z == 1 {
        x += copy_idx_w;
        y += copy_idx_h;
    }

    let random = u32(randomize) * pcg(x ^ 2u * y ^ 3u * z);
    let angle = f32(random % 4u) * asin(1.);

    // Random rotation matrix
    let rotation = mat2x2(
        cos(angle), -sin(angle),
        sin(angle), cos(angle),
    );

    // Apply random rotation to mapping
    let mapping_0 = mapping - 0.5;
    let rotated_mapping_0 = rotation * mapping_0;
    let rotated_mapping = rotated_mapping_0 + 0.5;

    // Sample texture with unchopped UV derivatives to avoid artifacts
    var color_sample = textureSampleGrad(
        tiles,
        sanpler,
        rotated_mapping,
        tile,
        dpdx(v.mapping),
        dpdy(v.mapping),
    );

    // Threshold alpha if the block is not translucent
    color_sample.a = select(
        round(color_sample.a),
        color_sample.a,
        bool(p.translucent),
    );

    // Discard invisible pixels
    if color_sample.a == 0. { discard; }

    // Load value from mask to apply tint
    var mask_sample = textureLoad(
        masks,
        vec2i(16.0 * rotated_mapping) % 16,
        v.tint_mask,
        0,
    );

    // Apply tint based on the R channel — G and B are currently unused!
    let biome_tint = vec4f(0.529, 0.741, 0.341, 1.);
    color_sample *= mix(
        vec4f(1.),
        biome_tint,
        mask_sample.r,
    );

    let shaded = color_sample - vec4f(color_sample.xyz * v.shade, 0.);
    let lit = shaded * vec4(vec3(v.sky_light), 1.);
    return lit;
}

fn pcg(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

    return (word >> 22u) ^ word;
}

fn shade(normal: vec3f) -> f32 {
    let incidence = vec3f(abs(normal.xy), normal.z);
    let weights = vec3f(0.5, 0.3, -0.6);

    return max(0., dot(incidence, weights));
}

fn light(sky_exposure: u32, gamma: f32) -> f32 {
    let ratio = f32(sky_exposure + 1u) / 16.;
    return pow(ratio, gamma);
}
