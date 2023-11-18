struct Push {
    xform: mat4x4f,
    unxform: mat4x4f,
    location: vec3i,
    time: u32,
};

struct V2F {
    @builtin(position)
    position: vec4f,

    @location(0)
    mapping: vec2f,

    @location(1)
    shade: f32,

    @location(2)
    sky_light: f32,

    @location(3)
    tile: u32,
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
    tile: u32,
};

@group(0) @binding(0)
var<storage> vertex_atlas: array<Vertex>;

@group(0) @binding(1)
var tiles: texture_2d_array<f32>;

@group(0) @binding(2)
var sanpler: sampler;

var<push_constant> p: Push;

@vertex
fn vert_main(
    @location(0)
    offset: u32,

    @location(1)
    blob: u32,
) -> V2F {
    let va = vertex_atlas[offset];

    let x = extractBits(blob, 0u, 5u);
    let y = extractBits(blob, 5u, 5u);
    let z = extractBits(blob, 10u, 5u);
    _ = extractBits(blob, 15u, 1u);
    let sky_exposure = extractBits(blob, 16u, 4u);
    let num_copies = extractBits(blob, 20u, 5u);
    let block_loc = vec3f(vec3u(x, y, z));
    let tile = va.tile;

    // jmi2k: suspicious of precision problems
    let position = vec3f(va.x, va.y, va.z) + block_loc + vec3f(p.location);
    let shade = max(0., dot(vec3f(abs(va.nx), abs(va.ny), va.nz), vec3f(0.5, 0.3, -0.6)));
    // jmi2k: this doesn't work apparently
    let mapping = vec2f(va.u + f32(num_copies), va.v);

    return V2F(
        p.xform * vec4f(position, 1.),
        mapping,
        shade,
        light(sky_exposure, 2.75),
        tile,
    );
}

@fragment
fn frag_main(v: V2F) -> @location(0) vec4f {
    var sample = textureSample(tiles, sanpler, v.mapping, v.tile);
    sample.a = select(0., 1., sample.a >= 0.5);
    if sample.a == 0. { discard; }

    let unprojected = p.unxform * v.position;
    let distance = length(unprojected);

    let shaded = sample - vec4f(sample.xyz * v.shade, 0.);
    let lit = shaded * vec4(vec3(v.sky_light), 1.);
    //let fogged = lit * (distance / 512.); // jmi2k: todo
    let fogged = lit;
    return fogged;
}

fn light(sky_exposure: u32, gamma: f32) -> f32 {
    let ratio = f32(sky_exposure + 1u) / 16.;
    return pow(ratio, gamma);
}
