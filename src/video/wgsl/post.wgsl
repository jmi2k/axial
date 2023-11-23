struct Push {
    viewport: vec2u,
    time: f32,
};

struct V2F {
    @builtin(position)
    position: vec4f,
};

@group(0) @binding(0)
var frame: texture_2d<f32>;

var<push_constant> p: Push;

@vertex
fn vert_main(
    @builtin(vertex_index)
    index: u32,
) -> V2F {
    // Hardcoded vertices for a triangle covering the entire viewport
    var positions = array(
        vec2f(-1., -1.),
        vec2f(3., -1.),
        vec2f(-1., 3.),
    );

    let position = vec4f(positions[index], 0., 1.);
    return V2F(position);
}

@fragment
fn frag_main(v: V2F) -> @location(0) vec4f {
    let sample = textureLoad(frame, vec2i(v.position.xy), 0);
    // jmi2k: vignette doesn't work
    //return grayscale(vignette(sample, v.position.xy));
    return sample;
}

fn vignette(in: vec4f, point: vec2f) -> vec4f {
    // Hardcoded vignette parameters
    let strength = 50.;
    let extent = 0.1;

    let uv = point / vec2f(p.viewport);
    let foo = point * (1. - point.yx);
    let amount = pow(foo.x * foo.y * strength, extent);

    return mix(vec4f(0., 0., 0., 1.), in, clamp(amount / 100., 0., 1.));
}

fn grayscale(in: vec4f) -> vec4f {
    let gray = dot(in, vec4f(0.2126, 0.7152, 0.0722, 0.));
    return vec4(vec3(gray), in.a);
}
