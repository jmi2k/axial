struct Push {
    xform: mat4x4f,
    location: vec3i,
    direction: u32,
    time: f32,
};

var<push_constant> p: Push;

@vertex
fn vert_main(
    @builtin(vertex_index)
    index: u32,
) -> @builtin(position) vec4f {
    var positions = array<vec2f, 6>(
        vec2f(0., 0.),
        vec2f(1., 0.),
        vec2f(0., 1.),
        vec2f(1., 1.),
        vec2f(1., 0.),
        vec2f(0., 1.),
    );

    var u = positions[index].x;
    var v = positions[index].y;
    var xyz = vec3f(1.);

    if p.direction == 0u {
        xyz = vec3f(0., 1. - u, v);
    } else if p.direction == 1u {
        xyz = vec3f(1., u, v);
    } else if p.direction == 2u {
        xyz = vec3f(u, 0., v);
    } else if p.direction == 3u {
        xyz = vec3f(1. - u, 1., v);
    } else if p.direction == 4u {
        xyz = vec3f(u, 1. - v, 0.);
    } else if p.direction == 5u {
        xyz = vec3f(u, v, 1.);
    }

    return p.xform * vec4f(vec3f(p.location) + xyz, 1.);
}

@fragment
fn frag_main() -> @location(0) vec4f {
    return vec4f(1., 1., 1., 0.25);
}
