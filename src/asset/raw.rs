use std::path::{Path, PathBuf};

use arrayvec::ArrayVec;
use glam::{Mat3, Vec2, Vec3};
use serde::Deserialize;

use crate::types::{DirMap, Direction, Side};

use super::{Quad, Vertex};

#[derive(Deserialize)]
pub(super) struct Mapping<'src> {
    #[serde(borrow)]
    pub tile: &'src Path,

    #[serde(default)]
    pub p: Vec2,

    #[serde(default = "vec2_x")]
    pub u: Vec2,

    #[serde(default = "vec2_y")]
    pub v: Vec2,
}

#[allow(clippy::large_enum_variant)]
#[derive(Deserialize)]
pub(super) enum Part<'src> {
    Cuboid {
        #[serde(default)]
        o: Vec3,

        #[serde(default = "vec3_x")]
        x: Vec3,

        #[serde(default = "vec3_y")]
        y: Vec3,

        #[serde(default = "vec3_z")]
        z: Vec3,

        #[serde(borrow)]
        #[serde(flatten)]
        mappings: DirMap<Mapping<'src>>,
    },

    Rect {
        o: Vec3,
        x: Vec3,
        y: Vec3,

        #[serde(flatten)]
        mapping: Mapping<'src>,
    },
}

impl Part<'_> {
    pub fn desugar(self, tile_names: &[PathBuf]) -> ArrayVec<(Side, Quad), 6> {
        match self {
            Part::Cuboid {
                o,
                x,
                y,
                z,
                mappings,
            } => {
                let ___ = o;
                let x__ = x;
                let _y_ = y;
                let xy_ = x + y - o;
                let __z = z;
                let x_z = x + z - o;
                let _yz = y + z - o;
                let xyz = x + y + z - o - o;

                let quads = [
                    desugar_rect(_y_, ___, _yz, mappings.west, tile_names),
                    desugar_rect(x__, xy_, x_z, mappings.east, tile_names),
                    desugar_rect(___, x__, __z, mappings.south, tile_names),
                    desugar_rect(xy_, _y_, xyz, mappings.north, tile_names),
                    desugar_rect(_y_, xy_, ___, mappings.down, tile_names),
                    desugar_rect(__z, x_z, _yz, mappings.up, tile_names),
                ];

                ArrayVec::from_iter(quads)
            }

            Part::Rect { o, x, y, mapping } => {
                let quad = [desugar_rect(o, x, y, mapping, tile_names)];
                ArrayVec::from_iter(quad)
            }
        }
    }
}

#[derive(Deserialize)]
pub(super) struct Model<'src> {
    pub hides: DirMap<bool>,

    #[serde(borrow)]
    pub parts: Vec<Part<'src>>,
}

fn vec2_x() -> Vec2 {
    Vec2::X
}

fn vec2_y() -> Vec2 {
    Vec2::Y
}

fn vec3_x() -> Vec3 {
    Vec3::X
}

fn vec3_y() -> Vec3 {
    Vec3::Y
}

fn vec3_z() -> Vec3 {
    Vec3::Z
}

fn desugar_rect(
    o: Vec3,
    x: Vec3,
    y: Vec3,
    mapping: Mapping<'_>,
    tile_names: &[PathBuf],
) -> (Side, Quad) {
    let Mapping { tile, p, u, v } = mapping;
    let normal = Vec3::cross(x - o, y - o);
    let coords = Mat3::from_cols(o, x, y).transpose();

    let cull = match coords.to_cols_array_2d() {
        [[0., 0., 0.], _, _] => Some(Direction::West),
        [[1., 1., 1.], _, _] => Some(Direction::East),
        [_, [0., 0., 0.], _] => Some(Direction::South),
        [_, [1., 1., 1.], _] => Some(Direction::North),
        [_, _, [0., 0., 0.]] => Some(Direction::Down),
        [_, _, [1., 1., 1.]] => Some(Direction::Up),
        _ => None,
    };

    let oo = o;
    let x_ = x;
    let _y = y;
    let xy = x + y - o;

    let pp = p;
    let u_ = u;
    let _v = v;
    let uv = u + v - p;

    let idx = tile_names
        .binary_search_by_key(&tile, PathBuf::as_path)
        .unwrap() as u32;

    let vertices = [
        Vertex::new(oo, _v, normal, idx),
        Vertex::new(x_, uv, normal, idx),
        Vertex::new(_y, pp, normal, idx),
        Vertex::new(xy, u_, normal, idx),
    ];

    let quad = normalize_quad(vertices, normal);
    (cull, quad)
}

fn normalize_quad(mut vertices: Quad, normal: Vec3) -> Quad {
    // jmi2k: enforce canonical ordering in position
    // jmi2k: enforce canonical ordering in UV (signs)

    // hacks to make the solid cube work for testing purposes
    if normal.x != 0. && normal.y != 0. && normal.z != 0. {
        return vertices;
    }

    if normal.x < 0. || normal.y > 0. {
        let [a, b, c, d] = vertices;
        return [b, d, a, c];
    }

    if normal.z < 0. {
        let [a, b, c, d] = vertices;
        return [c, a, d, b];
    }

    match normal.to_array() {
        [x, 0., 0.] if x < 0. => vertices,
        [x, 0., 0.] if x > 0. => vertices,
        [0., y, 0.] if y < 0. => vertices,
        [0., y, 0.] if y > 0. => vertices,
        [0., 0., z] if z < 0. => vertices,
        [0., 0., z] if z > 0. => vertices,
        [_, _, _] => vertices,
    }
}
