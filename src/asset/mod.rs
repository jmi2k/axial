mod raw;

use std::{
    array,
    collections::BTreeMap,
    fs,
    ops::Range,
    path::{Path, PathBuf},
};

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use image::{
    imageops,
    RgbaImage,
};

use crate::types::{DirMap, SideMap, SIDES};

pub const TILE_LENGTH: u32 = 16;
pub const MIP_LEVELS: usize = 1 + TILE_LENGTH.ilog2() as usize;

pub type Quad = [Vertex; 4];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub mapping: Vec2,
    pub normal: Vec3,
    pub tile: u32,
}

impl Vertex {
    pub fn new(position: Vec3, mapping: Vec2, normal: Vec3, tile: u32) -> Self {
        Self {
            position,
            mapping,
            normal,
            tile,
        }
    }
}

pub struct Model {
    pub hides: DirMap<bool>,
    pub ranges: SideMap<Option<Range<usize>>>,
}

pub struct Pack {
    tiles: Vec<[RgbaImage; MIP_LEVELS]>,
    vertex_atlas: Vec<Quad>,
    models: BTreeMap<String, Model>,
}

impl Pack {
    pub fn tiles(&self) -> &[[RgbaImage; MIP_LEVELS]] {
        &self.tiles
    }

    pub fn vertex_atlas(&self) -> &[Quad] {
        &self.vertex_atlas
    }

    pub fn model(&self, name: &str) -> Option<&Model> {
        self.models.get(name)
    }
}

pub fn open(path: impl AsRef<Path>) -> Pack {
    let path = path.as_ref();

    let (tiles, tile_names) = open_tiles(path);
    let (vertex_atlas, models) = open_models(path, &tile_names);

    Pack {
        tiles,
        vertex_atlas,
        models,
    }
}

fn open_tiles(root: &Path) -> (Vec<[RgbaImage; MIP_LEVELS]>, Vec<PathBuf>) {
    let root = root.join("tiles");

    let mut names = fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|entry| PathBuf::from(entry.file_name()))
        .collect::<Vec<_>>();

    let mut tiles = vec![];
    names.sort_unstable();

    for (idx, name) in names.iter_mut().enumerate() {
        let path = root.join(&name);
        name.set_extension("");

        let image = image::open(path).unwrap().into_rgba8();
        let width = image.width();
        let height = image.height();
        let tile = array::from_fn(|idx| imageops::thumbnail(&image, width >> idx, height >> idx));
        tiles.push(tile);
    }

    (tiles, names)
}

fn open_models(root: &Path, tile_names: &[PathBuf]) -> (Vec<Quad>, BTreeMap<String, Model>) {
    let root = root.join("models");
    let mut vertices = vec![];
    let mut models = BTreeMap::default();
    let mut mesh = SideMap::<Vec<_>>::default();

    for entry in fs::read_dir(&root).unwrap() {
        let entry = entry.unwrap();
        let mut name = PathBuf::from(entry.file_name());
        let path = root.join(&name);
        name.set_extension("");

        let src = fs::read_to_string(path).unwrap();
        let raw::Model { hides, parts } = ron::from_str(&src).unwrap();

        parts
            .into_iter()
            .flat_map(|part| part.desugar(tile_names))
            .collect_into(&mut mesh);

        let mut ranges = SideMap::default();

        for side in SIDES {
            if mesh[side].is_empty() {
                continue;
            }

            let start = vertices.len();
            let end = start + mesh[side].len();
            ranges[side] = Some(start..end);
            vertices.append(&mut mesh[side]);
        }

        let name = name.to_string_lossy().to_string();
        let model = Model { hides, ranges };
        models.insert(name, model);
    }

    (vertices, models)
}
