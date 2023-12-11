use std::ops::Range;

use glam::IVec3;

use crate::{
    asset::Pack,
    chunk::{self, Chunk},
    types::{SideMap, SIDES, Direction},
    video::render::{self, QuadRef},
};

#[rustfmt::skip]
const TRAVERSAL_ORDER: SideMap<(IVec3, IVec3, IVec3)> = SideMap {
    west:  (IVec3::X, IVec3::Y, IVec3::Z),
    east:  (IVec3::X, IVec3::Z, IVec3::Y),
    south: (IVec3::Y, IVec3::Z, IVec3::X),
    north: (IVec3::Y, IVec3::X, IVec3::Z),
    down:  (IVec3::Z, IVec3::X, IVec3::Y),
    up:    (IVec3::Z, IVec3::Y, IVec3::X),
    none:  (IVec3::Z, IVec3::Y, IVec3::X),
};

#[derive(Default)]
pub struct Mesh<'q> {
    pub quads: &'q [QuadRef],
    pub ranges: SideMap<Range<u32>>,
}

pub struct Mesher {
    solid_quads: Vec<QuadRef>,
    alpha_quads: Vec<QuadRef>,
}

impl Mesher {
    pub fn new() -> Self {
        Self {
            solid_quads: vec![],
            alpha_quads: vec![],
        }
    }

    pub fn mesh(
        &mut self,
        pack: &Pack,
        chunks: &SideMap<Option<&Chunk>>,
    ) -> (Mesh<'_>, Mesh<'_>) {
        // Discard unloaded chunks
        let Some(chunk) = chunks.none else {
            return Default::default();
        };

        // Discard empty chunks immediately
        if chunk.num_blocks() == 0 {
            return Default::default();
        }

        // Discard enclosed chunks immediately
        if chunks.all(|entry| entry.map(Chunk::num_blocks) == Some(32_768)) {
            return Default::default();
        }

        self.solid_quads.clear();
        self.alpha_quads.clear();

        let models = [
            None,
            pack.model("grass"),
            pack.model("dirt"),
            pack.model("stone"),
            pack.model("wheat_0"),
            pack.model("wheat"),
            pack.model("water"),
            pack.model("water_surface"),
            pack.model("glass"),
            pack.model("sand"),
            pack.model("wood"),
            pack.model("leaves"),
        ];

        let mut solid_ranges = SideMap::<Range<_>>::default();
        let mut alpha_ranges = SideMap::<Range<_>>::default();

        for side in SIDES {
            solid_ranges[side].start = self.solid_quads.len() as u32;
            alpha_ranges[side].start = self.alpha_quads.len() as u32;
            let (Δlayer, Δrow, Δblock) = TRAVERSAL_ORDER[side];

            for a in 0..32 {
                let base_solid = self.solid_quads.len() as u32;
                let base_alpha = self.alpha_quads.len() as u32;

                for b in 0..32 {
                    let mut last_face = None;

                    for c in 0..32 {
                        // Traverse chunk following the current side's canonical order
                        let block_loc = a * Δlayer + b * Δrow + c * Δblock;
                        let block = chunk[block_loc];

                        // Skip invisible blocks
                        let Some(model) = models[block as usize] else {
                            last_face = None;
                            continue;
                        };

                        // Skip empty faces
                        let Some(range) = model.ranges[side].as_ref() else {
                            last_face = None;
                            continue;
                        };

                        let translucent = (6..=7).contains(&block);

                        // jmi2k: ugly...
                        let neighbor = side.map_or(0, |direction| {
                            let neighbor_loc = block_loc + IVec3::from(direction);
                            let block_loc = chunk::mask_block_loc(neighbor_loc);

                            match (neighbor_loc.min_element(), neighbor_loc.max_element()) {
                                (-1, _) | (_, 32) => {
                                    chunks[side].map(|chunk| chunk[block_loc]).unwrap_or(0)
                                }
                                _ => chunk[block_loc],
                            }
                        });

                        let culled = side.is_some()
                            && ((1..=3).contains(&neighbor)
                                || ((6..=7).contains(&block) && (6..=7).contains(&neighbor))
                                || (9..=11).contains(&neighbor));

                        // Skip hidden faces
                        if culled {
                            last_face = None;
                            continue;
                        }

                        for idx in range.clone() {
                            let sky_exposure = 15;
                            let mesh = if translucent {
                                &mut self.alpha_quads
                            } else {
                                &mut self.solid_quads
                            };

                            // Simple inline 1D greedy meshing.
                            //
                            // This code will fail to optimize sides with more than 1 quad,
                            // but this is an acceptable limitation
                            // as those should not be optimized anyway.
                            //
                            // The face extension is done at the reference level.
                            // This is possible because faces are canonicalized at pack load time.
                            if last_face == Some((idx, sky_exposure)) {
                                let old_quad_ref = mesh.last_mut().unwrap();
                                render::extend_quad_ref_w(old_quad_ref);
                            } else {
                                let quad_ref =
                                    render::quad_ref(idx, block_loc, sky_exposure);
                                mesh.push(quad_ref);
                                last_face = Some((idx, sky_exposure));
                            }
                        }
                    }
                }

                // Skip 2D greedy meshing for inner quads
                let Some(direction) = side else { continue; };

                // Merge set to get 2D greedy meshing
                //
                // The face extension is done at the reference level.
                // This is possible because faces are canonicalized at pack load time.
                for (base, mesh) in [(base_solid, &mut self.solid_quads), (base_alpha, &mut self.alpha_quads)] {
                    let mut dest = base as usize;
                    let mut back = base as usize;
                    let mut lead = base as usize;

                    let xo = match direction {
                        Direction::West => 42,
                        Direction::East => 37,
                        Direction::South => 32,
                        Direction::North => 42,
                        Direction::Down => 37,
                        Direction::Up => 32,
                    };

                    let yo = match direction {
                        Direction::West => 37,
                        Direction::East => 42,
                        Direction::South => 42,
                        Direction::North => 32,
                        Direction::Down => 32,
                        Direction::Up => 37,
                    };

                    let xm = 0x1F << xo;

                    while back < mesh.len() {
                        if lead == mesh.len() {
                            *unsafe { mesh.get_unchecked_mut(dest) } = *unsafe { mesh.get_unchecked(back) };
                            dest += 1;
                            back += 1;
                            continue;
                        }

                        let b = *unsafe { mesh.get_unchecked(back) };
                        let l = *unsafe { mesh.get_unchecked(lead) };

                        let bcwx0 = b & (0x00F8_0000_FFFF_FFFF | xm);
                        let bh = (b >> 56) & 0x1F;
                        let bx0 = (b >> xo) & 0x1F;
                        let by0 = (b >> yo) & 0x1F;

                        let lcwx0 = l & (0x00F8_0000_FFFF_FFFF | xm);
                        let lx0 = (l >> xo) & 0x1F;
                        let ly0 = (l >> yo) & 0x1F;

                        let Δy = ly0 - by0 - bh;

                        if Δy == 0 {
                            lead += 1;
                        } else if Δy > 1 {
                            *unsafe { mesh.get_unchecked_mut(dest) } = b;
                            dest += 1;
                            back += 1;
                        } else if bx0 > lx0 {
                            lead += 1;
                        } else if bcwx0 == lcwx0 {
                            *unsafe { mesh.get_unchecked_mut(lead) } = b;
                            render::extend_quad_ref_h(unsafe { mesh.get_unchecked_mut(lead) });
                            back += 1;
                            lead += 1;
                        } else {
                            *unsafe { mesh.get_unchecked_mut(dest) } = b;
                            dest += 1;
                            back += 1;
                        }
                    }

                    mesh.truncate(dest);
                }
            }

            solid_ranges[side].end = self.solid_quads.len() as u32;
            alpha_ranges[side].end = self.alpha_quads.len() as u32;
        }

        let solid_mesh = Mesh {
            quads: &self.solid_quads,
            ranges: solid_ranges,
        };

        let alpha_mesh = Mesh {
            quads: &self.alpha_quads,
            ranges: alpha_ranges,
        };

        (solid_mesh, alpha_mesh)
    }
}
