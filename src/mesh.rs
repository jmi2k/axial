use glam::IVec3;

use crate::{
    asset::Pack,
    chunk::{self, Chunk},
    types::{SideMap, SIDES},
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

pub struct Mesher {
    solid_mesh: Vec<QuadRef>,
    alpha_mesh: Vec<QuadRef>,
}

impl Mesher {
    pub fn new() -> Self {
        Self {
            solid_mesh: vec![],
            alpha_mesh: vec![],
        }
    }

    pub fn mesh(
        &mut self,
        pack: &Pack,
        chunks: &SideMap<Option<&Chunk>>,
    ) -> (&[QuadRef], &[QuadRef]) {
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

        self.solid_mesh.clear();
        self.alpha_mesh.clear();

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

        for side in SIDES {
            let (Δlayer, Δrow, Δblock) = TRAVERSAL_ORDER[side];

            for a in 0..32 {
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
                            &mut self.alpha_mesh
                        } else {
                            &mut self.solid_mesh
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
                            render::extend_quad_ref(old_quad_ref, Δblock);
                        } else {
                            let quad_ref =
                                render::quad_ref(idx, block_loc, translucent, sky_exposure);
                            mesh.push(quad_ref);
                            last_face = Some((idx, sky_exposure));
                        }
                    }
                }
            }
            }
        }

        (&self.solid_mesh, &self.alpha_mesh)
    }
}
