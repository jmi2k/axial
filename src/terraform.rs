use std::{
    array,
    f64::consts::SQRT_2,
    sync::mpsc::{Receiver, Sender, self},
    thread,
};

use glam::IVec3;
use noise::{NoiseFn, Perlin};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::{chunk::{self, Chunk}, types::Layer};

const NUM_THREADS: usize = 4;

pub struct Terraformer {
    location_tx: [Sender<IVec3>; NUM_THREADS],
    chunk_rx: Receiver<(IVec3, Chunk, Box<Layer<i32, 32>>)>,
}

impl Terraformer {
    pub fn new(seed: u64) -> Self {
        let (chunk_tx, chunk_rx) = mpsc::channel();
        let location_tx = array::from_fn(|_| spawn(seed, chunk_tx.clone()));

        Self { location_tx, chunk_rx }
    }

    pub fn chunk_rx(&self) -> &Receiver<(IVec3, Chunk, Box<Layer<i32, 32>>)> {
        &self.chunk_rx
    }

    pub fn terraform(&self, location: IVec3) {
        let key = location.x ^ location.y ^ location.z;
        let _ = self.location_tx[key as usize % NUM_THREADS].send(location);
    }
}

fn spawn(seed: u64, chunk_tx: Sender<(IVec3, Chunk, Box<Layer<i32, 32>>)>) -> Sender<IVec3> {
    let mut rng = StdRng::seed_from_u64(seed);
    let perlins = array::from_fn::<_, 3, _>(|_| Perlin::new(rng.next_u32()));
    let (location_tx, location_rx) = mpsc::channel();

    thread::spawn(move || for location in location_rx { 
        let (chunk, height_map) = terraform(location, &perlins);
        let _ = chunk_tx.send((location, chunk, height_map));
    });

    location_tx
}

fn terraform(location: IVec3, perlins: &[Perlin]) -> (Chunk, Box<Layer<i32, 32>>) {
    let mut chunk = Chunk::new();
    let mut light_map = Box::<Layer<_, 32>>::default();
    let mut overflow = Chunk::new();

    let air = 0;
    let grass = 1;
    let dirt = 2;
    let stone = 3;
    let wheat_0 = 4;
    let wheat = 5;
    let water = 6;
    let water_surface = 7;
    let glass = 8;
    let sand = 9;

    #[rustfmt::skip]
    for y in 0..32 {
    for x in 0..32 {
        let mut accumulator = 0;

        for z in (0..32 + 5).rev() {
            let block_loc = IVec3 { x, y, z };
            let chunk = if z < 32 { &mut chunk } else { &mut overflow };
            let shift = if z < 32 { IVec3::ZERO } else { IVec3::Z };
            let IVec3 { x, y, z } = chunk::merge_loc(location + shift, block_loc);
            let factor = SQRT_2 / 100.;

            if z < -64 {
                continue;
            }

            let terrain_noises = array::from_fn::<_, 2, _>(|idx| {
                let scale = 10f64.powi(idx as _);
                let damp = 0.45 / scale;
                let x = x as f64 * factor * scale;
                let y = y as f64 * factor * scale;
                let z = z as f64 * factor * scale;
                perlins[idx].get([x, y, z]) * damp
            });

            let cave_sample = {
                let scale = 10f64;
                let x = x as f64 * factor * scale;
                let y = y as f64 * factor * scale;
                let z = z as f64 * factor * scale;
                perlins[2].get([x, y, z])
            };

            let terrain_sample = 0.5 + terrain_noises.into_iter().sum::<f64>();

            if terrain_sample * z as f64 > 32. || cave_sample > 0.5 {
                let block = match z {
                    16.. => air,
                    15 => water_surface,
                    _ => water,
                };

                chunk.place(block_loc, block);
                accumulator = 0;
                continue;
            }

            let block = match (z, accumulator) {
                (48.., 0) => /*wheat_0*/air,
                (47.., 1) => grass,
                (46.., 2..=4) => dirt,

                (47, 0) => sand,
                (..=46, 0) => water,
                (..=46, 1..=4) => dirt,

                _ => stone,
            };

            chunk.place(block_loc, block);

            if block != air && shift == IVec3::ZERO && light_map[block_loc.y as usize][block_loc.x as usize] < z {
                light_map[block_loc.y as usize][block_loc.x as usize] = z;
            }

            accumulator += 1;
        }
    }
    }

    (chunk, light_map)
}

