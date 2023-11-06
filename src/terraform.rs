use std::{
    array,
    f64::consts::SQRT_2,
    sync::mpsc::{Receiver, Sender, self},
    thread,
};

use glam::IVec3;
use noise::{NoiseFn, Perlin};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::{
    chunk::{self, Chunk},
    types::Layer,
};

const NUM_THREADS: usize = 4;

pub struct Terraformer {
    location_tx: [Sender<IVec3>; NUM_THREADS],
    chunk_rx: Receiver<(IVec3, Chunk)>,
}

impl Terraformer {
    pub fn new(seed: u64) -> Self {
        let (chunk_tx, chunk_rx) = mpsc::channel();
        let location_tx = array::from_fn(|_| spawn(seed, chunk_tx.clone()));

        Self { location_tx, chunk_rx }
    }

    pub fn chunk_rx(&self) -> &Receiver<(IVec3, Chunk)> {
        &self.chunk_rx
    }

    pub fn terraform(&self, location: IVec3) {
        let key = location.x ^ location.y ^ location.z;
        let _ = self.location_tx[key as usize % NUM_THREADS].send(location);
    }
}

fn spawn(seed: u64, chunk_tx: Sender<(IVec3, Chunk)>) -> Sender<IVec3> {
    let mut rng = StdRng::seed_from_u64(seed);
    let perlins = array::from_fn::<_, 2, _>(|_| Perlin::new(rng.next_u32()));
    let (location_tx, location_rx) = mpsc::channel();

    thread::spawn(move || for location in location_rx { 
        let chunk = terraform(location, &perlins);
        let _ = chunk_tx.send((location, chunk));
    });

    location_tx
}

fn terraform(location: IVec3, perlins: &[Perlin]) -> Chunk {
    let mut chunk = Chunk::new();
    let mut underflow = Layer::<i16, 32>::default();
    let mut overflow = Layer::<i16, 32>::default();

    let id_air = 0;
    let id_grass = 1;
    let id_dirt = 2;
    let id_stone = 3;
    let id_wheat_0 = 4;

    let mut place = |location, block| {
        let IVec3 { x, y, z } = location;

        match z {
            32 => unsafe {
                *overflow
                    .get_unchecked_mut(y as usize)
                    .get_unchecked_mut(x as usize) = block;
            }

            _ => {
                chunk.place(location, block);
            }
        }
    };

    // Generate terrain shape
    #[rustfmt::skip]
    for z in 0..33 {
    for y in 0..32 {
    for x in 0..32 {
        let block_loc = IVec3 { x, y, z };

        let shift = match z {
            32 => IVec3::Z,
            _ => IVec3::ZERO,
        };

        let IVec3 { x, y, z } = chunk::merge_loc(location + shift, block_loc);
        let factor = SQRT_2 / 100.;

        let noises = array::from_fn::<_, 2, _>(|idx| {
            let scale = 10f64.powi(idx as _);
            let damp = 0.45 / scale;
            let x = x as f64 * factor * scale;
            let y = y as f64 * factor * scale;
            let z = z as f64 * factor * scale;
            perlins[idx].get([x, y, z]) * damp
        });

        let sample = 0.5 + noises.into_iter().sum::<f64>();

        if sample * z as f64 > 8. {
            place(block_loc, id_air);
            continue;
        }

        place(block_loc, id_stone);
    }
    }
    }

    // Create layers of dirt and grass
    #[rustfmt::skip]
    for y in 0..32 {
    for x in 0..32 {
        let mut accumulator = 0;

        let block_above = unsafe {
            *overflow
                .get_unchecked(y as usize)
                .get_unchecked(x as usize)
        };

        if block_above == id_stone {
            accumulator += 1;
        }

        for z in (0..32).rev() {
            let block_loc = IVec3 { x, y, z };
            let current_block = chunk[block_loc];

            if current_block != id_stone {
                accumulator = 0;
                continue;
            }

            let replacement = match accumulator {
                0 => id_grass,
                1..=3 => id_dirt,
                _ => continue,
            };

            chunk.place(block_loc, replacement);
            accumulator += 1;
        }
    }
    }

    chunk
}