use std::{
    collections::{HashMap, hash_map::Entry},
    sync::atomic::{AtomicU64, Ordering},
};

use glam::IVec3;

use crate::chunk::{self, Chunk};

pub struct World {
    next_tick: AtomicU64,
    loaded_chunks: HashMap<IVec3, Chunk>,
}

impl World {
    pub fn new() -> Self {
        Self {
            next_tick: AtomicU64::default(),
            loaded_chunks: HashMap::with_capacity(16 * 16 * 16),
        }
    }

    pub fn chunk(&self, location: IVec3) -> Option<&Chunk> {
        self.loaded_chunks.get(&location)
    }

    pub fn chunk_entry(&mut self, location: IVec3) -> Entry<IVec3, Chunk> {
        self.loaded_chunks.entry(location)
    }

    pub fn load(&mut self, location: IVec3, chunk: Chunk) {
        let location = chunk::mask_chunk_loc(location);
        self.loaded_chunks.insert(location, chunk);
    }

    pub fn unload(&mut self, location: IVec3) {
        let location = chunk::mask_chunk_loc(location);
        self.loaded_chunks.remove(&location);
    }

    pub fn tick(&mut self) -> u64 {
        let tick = self.next_tick.fetch_add(1, Ordering::Relaxed);

        for chunk in self.loaded_chunks.values_mut() {
            chunk.tick(tick);
        }

        tick
    }
}
