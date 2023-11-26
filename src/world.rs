use std::{
    collections::{HashMap, hash_map::Entry},
    sync::atomic::{AtomicU64, Ordering},
};

use glam::IVec3;

use crate::{chunk::{self, Chunk}, types::Layer};

pub struct World {
    next_tick: AtomicU64,
    loaded_chunks: HashMap<IVec3, Chunk, ahash::RandomState>,
}

impl World {
    pub fn new() -> Self {
        Self {
            next_tick: AtomicU64::default(),
            loaded_chunks: HashMap::default(),
        }
    }

    pub fn chunk(&self, location: IVec3) -> Option<&Chunk> {
        self.loaded_chunks.get(&location)
    }

    pub fn chunk_mut(&mut self, location: IVec3) -> Option<&mut Chunk> {
        self.loaded_chunks.get_mut(&location)
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

    pub fn place(&mut self, location: IVec3, block: i16) {
        let (chunk_loc, block_loc) = chunk::split_loc(location);

        if let Some(chunk) = self.chunk_mut(chunk_loc) {
            chunk.place(block_loc, block);
        }
    }

    pub fn destroy(&mut self, location: IVec3) {
        let (chunk_loc, block_loc) = chunk::split_loc(location);

        if let Some(chunk) = self.chunk_mut(chunk_loc) {
            chunk.destroy(block_loc);
        }
    }

    pub fn block(&self, location: IVec3) -> Option<i16> {
        let (chunk_loc, block_loc) = chunk::split_loc(location);
        let chunk = self.chunk(chunk_loc)?;
        Some(chunk[block_loc])
    }
}
