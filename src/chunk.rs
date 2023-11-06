use std::{
    ops::Index,
    sync::atomic::{AtomicU64, Ordering},
};

use glam::IVec3;

use crate::types::{Cube, SideMap};

static NONCE: AtomicU64 = AtomicU64::new(0);

pub struct Chunk {
    nonces: SideMap<u64>,
    indices: Box<Cube<i16, 32>>,
}

impl Chunk {
    fn block_mut(&mut self, index: IVec3) -> &mut i16 {
        let IVec3 { x, y, z } = mask_block_loc(index);

        unsafe {
            // Location is already masked into range
            self.indices
                .get_unchecked_mut(z as usize)
                .get_unchecked_mut(y as usize)
                .get_unchecked_mut(x as usize)
        }
    }

    fn update_nonces(&mut self, location: IVec3) {
        self.nonces.none = fresh_nonce();

        match location.x {
            0 => self.nonces.west = fresh_nonce(),
            31 => self.nonces.east = fresh_nonce(),
            _ => {}
        }

        match location.y {
            0 => self.nonces.south = fresh_nonce(),
            31 => self.nonces.north = fresh_nonce(),
            _ => {}
        }

        match location.z {
            0 => self.nonces.down = fresh_nonce(),
            31 => self.nonces.up = fresh_nonce(),
            _ => {}
        }
    }

    pub fn nonces(&self) -> &SideMap<u64> {
        &self.nonces
    }

    pub fn new() -> Self {
        let indices = unsafe {
            // 0 is always a valid block ID, hardcoded as "air"
            Box::new_zeroed().assume_init()
        };

        Self {
            nonces: SideMap::from_fn(|_| fresh_nonce()),
            indices,
        }
    }

    pub fn place(&mut self, location: IVec3, block: i16) {
        *self.block_mut(location) = block;
        self.update_nonces(location);
    }

    pub fn destroy(&mut self, location: IVec3) {
        *self.block_mut(location) = 0;
        self.update_nonces(location);
    }

    // jmi2k: should it have access to the world? I think so
    pub fn tick(&mut self, tick: u64) {
        let location = mask_block_loc(rand::random());

        if self[location] != 4 {
            return;
        }

        self.place(location, 5);
    }
}

impl Index<IVec3> for Chunk {
    type Output = i16;

    fn index(&self, index: IVec3) -> &Self::Output {
        let IVec3 { x, y, z } = mask_block_loc(index);

        unsafe {
            // Location is already masked into range
            self.indices
                .get_unchecked(z as usize)
                .get_unchecked(y as usize)
                .get_unchecked(x as usize)
        }
    }
}

pub fn mask_block_loc(location: IVec3) -> IVec3 {
    debug_assert!(
        location.min_element() >= -32 && location.max_element() <= 31,
        "block location out of chunk bounds",
    );

    location & 31
}

pub fn mask_chunk_loc(location: IVec3) -> IVec3 {
    debug_assert!(
        location.min_element() >= i32::MIN / 32 && location.max_element() <= i32::MAX / 32,
        "chunk location out of world bounds",
    );

    location << 5 >> 5
}

pub fn split_loc(location: IVec3) -> (IVec3, IVec3) {
    let chunk_loc = location >> 5;
    let block_loc = location & 31;

    (chunk_loc, block_loc)
}

pub fn merge_loc(chunk_loc: IVec3, block_loc: IVec3) -> IVec3 {
    let chunk_loc = mask_chunk_loc(chunk_loc);
    let block_loc = mask_block_loc(block_loc);

    chunk_loc << 5 | block_loc
}

fn fresh_nonce() -> u64 {
    NONCE.fetch_add(1, Ordering::Relaxed)
}
