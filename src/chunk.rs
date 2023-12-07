use std::{
    ops::Index,
    sync::atomic::{AtomicU64, Ordering}, num::NonZeroU64,
};

use glam::{IVec3, IVec2};

use crate::types::{Cube, SideMap, Layer};

static NONCE: AtomicU64 = AtomicU64::new(1);

pub struct Chunk {
    nonces: SideMap<NonZeroU64>,
    num_blocks: u16,
    heights: Option<Box<Layer<i8, 32>>>,
    indices: Option<Box<Cube<i16, 32>>>,
}

impl Chunk {
    fn block_mut(&mut self, index: IVec3) -> &mut i16 {
        let IVec3 { x, y, z } = mask_block_loc(index);

        unsafe {
            let indices = self.indices.get_or_insert_with(|| Box::new_zeroed().assume_init());

            // Location is already masked into range
            indices
                .get_unchecked_mut(z as usize)
                .get_unchecked_mut(y as usize)
                .get_unchecked_mut(x as usize)
        }
    }

    fn height_mut(&mut self, index: IVec2) -> &mut i8 {
        // jmi2k: hack
        let IVec3 { x, y, .. } = mask_block_loc(index.extend(0));

        unsafe {
            let heights = self.heights.get_or_insert_with(|| Box::new_zeroed().assume_init());

            // Location is already masked into range
            heights
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

    fn alloc_indices(&mut self) {
        let indices = unsafe {
            // 0 is always a valid block ID, hardcoded as "air"
            Box::new_zeroed().assume_init()
        };

        self.indices = Some(indices);
    }

    pub fn new() -> Self {
        Self {
            nonces: SideMap::from_fn(|_| fresh_nonce()),
            num_blocks: 0,
            heights: None,
            indices: None,
        }
    }

    pub fn nonces(&self) -> &SideMap<NonZeroU64> {
        &self.nonces
    }

    pub fn num_blocks(&self) -> u16 {
        self.num_blocks
    }

    pub fn height(&self, index: IVec2) -> i8 {
        // jmi2k: hack
        let IVec3 { x, y, .. } = mask_block_loc(index.extend(0));
        let Some(ref heights) = self.heights else { return 0; };

        unsafe {
            // Location is already masked into range
            *heights
                .get_unchecked(y as usize)
                .get_unchecked(x as usize)
        }
    }

    pub fn place(&mut self, location: IVec3, block: i16) {
        let IVec3 { x, y, z } = location;
        let height_loc = IVec2 { x, y };
        let last_block = self[location];
        *self.block_mut(location) = block;
        self.update_nonces(location);

        // jmi2k: be more specific with he kind of blocks influencing these fields
        if block != 0 && self.height(height_loc) <= z as i8 {
            *self.height_mut(height_loc) = 1 + z as i8;
        }

        self.num_blocks += (last_block != 0) as u16 - (block != 0) as u16;
    }

    pub fn destroy(&mut self, location: IVec3) {
        let IVec3 { x, y, z } = location;
        let height_loc = IVec2 { x, y };
        let last_block = self[location];
        *self.block_mut(location) = 0;
        self.update_nonces(location);

        // jmi2k: be more specific with he kind of blocks influencing these fields
        if self.height(height_loc) == z as i8 {
            *self.height_mut(height_loc) = (1..=z)
                .rev()
                .find(|h| self[IVec3::new(x, y, *h - 1)] != 0)
                .unwrap_or_default() as i8;
        }

        self.num_blocks -= (last_block != 0) as u16;
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
        let Some(ref indices) = self.indices else { return &0; };

        unsafe {
            // Location is already masked into range
            indices
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

fn fresh_nonce() -> NonZeroU64 {
    let value = NONCE.fetch_add(1, Ordering::Relaxed);

    unsafe {
        // This is actually unsafe!
        // It will cause undefined behavior when NONCE rolls over.
        // If a value were to be generated every nanosecond,
        // it would take ~584 years to roll the counter over.
        // Let's just embrace the danger...
        NonZeroU64::new_unchecked(value)
    }
}
