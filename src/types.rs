use std::{
    mem,
    ops::{Index, IndexMut, Neg},
};

use glam::IVec3;
use serde::Deserialize;

pub const SIDES: [Side; mem::variant_count::<Direction>() + 1] = [
    Some(Direction::West),
    Some(Direction::East),
    Some(Direction::South),
    Some(Direction::North),
    Some(Direction::Down),
    Some(Direction::Up),
    None,
];

pub type Side = Option<Direction>;
pub type Layer<T, const N: usize> = [[T; N]; N];
pub type Cube<T, const N: usize> = [Layer<T, N>; N];

#[repr(u8)]
#[derive(Copy, Clone)]
#[derive(Debug)]
pub enum Direction {
    West,
    East,
    South,
    North,
    Down,
    Up,
}

impl From<Direction> for IVec3 {
    fn from(value: Direction) -> Self {
        match value {
            Direction::West => IVec3::NEG_X,
            Direction::East => IVec3::X,
            Direction::South => IVec3::NEG_Y,
            Direction::North => IVec3::Y,
            Direction::Down => IVec3::NEG_Z,
            Direction::Up => IVec3::Z,
        }
    }
}

impl Neg for Direction {
    type Output = Direction;

    fn neg(self) -> Self::Output {
        match self {
            Direction::West => Direction::East,
            Direction::East => Direction::West,
            Direction::South => Direction::North,
            Direction::North => Direction::South,
            Direction::Down => Direction::Up,
            Direction::Up => Direction::Down,
        }
    }
}

#[derive(Default, Deserialize)]
pub struct DirMap<T> {
    pub west: T,
    pub east: T,
    pub south: T,
    pub north: T,
    pub down: T,
    pub up: T,
}

impl From<&DirMap<bool>> for IVec3 {
    fn from(d: &DirMap<bool>) -> Self {
        IVec3 {
            x: d.east as i32 - d.west as i32,
            y: d.north as i32 - d.south as i32,
            z: d.up as i32 - d.down as i32,
        }
    }
}

impl<T> Index<Direction> for DirMap<T> {
    type Output = T;

    fn index(&self, index: Direction) -> &Self::Output {
        match index {
            Direction::West => &self.west,
            Direction::East => &self.east,
            Direction::South => &self.south,
            Direction::North => &self.north,
            Direction::Down => &self.down,
            Direction::Up => &self.up,
        }
    }
}

impl<T> IndexMut<Direction> for DirMap<T> {
    fn index_mut(&mut self, index: Direction) -> &mut Self::Output {
        match index {
            Direction::West => &mut self.west,
            Direction::East => &mut self.east,
            Direction::South => &mut self.south,
            Direction::North => &mut self.north,
            Direction::Down => &mut self.down,
            Direction::Up => &mut self.up,
        }
    }
}

#[derive(Clone, PartialEq, Default)]
pub struct SideMap<T> {
    pub west: T,
    pub east: T,
    pub south: T,
    pub north: T,
    pub down: T,
    pub up: T,
    pub none: T,
}

impl<T> SideMap<T> {
    pub fn from_fn(mut cb: impl FnMut(Side) -> T) -> Self {
        SideMap {
            west: cb(Some(Direction::West)),
            east: cb(Some(Direction::East)),
            south: cb(Some(Direction::South)),
            north: cb(Some(Direction::North)),
            down: cb(Some(Direction::Down)),
            up: cb(Some(Direction::Up)),
            none: cb(None),
        }
    }
}

// jmi2k: ugly...
impl<T, U> Extend<(Side, T)> for SideMap<U>
where
    U: Extend<T>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (Side, T)>,
    {
        for (side, value) in iter {
            self[side].extend_one(value);
        }
    }
}

impl<T> Index<Side> for SideMap<T> {
    type Output = T;

    fn index(&self, index: Side) -> &Self::Output {
        match index {
            Some(Direction::West) => &self.west,
            Some(Direction::East) => &self.east,
            Some(Direction::South) => &self.south,
            Some(Direction::North) => &self.north,
            Some(Direction::Down) => &self.down,
            Some(Direction::Up) => &self.up,
            None => &self.none,
        }
    }
}

impl<T> IndexMut<Side> for SideMap<T> {
    fn index_mut(&mut self, index: Side) -> &mut Self::Output {
        match index {
            Some(Direction::West) => &mut self.west,
            Some(Direction::East) => &mut self.east,
            Some(Direction::South) => &mut self.south,
            Some(Direction::North) => &mut self.north,
            Some(Direction::Down) => &mut self.down,
            Some(Direction::Up) => &mut self.up,
            None => &mut self.none,
        }
    }
}
