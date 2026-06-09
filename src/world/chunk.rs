use glam::IVec3;

use crate::world::block::BlockId;

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_HEIGHT: usize = 256;

/// Flat array storage: index = x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
#[derive(Clone)]
pub struct Chunk {
    pub origin: IVec3,
    pub dirty: bool,
    /// Highest Y level that contains any non-air block. Used to skip empty height in meshing.
    pub max_height: usize,
    blocks: Box<[BlockId; CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT]>,
}

impl Chunk {
    pub fn new(origin: IVec3) -> Self {
        Self {
            origin,
            dirty: true,
            max_height: 0,
            blocks: Box::new([BlockId::default(); CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT]),
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.blocks[x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, id: BlockId) {
        self.blocks[x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE] = id;
        self.dirty = true;
        if y > self.max_height {
            self.max_height = y;
        }
    }
}
