use crate::world::{
    block::{get_model, is_opaque},
    chunk::{Chunk, CHUNK_HEIGHT, CHUNK_SIZE},
};

/// One bit per (y, lateral) pair on a chunk face. lateral = z for X-faces, x for Z-faces.
/// Bit index = y * CHUNK_SIZE + lateral.
pub const FACE_BYTES: usize = CHUNK_HEIGHT * CHUNK_SIZE / 8; // 256*32/8 = 1024 bytes

/// Solid-block masks for all 4 vertical boundary faces of a chunk.
/// Stored per chunk and shared with neighbours so they can cull their border quads.
#[derive(Clone)]
pub struct ChunkFaceData {
    pub neg_x: [u8; FACE_BYTES], // is block (0, y, z) solid?
    pub pos_x: [u8; FACE_BYTES], // is block (CHUNK_SIZE-1, y, z) solid?
    pub neg_z: [u8; FACE_BYTES], // is block (x, y, 0) solid?
    pub pos_z: [u8; FACE_BYTES], // is block (x, y, CHUNK_SIZE-1) solid?
}

/// Per-direction neighbour masks passed into mesh_chunk.
/// None = neighbour not yet loaded → treat boundary as exposed.
#[derive(Clone)]
pub struct NeighborMasks {
    pub pos_x: Option<[u8; FACE_BYTES]>, // neg_x face of the +X neighbour
    pub neg_x: Option<[u8; FACE_BYTES]>, // pos_x face of the -X neighbour
    pub pos_z: Option<[u8; FACE_BYTES]>, // neg_z face of the +Z neighbour
    pub neg_z: Option<[u8; FACE_BYTES]>, // pos_z face of the -Z neighbour
}

impl ChunkFaceData {
    pub fn extract(chunk: &Chunk) -> Self {
        let mut neg_x = [0u8; FACE_BYTES];
        let mut pos_x = [0u8; FACE_BYTES];
        let mut neg_z = [0u8; FACE_BYTES];
        let mut pos_z = [0u8; FACE_BYTES];

        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_SIZE {
                let bit = y * CHUNK_SIZE + z;
                if is_opaque(chunk.get(0, y, z)) {
                    neg_x[bit / 8] |= 1 << (bit % 8);
                }
                if is_opaque(chunk.get(CHUNK_SIZE - 1, y, z)) {
                    pos_x[bit / 8] |= 1 << (bit % 8);
                }
            }
            for x in 0..CHUNK_SIZE {
                let bit = y * CHUNK_SIZE + x;
                if is_opaque(chunk.get(x, y, 0)) {
                    neg_z[bit / 8] |= 1 << (bit % 8);
                }
                if is_opaque(chunk.get(x, y, CHUNK_SIZE - 1)) {
                    pos_z[bit / 8] |= 1 << (bit % 8);
                }
            }
        }

        Self { neg_x, pos_x, neg_z, pos_z }
    }
}

/// Is the block at (y, lateral) solid according to `mask`?
/// lateral = z for X-direction faces, x for Z-direction faces.
#[inline]
pub fn mask_solid(mask: &[u8; FACE_BYTES], y: usize, lateral: usize) -> bool {
    let bit = y * CHUNK_SIZE + lateral;
    (mask[bit / 8] >> (bit % 8)) & 1 != 0
}
