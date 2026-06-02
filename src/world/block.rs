use crate::models::{
    block_model::BlockModel,
    builtin::{cube, slab, water},
    face::FaceDir,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BlockId(pub u16);

pub const AIR:        BlockId = BlockId(0);
pub const DIRT:       BlockId = BlockId(1);
pub const STONE:      BlockId = BlockId(2);
pub const BRICK_SLAB: BlockId = BlockId(3);
pub const WATER:      BlockId = BlockId(4);

// Texture layer indices — must match LAYER_FILES order in render/texture.rs
pub const TEX_STONE:      u32 = 0;
pub const TEX_DIRT:       u32 = 1;
pub const TEX_GRASS_TOP:  u32 = 2;
pub const TEX_GRASS_SIDE: u32 = 3;

/// Returns the texture array layer index for a given block face.
pub fn face_tex(id: BlockId, dir: FaceDir) -> u32 {
    match id {
        DIRT => match dir {
            FaceDir::PosY => TEX_GRASS_TOP,
            FaceDir::NegY => TEX_DIRT,
            _             => TEX_GRASS_SIDE,
        },
        _ => TEX_STONE,
    }
}

pub fn get_model(id: BlockId) -> Option<BlockModel> {
    match id {
        DIRT => Some(cube()),
        STONE => Some(cube()),
        BRICK_SLAB => Some(slab()),
        WATER => Some(water()),
        AIR => None,
        _ => None,
    }
}