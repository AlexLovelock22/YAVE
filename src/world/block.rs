use crate::models::{
    block_model::BlockModel,
    builtin::{cube, slab, water},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BlockId(pub u16);

pub const AIR:        BlockId = BlockId(0);
pub const DIRT:       BlockId = BlockId(1);
pub const STONE:      BlockId = BlockId(2);
pub const BRICK_SLAB: BlockId = BlockId(3);
pub const WATER:      BlockId = BlockId(4);

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