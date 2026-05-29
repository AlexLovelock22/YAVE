use crate::models::face::{FaceDir, FaceGeometry};

/// Per-block geometry: one optional quad per face direction, in local [0, 1]³ space.
#[derive(Clone, Debug)]
pub struct BlockModel {
    pub faces: [Option<FaceGeometry>; 6],
}

impl BlockModel {
    pub fn face(&self, dir: FaceDir) -> Option<&FaceGeometry> {
        self.faces[dir as usize].as_ref()
    }
}
