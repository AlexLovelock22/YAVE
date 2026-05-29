#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceDir {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl FaceDir {
    pub const ALL: [FaceDir; 6] = [
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ];

    pub fn normal(self) -> [f32; 3] {
        match self {
            FaceDir::PosX => [1.0, 0.0, 0.0],
            FaceDir::NegX => [-1.0, 0.0, 0.0],
            FaceDir::PosY => [0.0, 1.0, 0.0],
            FaceDir::NegY => [0.0, -1.0, 0.0],
            FaceDir::PosZ => [0.0, 0.0, 1.0],
            FaceDir::NegZ => [0.0, 0.0, -1.0],
        }
    }

    pub fn opposite(self) -> FaceDir {
        match self {
            FaceDir::PosX => FaceDir::NegX,
            FaceDir::NegX => FaceDir::PosX,
            FaceDir::PosY => FaceDir::NegY,
            FaceDir::NegY => FaceDir::PosY,
            FaceDir::PosZ => FaceDir::NegZ,
            FaceDir::NegZ => FaceDir::PosZ,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaceGeometry {
    /// 4 quad corners in local block space [0, 1]³, winding order: CCW when facing outward
    pub verts: [[f32; 3]; 4],
    pub uvs: [[f32; 2]; 4],
    /// True when this face completely covers its cell side — enables face culling and greedy merge
    pub is_full: bool,
    pub texture: u32,
}
