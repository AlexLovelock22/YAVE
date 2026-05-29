use crate::models::{
    block_model::BlockModel,
    face::{FaceDir, FaceGeometry},
};

pub const WATER_TOP_HEIGHT: f32 = 0.9;

/// Full unit cube with corners at (0,0,0)–(1,1,1).
pub fn cube() -> BlockModel {
    BlockModel {
        faces: [
            Some(make_face(FaceDir::PosX)),
            Some(make_face(FaceDir::NegX)),

            Some(make_face(FaceDir::PosY)),
            Some(make_face(FaceDir::NegY)),

            Some(make_face(FaceDir::PosZ)),
            Some(make_face(FaceDir::NegZ)),
        ],
    }
}

/// Water block: identical geometry to a cube so all faces greedy-merge.
pub fn water() -> BlockModel {
    cube()
}

/// Half-height slab occupying y=[0, 0.5].
pub fn slab() -> BlockModel {
    BlockModel {
        faces: [
            Some(make_face_scaled(FaceDir::PosX, 0.5)),
            Some(make_face_scaled(FaceDir::NegX, 0.5)),
            Some(make_face_top_slab()),
            Some(make_face(FaceDir::NegY)),
            Some(make_face_scaled(FaceDir::PosZ, 0.5)),
            Some(make_face_scaled(FaceDir::NegZ, 0.5)),
        ],
    }
}

fn make_face(dir: FaceDir) -> FaceGeometry {
    FaceGeometry {
        verts: face_verts_scaled(dir, 1.0),
        uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        is_full: true,
        texture: 0,
    }
}

/// Side face of a slab — height clamped to `h` (0.5 for a standard slab). Not full.
fn make_face_scaled(dir: FaceDir, h: f32) -> FaceGeometry {
    FaceGeometry {
        verts: face_verts_scaled(dir, h),
        uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, h], [0.0, h]],
        is_full: false,
        texture: 0,
    }
}

/// Top face of water, sitting at WATER_TOP_HEIGHT. Not full — creates the surface lip.
fn make_face_water_top() -> FaceGeometry {
    let y = WATER_TOP_HEIGHT;
    FaceGeometry {
        verts: [[1.0, y, 0.0], [0.0, y, 0.0], [0.0, y, 1.0], [1.0, y, 1.0]],
        uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        is_full: false,
        texture: 0,
    }
}

/// Top face of a slab, sitting at y=0.5. Not full (not flush with cell top).
fn make_face_top_slab() -> FaceGeometry {
    let y = 0.5_f32;
    FaceGeometry {
        verts: [[1.0, y, 0.0], [0.0, y, 0.0], [0.0, y, 1.0], [1.0, y, 1.0]],
        uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        is_full: false,
        texture: 0,
    }
}

fn face_verts_scaled(dir: FaceDir, h: f32) -> [[f32; 3]; 4] {
    match dir {
        FaceDir::PosX => [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, h, 0.0], [1.0, h, 1.0]],
        FaceDir::NegX => [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, h, 1.0], [0.0, h, 0.0]],
        FaceDir::PosY => [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        FaceDir::NegY => [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        FaceDir::PosZ => [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, h, 1.0], [0.0, h, 1.0]],
        FaceDir::NegZ => [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, h, 0.0], [1.0, h, 0.0]],
    }
}
