use crate::{
    models::face::{FaceDir, FaceGeometry},
    render::mesh::Vertex,
    world::{
        block::{get_model, BlockId},
        chunk::{Chunk, CHUNK_HEIGHT, CHUNK_SIZE},
        neighbor::{mask_solid, NeighborMasks},
    },
};

/// LOD variant: same greedy algorithm but sampled at `scale`-block intervals.
/// Boundary faces are always emitted (no neighbour-mask lookup) — acceptable for
/// distant chunks where the overdraw is sub-pixel.
pub fn mesh_chunk_lod(chunk: &Chunk, scale: usize) -> (Vec<Vertex>, Vec<u32>) {
    debug_assert!(scale > 1 && CHUNK_SIZE % scale == 0);
    let mut vertices = Vec::new();
    let mut indices  = Vec::new();

    let cs = CHUNK_SIZE   / scale;
    let ch = CHUNK_HEIGHT / scale;
    let mh = ((chunk.max_height / scale) + 1).min(ch);

    for dir in FaceDir::ALL {
        if dir == FaceDir::NegY { continue; }
        let (dl, ul, vl) = match dir {
            FaceDir::PosX | FaceDir::NegX => (cs, mh, cs),
            FaceDir::PosY | FaceDir::NegY => (mh, cs, cs),
            FaceDir::PosZ | FaceDir::NegZ => (cs, cs, mh),
        };

        let mut mask = vec![None::<BlockId>; ul * vl];
        let mut used = vec![false;          ul * vl];

        for d in 0..dl {
            mask.fill(None);
            used.fill(false);

            for u in 0..ul {
                for v in 0..vl {
                    // Sample one block per meta-cell (nearest-neighbour).
                    let (bx, by, bz) = to_xyz(dir, d * scale, u * scale, v * scale);
                    if bx >= CHUNK_SIZE || by >= CHUNK_HEIGHT || bz >= CHUNK_SIZE { continue; }

                    let id = chunk.get(bx, by, bz);
                    let Some(model) = get_model(id) else { continue };
                    // Only full faces participate in greedy merging at LOD; skip sub-block models.
                    if model.face(dir).map(|f| !f.is_full).unwrap_or(true) { continue; }
                    if !lod_exposed(chunk, bx, by, bz, dir, scale) { continue; }

                    mask[u * vl + v] = Some(id);
                }
            }

            for u in 0..ul {
                for v in 0..vl {
                    let i = u * vl + v;
                    if used[i] || mask[i].is_none() { continue; }
                    let block = mask[i].unwrap();

                    let mut dv = 1;
                    while v + dv < vl && mask[u * vl + v + dv] == Some(block) { dv += 1; }

                    let mut du = 1;
                    'expand: loop {
                        if u + du >= ul { break; }
                        for vv in v..v + dv {
                            if mask[(u + du) * vl + vv] != Some(block) { break 'expand; }
                        }
                        du += 1;
                    }

                    emit_greedy_quad_scaled(
                        &mut vertices, &mut indices,
                        dir, d, u, v, du, dv, scale, chunk, block,
                    );

                    for uu in u..u + du {
                        for vv in v..v + dv {
                            used[uu * vl + vv] = true;
                        }
                    }
                }
            }
        }
    }
    (vertices, indices)
}

/// Exposure check for LOD meshes: checks the adjacent meta-block inside the same chunk,
/// or exposes the face unconditionally at a chunk boundary.
fn lod_exposed(chunk: &Chunk, bx: usize, by: usize, bz: usize, dir: FaceDir, s: usize) -> bool {
    match dir {
        FaceDir::PosX => { let nx = bx + s; if nx >= CHUNK_SIZE   { return true; } get_model(chunk.get(nx, by, bz)).is_none() }
        FaceDir::NegX => {                   if bx < s             { return true; } get_model(chunk.get(bx - s, by, bz)).is_none() }
        FaceDir::PosY => { let ny = by + s; if ny >= CHUNK_HEIGHT  { return true; } get_model(chunk.get(bx, ny, bz)).is_none() }
        FaceDir::NegY => {                   if by < s             { return true; } get_model(chunk.get(bx, by - s, bz)).is_none() }
        FaceDir::PosZ => { let nz = bz + s; if nz >= CHUNK_SIZE   { return true; } get_model(chunk.get(bx, by, nz)).is_none() }
        FaceDir::NegZ => {                   if bz < s             { return true; } get_model(chunk.get(bx, by, bz - s)).is_none() }
    }
}

/// Like `emit_greedy_quad` but multiplies all coordinates by `scale` so the quad covers
/// `scale` world units per meta-block unit.
fn emit_greedy_quad_scaled(
    vertices: &mut Vec<Vertex>,
    indices:  &mut Vec<u32>,
    dir: FaceDir,
    d: usize, u: usize, v: usize,
    du: usize, dv: usize,
    scale: usize,
    chunk: &Chunk,
    block: BlockId,
) {
    let s  = scale as f32;
    let (ox, oy, oz) = (chunk.origin.x as f32, chunk.origin.y as f32, chunk.origin.z as f32);
    let (d, u, v, du, dv) = (d as f32 * s, u as f32 * s, v as f32 * s, du as f32 * s, dv as f32 * s);

    let quad: [[f32; 3]; 4] = match dir {
        FaceDir::PosX => [[d+s+ox, u+oy,     v+dv+oz], [d+s+ox, u+oy,     v+oz    ], [d+s+ox, u+du+oy, v+oz    ], [d+s+ox, u+du+oy, v+dv+oz]],
        FaceDir::NegX => [[d+ox,   u+oy,     v+oz    ], [d+ox,   u+oy,     v+dv+oz], [d+ox,   u+du+oy, v+dv+oz], [d+ox,   u+du+oy, v+oz    ]],
        FaceDir::PosY => [[u+du+ox, d+s+oy,  v+oz    ], [u+ox,   d+s+oy,  v+oz    ], [u+ox,   d+s+oy,  v+dv+oz], [u+du+ox, d+s+oy,  v+dv+oz]],
        FaceDir::NegY => [[u+du+ox, d+oy,    v+dv+oz], [u+ox,   d+oy,    v+dv+oz], [u+ox,   d+oy,    v+oz    ], [u+du+ox, d+oy,    v+oz    ]],
        FaceDir::PosZ => [[u+ox,    v+oy,    d+s+oz  ], [u+du+ox, v+oy,  d+s+oz  ], [u+du+ox, v+dv+oy, d+s+oz], [u+ox,    v+dv+oy, d+s+oz  ]],
        FaceDir::NegZ => [[u+du+ox, v+oy,    d+oz    ], [u+ox,   v+oy,    d+oz   ], [u+ox,   v+dv+oy, d+oz    ], [u+du+ox, v+dv+oy, d+oz    ]],
    };

    let base   = vertices.len() as u32;
    let normal = dir.normal();
    let color  = block.0 as f32;
    for pos in &quad {
        vertices.push(Vertex { pos: *pos, normal, uv: [color, 0.0] });
    }
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

pub fn mesh_chunk(chunk: &Chunk, neighbors: &NeighborMasks) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Only iterate up to the highest occupied Y level — skips empty air above terrain.
    let mh = chunk.max_height + 1;

    for dir in FaceDir::ALL {
        // Bottom faces are never visible in an above-ground heightmap world.
        if dir == FaceDir::NegY { continue; }
        let (depth_len, u_len, v_len) = dir_dims(dir, mh);

        // Reuse allocations across depth layers
        let mut mask = vec![None::<BlockId>; u_len * v_len];
        let mut used = vec![false; u_len * v_len];

        for d in 0..depth_len {
            mask.fill(None);
            used.fill(false);

            // Build the face mask for this slice
            for u in 0..u_len {
                for v in 0..v_len {
                    let (x, y, z) = to_xyz(dir, d, u, v);
                    let id = chunk.get(x, y, z);
                    let Some(model) = get_model(id) else { continue };
                    let Some(face) = model.face(dir) else { continue };
                    if !is_exposed(chunk, neighbors, x, y, z, dir) { continue; }

                    if face.is_full {
                        // Will be greedy-merged below
                        mask[u * v_len + v] = Some(id);
                    } else {
                        // Non-full face (slab sides, etc.): emit using the model's exact geometry
                        emit_model_face(&mut vertices, &mut indices, face, dir, x, y, z, id, chunk);
                    }
                }
            }

            // Greedy merge: expand each unprocessed cell into the largest same-block rectangle
            for u in 0..u_len {
                for v in 0..v_len {
                    let i = u * v_len + v;
                    if used[i] || mask[i].is_none() { continue; }
                    let block = mask[i].unwrap();

                    // Expand right (v direction) as far as the same block type continues
                    let mut dv = 1;
                    while v + dv < v_len && mask[u * v_len + v + dv] == Some(block) {
                        dv += 1;
                    }

                    // Expand down (u direction): every row must match across the full v span
                    let mut du = 1;
                    'expand: loop {
                        if u + du >= u_len { break; }
                        for vv in v..v + dv {
                            if mask[(u + du) * v_len + vv] != Some(block) { break 'expand; }
                        }
                        du += 1;
                    }

                    emit_greedy_quad(&mut vertices, &mut indices, dir, d, u, v, du, dv, chunk, block);

                    // Mark the merged rectangle so we don't re-process its cells
                    for uu in u..u + du {
                        for vv in v..v + dv {
                            used[uu * v_len + vv] = true;
                        }
                    }
                }
            }
        }
    }

    (vertices, indices)
}

// ── Dimension helpers ────────────────────────────────────────────────────────

/// (depth_len, u_len, v_len) for the 2D slice perpendicular to each direction.
///   PosX/NegX: d=x,  u=y (height), v=z
///   PosY/NegY: d=y,  u=x,          v=z
///   PosZ/NegZ: d=z,  u=x,          v=y (height)
/// Returns (depth_len, u_len, v_len) for iteration. `mh` caps the height dimension
/// so we skip the empty air column above the highest block in the chunk.
fn dir_dims(dir: FaceDir, mh: usize) -> (usize, usize, usize) {
    match dir {
        FaceDir::PosX | FaceDir::NegX => (CHUNK_SIZE, mh,         CHUNK_SIZE),
        FaceDir::PosY | FaceDir::NegY => (mh,         CHUNK_SIZE, CHUNK_SIZE),
        FaceDir::PosZ | FaceDir::NegZ => (CHUNK_SIZE, CHUNK_SIZE, mh        ),
    }
}

fn to_xyz(dir: FaceDir, d: usize, u: usize, v: usize) -> (usize, usize, usize) {
    match dir {
        FaceDir::PosX | FaceDir::NegX => (d, u, v), // x=d, y=u, z=v
        FaceDir::PosY | FaceDir::NegY => (u, d, v), // x=u, y=d, z=v
        FaceDir::PosZ | FaceDir::NegZ => (u, v, d), // x=u, y=v, z=d
    }
}

fn is_exposed(chunk: &Chunk, neighbors: &NeighborMasks, x: usize, y: usize, z: usize, dir: FaceDir) -> bool {
    let (nx, ny, nz) = match dir {
        FaceDir::PosX => (x + 1,             y,                 z              ),
        FaceDir::NegX => (x.wrapping_sub(1), y,                 z              ),
        FaceDir::PosY => (x,                 y + 1,             z              ),
        FaceDir::NegY => (x,                 y.wrapping_sub(1), z              ),
        FaceDir::PosZ => (x,                 y,                 z + 1          ),
        FaceDir::NegZ => (x,                 y,                 z.wrapping_sub(1)),
    };
    if nx >= CHUNK_SIZE || ny >= CHUNK_HEIGHT || nz >= CHUNK_SIZE {
        // Cross-chunk boundary: use neighbour mask if available, else treat as exposed
        return match dir {
            FaceDir::PosX => neighbors.pos_x.as_ref().map_or(true, |m| !mask_solid(m, y, z)),
            FaceDir::NegX => neighbors.neg_x.as_ref().map_or(true, |m| !mask_solid(m, y, z)),
            FaceDir::PosZ => neighbors.pos_z.as_ref().map_or(true, |m| !mask_solid(m, y, x)),
            FaceDir::NegZ => neighbors.neg_z.as_ref().map_or(true, |m| !mask_solid(m, y, x)),
            FaceDir::PosY | FaceDir::NegY => true,
        };
    }
    get_model(chunk.get(nx, ny, nz)).is_none()
}

// ── Quad emitters ────────────────────────────────────────────────────────────

/// Emit a quad using the BlockModel's exact vertex geometry (used for non-full faces).
fn emit_model_face(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    face: &FaceGeometry,
    dir: FaceDir,
    x: usize, y: usize, z: usize,
    block: BlockId,
    chunk: &Chunk,
) {
    let base = vertices.len() as u32;
    let normal = dir.normal();
    let color = block.0 as f32;
    let (ox, oy, oz) = (
        chunk.origin.x as f32 + x as f32,
        chunk.origin.y as f32 + y as f32,
        chunk.origin.z as f32 + z as f32,
    );
    for i in 0..4 {
        vertices.push(Vertex {
            pos: [face.verts[i][0] + ox, face.verts[i][1] + oy, face.verts[i][2] + oz],
            normal,
            uv: [color, 0.0],
        });
    }
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

/// Emit a merged quad covering a du×dv rectangle at depth d in (u, v) slice space.
/// Vertex order matches the winding established in builtin::face_verts_scaled.
fn emit_greedy_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    dir: FaceDir,
    d: usize, u: usize, v: usize,
    du: usize, dv: usize,
    chunk: &Chunk,
    block: BlockId,
) {
    let (ox, oy, oz) = (
        chunk.origin.x as f32,
        chunk.origin.y as f32,
        chunk.origin.z as f32,
    );
    let (d, u, v, du, dv) = (d as f32, u as f32, v as f32, du as f32, dv as f32);

    // Positions are derived from the same winding used in builtin::face_verts_scaled,
    // scaled by (du, dv) to cover the merged rectangle.
    let quad: [[f32; 3]; 4] = match dir {
        // d=x, u=y, v=z  — face on +x wall at x=d+1
        FaceDir::PosX => [[d+1.+ox, u+oy,     v+dv+oz], [d+1.+ox, u+oy,     v+oz    ], [d+1.+ox, u+du+oy, v+oz    ], [d+1.+ox, u+du+oy, v+dv+oz]],
        // d=x, u=y, v=z  — face on -x wall at x=d
        FaceDir::NegX => [[d+ox,    u+oy,     v+oz    ], [d+ox,    u+oy,     v+dv+oz], [d+ox,    u+du+oy, v+dv+oz], [d+ox,    u+du+oy, v+oz    ]],
        // d=y, u=x, v=z  — face on +y ceiling at y=d+1
        FaceDir::PosY => [[u+du+ox, d+1.+oy,  v+oz    ], [u+ox,    d+1.+oy,  v+oz   ], [u+ox,    d+1.+oy,  v+dv+oz], [u+du+ox, d+1.+oy,  v+dv+oz]],
        // d=y, u=x, v=z  — face on -y floor at y=d
        FaceDir::NegY => [[u+du+ox, d+oy,     v+dv+oz], [u+ox,    d+oy,     v+dv+oz], [u+ox,    d+oy,     v+oz    ], [u+du+ox, d+oy,     v+oz    ]],
        // d=z, u=x, v=y  — face on +z wall at z=d+1
        FaceDir::PosZ => [[u+ox,    v+oy,     d+1.+oz], [u+du+ox, v+oy,     d+1.+oz], [u+du+ox, v+dv+oy, d+1.+oz], [u+ox,    v+dv+oy, d+1.+oz]],
        // d=z, u=x, v=y  — face on -z wall at z=d
        FaceDir::NegZ => [[u+du+ox, v+oy,     d+oz    ], [u+ox,    v+oy,     d+oz   ], [u+ox,    v+dv+oy, d+oz    ], [u+du+ox, v+dv+oy, d+oz    ]],
    };

    let base = vertices.len() as u32;
    let normal = dir.normal();
    let color = block.0 as f32;
    for pos in &quad {
        vertices.push(Vertex { pos: *pos, normal, uv: [color, 0.0] });
    }
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}
