use crate::{
    models::face::{FaceDir, FaceGeometry},
    render::mesh::Vertex,
    world::{
        block::{face_tex, get_model, is_opaque, BlockId, WATER},
        chunk::{Chunk, CHUNK_HEIGHT, CHUNK_SIZE},
        continents::SEA_LEVEL,
        neighbor::{mask_solid, NeighborMasks, FACE_BYTES},
    },
};

// ── Vertex AO helpers ─────────────────────────────────────────────────────────

// AO neighbor offsets per face direction, 4 corners each.
// Corner order: [u-end/v-start, u-start/v-start, u-start/v-end, u-end/v-end].
// Each entry is (side1_delta, side2_delta, corner_delta) relative to the face block.
fn ao_offsets(dir: FaceDir) -> [((i32,i32,i32),(i32,i32,i32),(i32,i32,i32)); 4] {
    match dir {
        FaceDir::PosY => [
            (( 1, 1, 0), ( 0, 1,-1), ( 1, 1,-1)),
            ((-1, 1, 0), ( 0, 1,-1), (-1, 1,-1)),
            ((-1, 1, 0), ( 0, 1, 1), (-1, 1, 1)),
            (( 1, 1, 0), ( 0, 1, 1), ( 1, 1, 1)),
        ],
        FaceDir::NegY => [
            (( 1,-1, 0), ( 0,-1,-1), ( 1,-1,-1)),
            ((-1,-1, 0), ( 0,-1,-1), (-1,-1,-1)),
            ((-1,-1, 0), ( 0,-1, 1), (-1,-1, 1)),
            (( 1,-1, 0), ( 0,-1, 1), ( 1,-1, 1)),
        ],
        FaceDir::PosX => [
            (( 1, 1, 0), ( 1, 0,-1), ( 1, 1,-1)),
            (( 1,-1, 0), ( 1, 0,-1), ( 1,-1,-1)),
            (( 1,-1, 0), ( 1, 0, 1), ( 1,-1, 1)),
            (( 1, 1, 0), ( 1, 0, 1), ( 1, 1, 1)),
        ],
        FaceDir::NegX => [
            ((-1, 1, 0), (-1, 0,-1), (-1, 1,-1)),
            ((-1,-1, 0), (-1, 0,-1), (-1,-1,-1)),
            ((-1,-1, 0), (-1, 0, 1), (-1,-1, 1)),
            ((-1, 1, 0), (-1, 0, 1), (-1, 1, 1)),
        ],
        FaceDir::PosZ => [
            (( 1, 0, 1), ( 0,-1, 1), ( 1,-1, 1)),
            ((-1, 0, 1), ( 0,-1, 1), (-1,-1, 1)),
            ((-1, 0, 1), ( 0, 1, 1), (-1, 1, 1)),
            (( 1, 0, 1), ( 0, 1, 1), ( 1, 1, 1)),
        ],
        FaceDir::NegZ => [
            (( 1, 0,-1), ( 0,-1,-1), ( 1,-1,-1)),
            ((-1, 0,-1), ( 0,-1,-1), (-1,-1,-1)),
            ((-1, 0,-1), ( 0, 1,-1), (-1, 1,-1)),
            (( 1, 0,-1), ( 0, 1,-1), ( 1, 1,-1)),
        ],
    }
}

fn ao_solid(chunk: &Chunk, neighbors: &NeighborMasks, x: i32, y: i32, z: i32) -> bool {
    if y < 0 { return true; }
    if y >= CHUNK_HEIGHT as i32 { return false; }

    let out_x = x < 0 || x >= CHUNK_SIZE as i32;
    let out_z = z < 0 || z >= CHUNK_SIZE as i32;
    if out_x && out_z { return false; }

    if out_x {
        let m = if x < 0 { neighbors.neg_x.as_ref() } else { neighbors.pos_x.as_ref() };
        return m.map_or(false, |m| mask_solid(m, y as usize, z as usize));
    }
    if out_z {
        let m = if z < 0 { neighbors.neg_z.as_ref() } else { neighbors.pos_z.as_ref() };
        return m.map_or(false, |m| mask_solid(m, y as usize, x as usize));
    }

    is_opaque(chunk.get(x as usize, y as usize, z as usize))
}

// Compute per-corner AO [0..=3] for a face. Corner order matches ao_offsets.
fn face_ao(chunk: &Chunk, neighbors: &NeighborMasks, bx: usize, by: usize, bz: usize, dir: FaceDir) -> [u8; 4] {
    let (bx, by, bz) = (bx as i32, by as i32, bz as i32);
    let offsets = ao_offsets(dir);
    let mut result = [0u8; 4];
    for (i, (s1, s2, c)) in offsets.iter().enumerate() {
        let side1  = ao_solid(chunk, neighbors, bx + s1.0, by + s1.1, bz + s1.2);
        let side2  = ao_solid(chunk, neighbors, bx + s2.0, by + s2.1, bz + s2.2);
        let corner = ao_solid(chunk, neighbors, bx +  c.0, by +  c.1, bz +  c.2);
        result[i] = if side1 || side2 || corner { 2 } else { 3 };
    }
    result
}

// ── LOD surface mesher ────────────────────────────────────────────────────────

/// Surface-only LOD mesh for medium-distance chunks.
pub fn mesh_chunk_surface(chunk: &Chunk, neighbors: &NeighborMasks) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices  = Vec::new();
    let cs = CHUNK_SIZE;
    let mh = chunk.max_height;

    let mut surf = vec![None::<(BlockId, usize)>; cs * cs];
    for x in 0..cs {
        for z in 0..cs {
            for y in (0..=mh).rev() {
                let b = chunk.get(x, y, z);
                if is_opaque(b) {
                    surf[x * cs + z] = Some((b, y));
                    break;
                }
            }
        }
    }

    let surf_h = |x: usize, z: usize| -> i32 {
        surf[x * cs + z].map_or(-1, |(_, y)| y as i32)
    };

    let face_top = |face_mask: &[u8; FACE_BYTES], lateral: usize| -> i32 {
        for y in (0..CHUNK_HEIGHT).rev() {
            if mask_solid(face_mask, y, lateral) { return y as i32; }
        }
        -1
    };

    // PosY: top face of each column, 2D greedy merge — no AO at LOD distance
    {
        let mut used = vec![false; cs * cs];
        for x in 0..cs {
            for z in 0..cs {
                let i = x * cs + z;
                if used[i] { continue; }
                let Some((block, y0)) = surf[i] else { continue };

                let mut dz = 1;
                while z + dz < cs {
                    match surf[x * cs + z + dz] {
                        Some((b, y)) if b == block && y == y0 => dz += 1,
                        _ => break,
                    }
                }
                let mut dx = 1;
                'ex: loop {
                    if x + dx >= cs { break; }
                    for zz in z..z + dz {
                        match surf[(x + dx) * cs + zz] {
                            Some((b, y)) if b == block && y == y0 => {}
                            _ => break 'ex,
                        }
                    }
                    dx += 1;
                }
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosY, y0, x, z, dx, dz, chunk, block, [1.0; 4]);
                for xx in x..x + dx {
                    for zz in z..z + dz { used[xx * cs + zz] = true; }
                }
            }
        }
    }

    let row = mh + 1;
    let mut mask = vec![None::<BlockId>; cs * row];
    let mut used = vec![false; cs * row];

    // PosX
    for d in 0..cs {
        mask.fill(None);
        used.fill(false);
        for lat in 0..cs {
            let th = surf_h(d, lat);
            if th < 0 { continue; }
            let nh = if d + 1 < cs {
                surf_h(d + 1, lat)
            } else {
                neighbors.pos_x.as_ref().map_or(th, |m| face_top(m, lat))
            };
            let start = (nh + 1).max(0) as usize;
            for y in start..=th as usize {
                let b = chunk.get(d, y, lat);
                if is_opaque(b) { mask[lat * row + y] = Some(b); }
            }
        }
        for lat in 0..cs {
            for y in 0..row {
                let i = lat * row + y;
                if used[i] || mask[i].is_none() { continue; }
                let block = mask[i].unwrap();
                let mut dy = 1;
                while y + dy < row && mask[lat * row + y + dy] == Some(block) { dy += 1; }
                let mut dlat = 1;
                'ex: loop {
                    if lat + dlat >= cs { break; }
                    for yy in y..y + dy {
                        if mask[(lat + dlat) * row + yy] != Some(block) { break 'ex; }
                    }
                    dlat += 1;
                }
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosX, d, y, lat, dy, dlat, chunk, block, [1.0; 4]);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    // NegX
    for d in 0..cs {
        mask.fill(None);
        used.fill(false);
        for lat in 0..cs {
            let th = surf_h(d, lat);
            if th < 0 { continue; }
            let nh = if d > 0 {
                surf_h(d - 1, lat)
            } else {
                neighbors.neg_x.as_ref().map_or(th, |m| face_top(m, lat))
            };
            let start = (nh + 1).max(0) as usize;
            for y in start..=th as usize {
                let b = chunk.get(d, y, lat);
                if is_opaque(b) { mask[lat * row + y] = Some(b); }
            }
        }
        for lat in 0..cs {
            for y in 0..row {
                let i = lat * row + y;
                if used[i] || mask[i].is_none() { continue; }
                let block = mask[i].unwrap();
                let mut dy = 1;
                while y + dy < row && mask[lat * row + y + dy] == Some(block) { dy += 1; }
                let mut dlat = 1;
                'ex: loop {
                    if lat + dlat >= cs { break; }
                    for yy in y..y + dy {
                        if mask[(lat + dlat) * row + yy] != Some(block) { break 'ex; }
                    }
                    dlat += 1;
                }
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::NegX, d, y, lat, dy, dlat, chunk, block, [1.0; 4]);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    // PosZ
    for d in 0..cs {
        mask.fill(None);
        used.fill(false);
        for lat in 0..cs {
            let th = surf_h(lat, d);
            if th < 0 { continue; }
            let nh = if d + 1 < cs {
                surf_h(lat, d + 1)
            } else {
                neighbors.pos_z.as_ref().map_or(th, |m| face_top(m, lat))
            };
            let start = (nh + 1).max(0) as usize;
            for y in start..=th as usize {
                let b = chunk.get(lat, y, d);
                if is_opaque(b) { mask[lat * row + y] = Some(b); }
            }
        }
        for lat in 0..cs {
            for y in 0..row {
                let i = lat * row + y;
                if used[i] || mask[i].is_none() { continue; }
                let block = mask[i].unwrap();
                let mut dy = 1;
                while y + dy < row && mask[lat * row + y + dy] == Some(block) { dy += 1; }
                let mut dlat = 1;
                'ex: loop {
                    if lat + dlat >= cs { break; }
                    for yy in y..y + dy {
                        if mask[(lat + dlat) * row + yy] != Some(block) { break 'ex; }
                    }
                    dlat += 1;
                }
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosZ, d, lat, y, dlat, dy, chunk, block, [1.0; 4]);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    // NegZ
    for d in 0..cs {
        mask.fill(None);
        used.fill(false);
        for lat in 0..cs {
            let th = surf_h(lat, d);
            if th < 0 { continue; }
            let nh = if d > 0 {
                surf_h(lat, d - 1)
            } else {
                neighbors.neg_z.as_ref().map_or(th, |m| face_top(m, lat))
            };
            let start = (nh + 1).max(0) as usize;
            for y in start..=th as usize {
                let b = chunk.get(lat, y, d);
                if is_opaque(b) { mask[lat * row + y] = Some(b); }
            }
        }
        for lat in 0..cs {
            for y in 0..row {
                let i = lat * row + y;
                if used[i] || mask[i].is_none() { continue; }
                let block = mask[i].unwrap();
                let mut dy = 1;
                while y + dy < row && mask[lat * row + y + dy] == Some(block) { dy += 1; }
                let mut dlat = 1;
                'ex: loop {
                    if lat + dlat >= cs { break; }
                    for yy in y..y + dy {
                        if mask[(lat + dlat) * row + yy] != Some(block) { break 'ex; }
                    }
                    dlat += 1;
                }
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::NegZ, d, lat, y, dlat, dy, chunk, block, [1.0; 4]);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    (vertices, indices)
}

// ── Full mesher ───────────────────────────────────────────────────────────────

pub fn mesh_chunk(chunk: &Chunk, neighbors: &NeighborMasks) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let mh = chunk.max_height + 1;

    for dir in FaceDir::ALL {
        let (depth_len, u_len, v_len) = dir_dims(dir, mh);

        // Mask stores (BlockId, per-corner AO [0..3]) for each greedy cell.
        let mut mask = vec![None::<(BlockId, [u8; 4])>; u_len * v_len];
        let mut used = vec![false; u_len * v_len];

        for d in 0..depth_len {
            mask.fill(None);
            used.fill(false);

            for u in 0..u_len {
                for v in 0..v_len {
                    let (x, y, z) = to_xyz(dir, d, u, v);
                    let id = chunk.get(x, y, z);
                    if dir == FaceDir::NegY { continue; }
                    let Some(model) = get_model(id) else { continue };
                    if id == WATER { continue; }
                    let Some(face) = model.face(dir) else { continue };
                    if !is_exposed(chunk, neighbors, x, y, z, dir) { continue; }

                    if face.is_full {
                        let ao = face_ao(chunk, neighbors, x, y, z, dir);
                        mask[u * v_len + v] = Some((id, ao));
                    } else {
                        emit_model_face(&mut vertices, &mut indices, face, dir, x, y, z, id, chunk);
                    }
                }
            }

            for u in 0..u_len {
                for v in 0..v_len {
                    let i = u * v_len + v;
                    if used[i] || mask[i].is_none() { continue; }
                    let (block, start_ao) = mask[i].unwrap();

                    // Non-uniform faces (different AO at corners) stay as 1×1 quads so
                    // the per-corner gradient is visible on each individual block face.
                    // Uniform faces (all 4 corners equal) can be greedy-merged freely.
                    let ao_uniform = start_ao[0] == start_ao[1]
                                  && start_ao[1] == start_ao[2]
                                  && start_ao[2] == start_ao[3];
                    let ao_value = start_ao[0];

                    let mut dv = 1;
                    let mut du = 1;

                    if ao_uniform {
                        while v + dv < v_len {
                            let ok = mask[u * v_len + v + dv].map_or(false, |(b, ao)| {
                                b == block && ao[0] == ao_value && ao[1] == ao_value
                                          && ao[2] == ao_value && ao[3] == ao_value
                            });
                            if !ok { break; }
                            dv += 1;
                        }
                        'expand: loop {
                            if u + du >= u_len { break; }
                            for vv in v..v + dv {
                                let ok = mask[(u + du) * v_len + vv].map_or(false, |(b, ao)| {
                                    b == block && ao[0] == ao_value && ao[1] == ao_value
                                              && ao[2] == ao_value && ao[3] == ao_value
                                });
                                if !ok { break 'expand; }
                            }
                            du += 1;
                        }
                    }

                    let corner_ao = if ao_uniform {
                        let f = ao_value as f32 / 3.0;
                        [f; 4]
                    } else {
                        [start_ao[0] as f32 / 3.0, start_ao[1] as f32 / 3.0,
                         start_ao[2] as f32 / 3.0, start_ao[3] as f32 / 3.0]
                    };

                    emit_greedy_quad(&mut vertices, &mut indices, dir, d, u, v, du, dv, chunk, block, corner_ao);

                    for uu in u..u + du {
                        for vv in v..v + dv { used[uu * v_len + vv] = true; }
                    }
                }
            }
        }
    }

    (vertices, indices)
}

// ── Dimension helpers ─────────────────────────────────────────────────────────

fn dir_dims(dir: FaceDir, mh: usize) -> (usize, usize, usize) {
    match dir {
        FaceDir::PosX | FaceDir::NegX => (CHUNK_SIZE, mh,         CHUNK_SIZE),
        FaceDir::PosY | FaceDir::NegY => (mh,         CHUNK_SIZE, CHUNK_SIZE),
        FaceDir::PosZ | FaceDir::NegZ => (CHUNK_SIZE, CHUNK_SIZE, mh        ),
    }
}

fn to_xyz(dir: FaceDir, d: usize, u: usize, v: usize) -> (usize, usize, usize) {
    match dir {
        FaceDir::PosX | FaceDir::NegX => (d, u, v),
        FaceDir::PosY | FaceDir::NegY => (u, d, v),
        FaceDir::PosZ | FaceDir::NegZ => (u, v, d),
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
        return match dir {
            FaceDir::PosX => neighbors.pos_x.as_ref().map_or(false, |m| !mask_solid(m, y, z)),
            FaceDir::NegX => neighbors.neg_x.as_ref().map_or(false, |m| !mask_solid(m, y, z)),
            FaceDir::PosZ => neighbors.pos_z.as_ref().map_or(false, |m| !mask_solid(m, y, x)),
            FaceDir::NegZ => neighbors.neg_z.as_ref().map_or(false, |m| !mask_solid(m, y, x)),
            FaceDir::PosY | FaceDir::NegY => true,
        };
    }
    !is_opaque(chunk.get(nx, ny, nz))
}

// ── Quad emitters ─────────────────────────────────────────────────────────────

fn quad_uvs(dir: FaceDir, s: f32, t: f32) -> [[f32; 2]; 4] {
    match dir {
        FaceDir::PosX => [[s, 0.], [0., 0.], [0., t], [s, t]],
        FaceDir::NegX => [[0., 0.], [s, 0.], [s, t], [0., t]],
        FaceDir::PosY => [[s, 0.], [0., 0.], [0., t], [s, t]],
        FaceDir::NegY => [[s, t],  [0., t],  [0., 0.], [s, 0.]],
        FaceDir::PosZ => [[0., 0.], [s, 0.], [s, t], [0., t]],
        FaceDir::NegZ => [[s, 0.], [0., 0.], [0., t], [s, t]],
    }
}

fn emit_model_face(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    face: &FaceGeometry,
    dir: FaceDir,
    x: usize, y: usize, z: usize,
    block: BlockId,
    chunk: &Chunk,
) {
    let base   = vertices.len() as u32;
    let normal = dir.normal();
    let layer  = face_tex(block, dir) as f32;
    let (ox, oy, oz) = (
        chunk.origin.x as f32 + x as f32,
        chunk.origin.y as f32 + y as f32,
        chunk.origin.z as f32 + z as f32,
    );
    for i in 0..4 {
        vertices.push(Vertex {
            pos:    [face.verts[i][0] + ox, face.verts[i][1] + oy, face.verts[i][2] + oz],
            normal,
            uv:     [face.uvs[i][0], face.uvs[i][1] + layer * 256.0],
            ao:     1.0,
        });
    }
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

pub fn mesh_chunk_water(chunk: &Chunk, neighbors: &NeighborMasks) -> (Vec<Vertex>, Vec<u32>) {
    let has_water = (0..CHUNK_SIZE).any(|x|
        (0..CHUNK_SIZE).any(|z| chunk.get(x, SEA_LEVEL, z) == WATER)
    );
    if !has_water { return (vec![], vec![]); }

    let cs = CHUNK_SIZE;
    let mh = chunk.max_height;
    let mut vertices = Vec::new();
    let mut indices  = Vec::new();
    let mut mask = vec![None::<usize>; cs * cs];
    let mut used = vec![false; cs * cs];

    for x in 0..cs {
        for z in 0..cs {
            for y in (0..=mh).rev() {
                if chunk.get(x, y, z) != WATER { continue; }
                if is_exposed(chunk, neighbors, x, y, z, FaceDir::PosY) {
                    mask[x * cs + z] = Some(y);
                }
                break;
            }
        }
    }

    for x in 0..cs {
        for z in 0..cs {
            let i = x * cs + z;
            if used[i] { continue; }
            let Some(y0) = mask[i] else { continue };

            let mut dz = 1;
            while z + dz < cs && mask[x * cs + z + dz] == Some(y0) { dz += 1; }
            let mut dx = 1;
            'ex: loop {
                if x + dx >= cs { break; }
                for zz in z..z + dz {
                    if mask[(x + dx) * cs + zz] != Some(y0) { break 'ex; }
                }
                dx += 1;
            }

            emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosY, y0, x, z, dx, dz, chunk, WATER, [1.0; 4]);
            let base = vertices.len() - 4;
            for v in &mut vertices[base..] { v.pos[1] -= 0.1; }

            for xx in x..x + dx {
                for zz in z..z + dz { used[xx * cs + zz] = true; }
            }
        }
    }

    (vertices, indices)
}

/// Emit a merged quad.
/// `ao`: corner AO values in [u-end/v-start, u-start/v-start, u-start/v-end, u-end/v-end] order,
/// normalised to [0.0, 1.0]. Each vertex gets the correct corner based on face winding.
fn emit_greedy_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    dir: FaceDir,
    d: usize, u: usize, v: usize,
    du: usize, dv: usize,
    chunk: &Chunk,
    block: BlockId,
    ao: [f32; 4],
) {
    let (ox, oy, oz) = (
        chunk.origin.x as f32,
        chunk.origin.y as f32,
        chunk.origin.z as f32,
    );
    let (d, u, v, du, dv) = (d as f32, u as f32, v as f32, du as f32, dv as f32);

    let quad: [[f32; 3]; 4] = match dir {
        FaceDir::PosX => [[d+1.+ox, u+oy,     v+dv+oz], [d+1.+ox, u+oy,     v+oz    ], [d+1.+ox, u+du+oy, v+oz    ], [d+1.+ox, u+du+oy, v+dv+oz]],
        FaceDir::NegX => [[d+ox,    u+oy,     v+oz    ], [d+ox,    u+oy,     v+dv+oz], [d+ox,    u+du+oy, v+dv+oz], [d+ox,    u+du+oy, v+oz    ]],
        FaceDir::PosY => [[u+du+ox, d+1.+oy,  v+oz    ], [u+ox,    d+1.+oy,  v+oz   ], [u+ox,    d+1.+oy,  v+dv+oz], [u+du+ox, d+1.+oy,  v+dv+oz]],
        FaceDir::NegY => [[u+du+ox, d+oy,     v+dv+oz], [u+ox,    d+oy,     v+dv+oz], [u+ox,    d+oy,     v+oz    ], [u+du+ox, d+oy,     v+oz    ]],
        FaceDir::PosZ => [[u+ox,    v+oy,     d+1.+oz], [u+du+ox, v+oy,     d+1.+oz], [u+du+ox, v+dv+oy, d+1.+oz], [u+ox,    v+dv+oy, d+1.+oz]],
        FaceDir::NegZ => [[u+du+ox, v+oy,     d+oz    ], [u+ox,    v+oy,     d+oz   ], [u+ox,    v+dv+oy, d+oz    ], [u+du+ox, v+dv+oy, d+oz    ]],
    };

    // Map corner AO indices [u-end/v-start, u-start/v-start, u-start/v-end, u-end/v-end]
    // to per-vertex order (matches quad winding above).
    let vertex_ao: [f32; 4] = match dir {
        FaceDir::PosX => [ao[2], ao[1], ao[0], ao[3]],
        FaceDir::NegX => [ao[1], ao[2], ao[3], ao[0]],
        FaceDir::PosY => [ao[0], ao[1], ao[2], ao[3]],
        FaceDir::NegY => [ao[3], ao[2], ao[1], ao[0]],
        FaceDir::PosZ => [ao[1], ao[0], ao[3], ao[2]],
        FaceDir::NegZ => [ao[0], ao[1], ao[2], ao[3]],
    };

    let (s, t) = match dir {
        FaceDir::PosX | FaceDir::NegX => (dv, du),
        _                              => (du, dv),
    };
    let uvs   = quad_uvs(dir, s, t);
    let layer = face_tex(block, dir) as f32;

    let base   = vertices.len() as u32;
    let normal = dir.normal();
    for i in 0..4 {
        vertices.push(Vertex {
            pos:    quad[i],
            normal,
            uv:     [uvs[i][0], uvs[i][1] + layer * 256.0],
            ao:     vertex_ao[i],
        });
    }
    if vertex_ao[0] + vertex_ao[2] < vertex_ao[1] + vertex_ao[3] {
        indices.extend_from_slice(&[base, base+1, base+3, base+1, base+2, base+3]);
    } else {
        indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
}
