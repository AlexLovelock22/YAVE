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

/// Surface-only LOD mesh for medium-distance chunks.
///
/// Builds a per-column heightfield and emits only the topmost face of each column
/// plus cliff side-faces where adjacent columns differ in height. Blocks are rendered
/// at full 1×1×1 world-unit resolution, so the result looks identical to mesh_chunk
/// for solid heightmap terrain. The savings vs mesh_chunk come from iterating
/// CHUNK_SIZE² columns instead of the full 3D block volume.
pub fn mesh_chunk_surface(chunk: &Chunk, neighbors: &NeighborMasks) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices  = Vec::new();
    let cs = CHUNK_SIZE;
    let mh = chunk.max_height;

    // ── Per-column surface: (block_id, y) of topmost solid block ─────────────
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

    // Column height as i32 (-1 = empty/air column).
    let surf_h = |x: usize, z: usize| -> i32 {
        surf[x * cs + z].map_or(-1, |(_, y)| y as i32)
    };

    // Topmost solid y in a neighbor face-mask (-1 = empty column in that neighbor).
    let face_top = |face_mask: &[u8; FACE_BYTES], lateral: usize| -> i32 {
        for y in (0..CHUNK_HEIGHT).rev() {
            if mask_solid(face_mask, y, lateral) { return y as i32; }
        }
        -1
    };

    // ── PosY: top face of each column, 2D greedy merge on (x, z) ─────────────
    // Two cells can merge only if same block type AND same surface height (flat quad).
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
                // PosY: d=y, u=x, v=z
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosY, y0, x, z, dx, dz, chunk, block);
                for xx in x..x + dx {
                    for zz in z..z + dz { used[xx * cs + zz] = true; }
                }
            }
        }
    }

    // ── Cliff side faces ──────────────────────────────────────────────────────
    // For each direction, build a per-depth-layer mask of exposed cliff cells
    // and greedy-merge in (lateral, y) space.
    //
    // Mask layout: mask[lat * row + y]
    //   PosX / NegX: lat = z,  y = height  →  emit(dir, d=x, u=y, v=z, du, dv)
    //   PosZ / NegZ: lat = x,  y = height  →  emit(dir, d=z, u=x, v=y, du_lat, dv_y)
    let row = mh + 1;
    let mut mask = vec![None::<BlockId>; cs * row];
    let mut used = vec![false; cs * row];

    // PosX ────────────────────────────────────────────────────────────────────
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
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosX, d, y, lat, dy, dlat, chunk, block);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    // NegX ────────────────────────────────────────────────────────────────────
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
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::NegX, d, y, lat, dy, dlat, chunk, block);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    // PosZ ────────────────────────────────────────────────────────────────────
    // lat = x; emit(PosZ, d=z, u=lat=x, v=y, du=dlat, dv=dy)
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
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosZ, d, lat, y, dlat, dy, chunk, block);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    // NegZ ────────────────────────────────────────────────────────────────────
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
                emit_greedy_quad(&mut vertices, &mut indices, FaceDir::NegZ, d, lat, y, dlat, dy, chunk, block);
                for ll in lat..lat + dlat {
                    for yy in y..y + dy { used[ll * row + yy] = true; }
                }
            }
        }
    }

    (vertices, indices)
}

pub fn mesh_chunk(chunk: &Chunk, neighbors: &NeighborMasks) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Only iterate up to the highest occupied Y level — skips empty air above terrain.
    let mh = chunk.max_height + 1;

    for dir in FaceDir::ALL {
        // NegY is invisible for solid terrain, but water needs its underside
        // (visible when underwater looking up). Skipped per-block below.
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
                    // Water is emitted in a separate pass (mesh_chunk_water) so it can
                    // be drawn after all opaque geometry for correct alpha blending.
                    if dir == FaceDir::NegY { continue; }
                    let Some(model) = get_model(id) else { continue };
                    if id == WATER { continue; }
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
        // Cross-chunk boundary: use neighbour mask if available.
        // Treat an unloaded neighbour as solid so we don't render tall walls at the
        // render-distance boundary (those walls would occlude all terrain behind them).
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

// ── Quad emitters ────────────────────────────────────────────────────────────

/// Per-direction tiling UV assignment for a quad of block-unit size (s, t).
/// s = the "horizontal" extent of the face, t = the "vertical" extent.
/// Coordinates are in block units so the REPEAT sampler tiles naturally.
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
            pos: [face.verts[i][0] + ox, face.verts[i][1] + oy, face.verts[i][2] + oz],
            normal,
            uv: [face.uvs[i][0], face.uvs[i][1] + layer * 256.0],
        });
    }
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

/// Water surface mesh: per-column greedy merge so the water edge follows the
/// exact coastline. The stencil buffer (water pipeline) prevents any pixel from
/// being alpha-blended more than once, so chunk-boundary seams are impossible.
pub fn mesh_chunk_water(chunk: &Chunk, neighbors: &NeighborMasks) -> (Vec<Vertex>, Vec<u32>) {
    // Fast-exit for land chunks with no water.
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

    // Find the topmost exposed water face per XZ column.
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

    // 2D greedy merge over same-height XZ cells.
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

            emit_greedy_quad(&mut vertices, &mut indices, FaceDir::PosY, y0, x, z, dx, dz, chunk, WATER);
            // Drop 0.1 below block top so water sits below coast terrain at y+1,
            // avoiding Z-fighting where land and water share the same grid row.
            let base = vertices.len() - 4;
            for v in &mut vertices[base..] { v.pos[1] -= 0.1; }

            for xx in x..x + dx {
                for zz in z..z + dz { used[xx * cs + zz] = true; }
            }
        }
    }

    (vertices, indices)
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

    let (s, t) = match dir {
        FaceDir::PosX | FaceDir::NegX => (dv, du),
        _                              => (du, dv),
    };
    let uvs   = quad_uvs(dir, s, t);
    let layer = face_tex(block, dir) as f32;

    let base   = vertices.len() as u32;
    let normal = dir.normal();
    for (pos, uv) in quad.iter().zip(uvs.iter()) {
        vertices.push(Vertex { pos: *pos, normal, uv: [uv[0], uv[1] + layer * 256.0] });
    }
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}
