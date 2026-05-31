use std::collections::{HashMap, HashSet};
use std::sync::{mpsc::{self, Receiver, Sender}, Arc};
use std::time::Instant;

use ash::vk;
use glam::{IVec2, IVec3, Vec3};

use crate::{
    meshing::greedy::mesh_chunk,
    render::{
        context::VulkanContext,
        mesh::{GpuMesh, Vertex},
    },
    world::{
        block::{DIRT, STONE, WATER},
        chunk::{Chunk, CHUNK_SIZE},
        continents::{build_defs, SEA_LEVEL},
        neighbor::{ChunkFaceData, NeighborMasks},
        terrain::sample_chunk_heights,
    },
};

const DESTROY_LAG: usize = 3;
const REBUILD_COOLDOWN_MS: u128 = 300;

type GenResult  = (IVec2, ChunkFaceData);
type MeshResult = (IVec2, Vec<Vertex>, Vec<u32>, u64, u64);
/// Result of a background assembly: flat vertex + index buffers ready to upload.
/// Carries assembly_us so the upload log line can show the full pipeline cost.
type RebuildResult = (Vec<Vertex>, Vec<u32>, u64);

pub struct World {
    render_distance: i32,
    loaded:          HashSet<IVec2>,
    gen_in_flight:   HashSet<IVec2>,
    pending_mesh:    HashSet<IVec2>,
    mesh_in_flight:  HashSet<IVec2>,
    last_cam_chunk:  IVec2,
    face_data:       HashMap<IVec2, ChunkFaceData>,
    /// Arc so the background assembly task can hold refs without copying.
    cpu_meshes:      HashMap<IVec2, Arc<(Vec<Vertex>, Vec<u32>)>>,
    combined:        Option<GpuMesh>,
    world_dirty:     bool,
    last_rebuild:    Instant,
    /// True while a rayon assembly task is running.
    rebuild_pending: bool,
    gen_rx:    Receiver<GenResult>,
    gen_tx:    Sender<GenResult>,
    mesh_rx:   Receiver<MeshResult>,
    mesh_tx:   Sender<MeshResult>,
    rebuild_rx: Receiver<RebuildResult>,
    rebuild_tx: Sender<RebuildResult>,
    frame_idx: usize,
    deferred:  Vec<Vec<GpuMesh>>,
    stats_timer:   Instant,
    gen_spawned:   u32,
    gen_done:      u32,
    mesh_done:     u32,
    total_gen_us:      u64,
    total_mesh_us:     u64,
    max_chunk_verts:   usize,
    total_chunk_verts: usize,
}

impl World {
    pub fn new(render_distance: i32) -> Self {
        let (gen_tx,     gen_rx)     = mpsc::channel();
        let (mesh_tx,    mesh_rx)    = mpsc::channel();
        let (rebuild_tx, rebuild_rx) = mpsc::channel();
        Self {
            render_distance,
            loaded:         HashSet::new(),
            gen_in_flight:  HashSet::new(),
            pending_mesh:   HashSet::new(),
            mesh_in_flight: HashSet::new(),
            last_cam_chunk: IVec2::new(i32::MAX, i32::MAX),
            face_data:      HashMap::new(),
            cpu_meshes:     HashMap::new(),
            combined:       None,
            world_dirty:    false,
            last_rebuild:   Instant::now(),
            rebuild_pending: false,
            gen_rx, gen_tx, mesh_rx, mesh_tx, rebuild_rx, rebuild_tx,
            frame_idx: 0,
            deferred: (0..DESTROY_LAG).map(|_| Vec::new()).collect(),
            stats_timer:       Instant::now(),
            gen_spawned:       0,
            gen_done:          0,
            mesh_done:         0,
            total_gen_us:      0,
            total_mesh_us:     0,
            max_chunk_verts:   0,
            total_chunk_verts: 0,
        }
    }

    fn defer_destroy(&mut self, mesh: GpuMesh) {
        let slot = (self.frame_idx + DESTROY_LAG - 1) % DESTROY_LAG;
        self.deferred[slot].push(mesh);
    }

    fn spawn_mesh(&mut self, coord: IVec2) {
        let neighbors = build_neighbor_masks(coord, &self.face_data);
        let tx     = self.mesh_tx.clone();
        let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
        self.mesh_in_flight.insert(coord);
        rayon::spawn(move || {
            let t0 = Instant::now();
            let chunk = generate(origin);
            let gen_us = t0.elapsed().as_micros() as u64;
            let t1 = Instant::now();
            let (verts, idxs) = mesh_chunk(&chunk, &neighbors);
            let mesh_us = t1.elapsed().as_micros() as u64;
            let _ = tx.send((coord, verts, idxs, gen_us, mesh_us));
        });
    }

    pub fn update(&mut self, camera_pos: Vec3, ctx: &VulkanContext, pool: vk::CommandPool) {
        let safe_slot = self.frame_idx % DESTROY_LAG;
        for mesh in self.deferred[safe_slot].drain(..) {
            mesh.destroy(ctx);
        }
        self.frame_idx += 1;

        let cam_chunk = world_to_chunk(camera_pos);
        let rd = self.render_distance;

        // ── Stage 1 results ───────────────────────────────────────────────────
        while let Ok((coord, fd)) = self.gen_rx.try_recv() {
            self.gen_in_flight.remove(&coord);
            self.gen_done += 1;

            if (coord.x - cam_chunk.x).abs() > rd || (coord.y - cam_chunk.y).abs() > rd { continue; }

            self.face_data.insert(coord, fd);

            if can_mesh(coord, &self.gen_in_flight) {
                self.spawn_mesh(coord);
            } else {
                self.pending_mesh.insert(coord);
            }

            let unblocked: Vec<IVec2> = [IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y]
                .iter().map(|&d| coord + d)
                .filter(|nb| self.pending_mesh.contains(nb) && can_mesh(*nb, &self.gen_in_flight))
                .collect();
            for nb in unblocked {
                self.pending_mesh.remove(&nb);
                self.spawn_mesh(nb);
            }
        }

        if self.gen_in_flight.is_empty() && !self.pending_mesh.is_empty() {
            let stuck: Vec<IVec2> = self.pending_mesh.iter().copied().collect();
            for coord in stuck {
                self.pending_mesh.remove(&coord);
                if !self.mesh_in_flight.contains(&coord) && !self.loaded.contains(&coord) {
                    self.spawn_mesh(coord);
                }
            }
        }

        // ── Stage 2 results ───────────────────────────────────────────────────
        while let Ok((coord, verts, idxs, gen_us, mesh_us)) = self.mesh_rx.try_recv() {
            self.mesh_in_flight.remove(&coord);
            self.mesh_done += 1;
            self.total_gen_us  += gen_us;
            self.total_mesh_us += mesh_us;
            let nv = verts.len();
            self.total_chunk_verts += nv;
            if nv > self.max_chunk_verts { self.max_chunk_verts = nv; }

            if (coord.x - cam_chunk.x).abs() > rd || (coord.y - cam_chunk.y).abs() > rd { continue; }

            self.loaded.insert(coord);
            if !verts.is_empty() {
                self.cpu_meshes.insert(coord, Arc::new((verts, idxs)));
                self.world_dirty = true;
            }
        }

        // ── Periodic stats ────────────────────────────────────────────────────
        let settled = self.gen_in_flight.is_empty()
            && self.pending_mesh.is_empty()
            && self.mesh_in_flight.is_empty();
        if !settled && self.stats_timer.elapsed().as_millis() >= 500 {
            self.stats_timer = Instant::now();
            let n = self.mesh_done.max(1) as u64;
            println!(
                "[chunks] gen {}/{}  mesh {}/{}  fly {} pend {} mfly {}  \
                 avg_gen {}us avg_mesh {}us  max_verts {} avg_verts {}",
                self.gen_done, self.gen_spawned,
                self.mesh_done, self.gen_spawned,
                self.gen_in_flight.len(), self.pending_mesh.len(), self.mesh_in_flight.len(),
                self.total_gen_us / n, self.total_mesh_us / n,
                self.max_chunk_verts,
                self.total_chunk_verts / self.mesh_done.max(1) as usize,
            );
        }

        // ── Receive completed background assembly → GPU upload ────────────────
        if let Ok((all_verts, all_idxs, assemble_us)) = self.rebuild_rx.try_recv() {
            self.rebuild_pending = false;

            if let Some(old) = self.combined.take() {
                self.defer_destroy(old);
            }
            if !all_verts.is_empty() {
                let t = Instant::now();
                if let Ok(mesh) = GpuMesh::from_data_device_local(&all_verts, &all_idxs, ctx, pool) {
                    let upload_us = t.elapsed().as_micros() as u64;
                    println!(
                        "[mesh] verts={}  assemble={}us  upload={}us  total={}ms",
                        all_verts.len(), assemble_us, upload_us,
                        (assemble_us + upload_us) / 1000,
                    );
                    self.combined = Some(mesh);
                }
            }
        }

        // ── Trigger background assembly ───────────────────────────────────────
        // Also re-run when camera has been still long enough to earn a VRAM promotion.
        let cooldown_ok    = self.last_rebuild.elapsed().as_millis() >= REBUILD_COOLDOWN_MS;
        let should_rebuild = self.world_dirty && cooldown_ok;

        if should_rebuild && !self.rebuild_pending {
            self.world_dirty     = false;
            self.last_rebuild    = Instant::now();
            self.rebuild_pending = true;

            // Snapshot Arc refs — O(chunks), not O(vertices).
            let snapshot: Vec<Arc<(Vec<Vertex>, Vec<u32>)>> =
                self.cpu_meshes.values().cloned().collect();
            let tx = self.rebuild_tx.clone();

            rayon::spawn(move || {
                let t = Instant::now();
                let mut all_verts: Vec<Vertex> = Vec::new();
                let mut all_idxs:  Vec<u32>    = Vec::new();
                for chunk in &snapshot {
                    let base = all_verts.len() as u32;
                    all_verts.extend_from_slice(&chunk.0);
                    all_idxs.extend(chunk.1.iter().map(|&i| i + base));
                }
                let _ = tx.send((all_verts, all_idxs, t.elapsed().as_micros() as u64));
            });
        }

        // ── O(RD²) load/unload scan on chunk boundary ─────────────────────────
        if cam_chunk != self.last_cam_chunk {
            self.last_cam_chunk = cam_chunk;

            let before = self.loaded.len();
            self.loaded.retain(|c| (c.x - cam_chunk.x).abs() <= rd && (c.y - cam_chunk.y).abs() <= rd);
            if self.loaded.len() != before {
                self.cpu_meshes.retain(|c, _| self.loaded.contains(c));
                self.world_dirty = true;
            }

            self.face_data.retain(|c, _| (c.x - cam_chunk.x).abs() <= rd + 2 && (c.y - cam_chunk.y).abs() <= rd + 2);
            self.pending_mesh.retain(|c| (c.x - cam_chunk.x).abs() <= rd && (c.y - cam_chunk.y).abs() <= rd);
            self.gen_in_flight.retain(|c| (c.x - cam_chunk.x).abs() <= rd + 2 && (c.y - cam_chunk.y).abs() <= rd + 2);
            self.mesh_in_flight.retain(|c| (c.x - cam_chunk.x).abs() <= rd + 2 && (c.y - cam_chunk.y).abs() <= rd + 2);

            let unblocked: Vec<IVec2> = self.pending_mesh.iter()
                .filter(|&&c| can_mesh(c, &self.gen_in_flight))
                .copied().collect();
            for coord in unblocked {
                self.pending_mesh.remove(&coord);
                if !self.mesh_in_flight.contains(&coord) { self.spawn_mesh(coord); }
            }

            let mut to_spawn: Vec<(IVec2, i32)> = Vec::new();
            for cx in (cam_chunk.x - rd)..=(cam_chunk.x + rd) {
                for cz in (cam_chunk.y - rd)..=(cam_chunk.y + rd) {
                    let coord = IVec2::new(cx, cz);
                    if !self.loaded.contains(&coord)
                        && !self.gen_in_flight.contains(&coord)
                        && !self.pending_mesh.contains(&coord)
                        && !self.mesh_in_flight.contains(&coord)
                    {
                        let dx = cx - cam_chunk.x;
                        let dz = cz - cam_chunk.y;
                        to_spawn.push((coord, dx * dx + dz * dz));
                    }
                }
            }
            to_spawn.sort_unstable_by_key(|&(_, d)| d);

            if !to_spawn.is_empty() && self.gen_in_flight.is_empty() {
                self.stats_timer      = Instant::now();
                self.gen_done         = 0;
                self.mesh_done        = 0;
                self.gen_spawned      = 0;
                self.total_gen_us     = 0;
                self.total_mesh_us    = 0;
                self.max_chunk_verts  = 0;
                self.total_chunk_verts = 0;
            }
            for (coord, _) in to_spawn {
                self.gen_in_flight.insert(coord);
                self.gen_spawned += 1;
                let tx     = self.gen_tx.clone();
                let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
                rayon::spawn(move || {
                    let chunk = generate(origin);
                    let fd    = ChunkFaceData::extract(&chunk);
                    let _ = tx.send((coord, fd));
                });
            }
        }
    }

    pub fn iter_meshes(&self) -> impl Iterator<Item = &GpuMesh> {
        self.combined.iter()
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        for bucket in &mut self.deferred {
            for mesh in bucket.drain(..) { mesh.destroy(ctx); }
        }
        if let Some(mesh) = self.combined.take() { mesh.destroy(ctx); }
    }
}

fn can_mesh(coord: IVec2, gen_in_flight: &HashSet<IVec2>) -> bool {
    [IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y]
        .iter().all(|&d| !gen_in_flight.contains(&(coord + d)))
}

fn build_neighbor_masks(coord: IVec2, face_data: &HashMap<IVec2, ChunkFaceData>) -> NeighborMasks {
    NeighborMasks {
        pos_x: face_data.get(&(coord + IVec2::X)).map(|d| d.neg_x),
        neg_x: face_data.get(&(coord - IVec2::X)).map(|d| d.pos_x),
        pos_z: face_data.get(&(coord + IVec2::Y)).map(|d| d.neg_z),
        neg_z: face_data.get(&(coord - IVec2::Y)).map(|d| d.pos_z),
    }
}

fn world_to_chunk(pos: Vec3) -> IVec2 {
    IVec2::new(
        pos.x.floor() as i32 / CHUNK_SIZE as i32,
        pos.z.floor() as i32 / CHUNK_SIZE as i32,
    )
}

fn generate(origin: IVec3) -> Chunk {
    let mut chunk    = Chunk::new(origin);
    let defs         = build_defs(origin.x, origin.z);
    let mut surface  = vec![0u16;  CHUNK_SIZE * CHUNK_SIZE];
    let mut is_ocean = vec![false; CHUNK_SIZE * CHUNK_SIZE];
    sample_chunk_heights(&defs, origin.x, origin.z, &mut surface, &mut is_ocean);

    const STONE_DEPTH: usize = 4;
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = x + z * CHUNK_SIZE;
            let sy  = surface[idx] as usize;
            if is_ocean[idx] {
                chunk.set(x, sy, z, STONE);
                for y in (sy + 1)..=SEA_LEVEL { chunk.set(x, y, z, WATER); }
            } else {
                for y in sy.saturating_sub(STONE_DEPTH)..sy { chunk.set(x, y, z, STONE); }
                chunk.set(x, sy, z, DIRT);
            }
        }
    }
    chunk
}
