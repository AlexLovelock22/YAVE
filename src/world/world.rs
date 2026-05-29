use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
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
        block::{DIRT, STONE},
        chunk::{Chunk, CHUNK_SIZE},
        neighbor::{ChunkFaceData, NeighborMasks},
    },
};

const DESTROY_LAG: usize = 3;
const REBUILD_INTERVAL: usize = 30;
const TERRAIN_HEIGHT: usize = 48;

/// Stage 1: generate block data and extract face mask. No mesh.
type GenResult = (IVec2, ChunkFaceData);
/// Stage 2: fully meshed chunk.
type MeshResult = (IVec2, Vec<Vertex>, Vec<u32>);

pub struct World {
    render_distance: i32,
    /// Fully meshed and uploaded chunks.
    loaded: HashSet<IVec2>,
    /// Stage 1 (generation) in progress.
    gen_in_flight: HashSet<IVec2>,
    /// Stage 1 done; waiting for all in-range neighbours to also finish Stage 1.
    pending_mesh: HashSet<IVec2>,
    /// Stage 2 (meshing) in progress.
    mesh_in_flight: HashSet<IVec2>,
    last_cam_chunk: IVec2,
    face_data: HashMap<IVec2, ChunkFaceData>,
    cpu_meshes: HashMap<IVec2, (Vec<Vertex>, Vec<u32>)>,
    combined: Option<GpuMesh>,
    world_dirty: bool,
    gen_rx: Receiver<GenResult>,
    gen_tx: Sender<GenResult>,
    mesh_rx: Receiver<MeshResult>,
    mesh_tx: Sender<MeshResult>,
    frame_idx: usize,
    deferred: Vec<Vec<GpuMesh>>,
    stats_timer: Instant,
    gen_spawned: u32,
    gen_done: u32,
    mesh_done: u32,
}

impl World {
    pub fn new(render_distance: i32) -> Self {
        let (gen_tx, gen_rx) = mpsc::channel();
        let (mesh_tx, mesh_rx) = mpsc::channel();
        Self {
            render_distance,
            loaded: HashSet::new(),
            gen_in_flight: HashSet::new(),
            pending_mesh: HashSet::new(),
            mesh_in_flight: HashSet::new(),
            last_cam_chunk: IVec2::new(i32::MAX, i32::MAX),
            face_data: HashMap::new(),
            cpu_meshes: HashMap::new(),
            combined: None,
            world_dirty: false,
            gen_rx,
            gen_tx,
            mesh_rx,
            mesh_tx,
            frame_idx: 0,
            deferred: (0..DESTROY_LAG).map(|_| Vec::new()).collect(),
            stats_timer: Instant::now(),
            gen_spawned: 0,
            gen_done: 0,
            mesh_done: 0,
        }
    }

    fn defer_destroy(&mut self, mesh: GpuMesh) {
        let slot = (self.frame_idx + DESTROY_LAG - 1) % DESTROY_LAG;
        self.deferred[slot].push(mesh);
    }

    /// Spawn Stage 2 for a chunk whose Stage 1 is done and all neighbours are ready.
    fn spawn_mesh(&mut self, coord: IVec2) {
        let neighbors = build_neighbor_masks(coord, &self.face_data);
        let tx = self.mesh_tx.clone();
        let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
        self.mesh_in_flight.insert(coord);
        rayon::spawn(move || {
            let chunk = generate(origin);
            let (verts, idxs) = mesh_chunk(&chunk, &neighbors);
            let _ = tx.send((coord, verts, idxs));
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

        // ── Stage 1 results: face data ready ─────────────────────────────────
        while let Ok((coord, fd)) = self.gen_rx.try_recv() {
            self.gen_in_flight.remove(&coord);
            self.gen_done += 1;

            let dx = (coord.x - cam_chunk.x).abs();
            let dz = (coord.y - cam_chunk.y).abs();
            if dx > rd || dz > rd { continue; }

            self.face_data.insert(coord, fd);

            // Try to advance coord to Stage 2
            if can_mesh(coord, &self.gen_in_flight) {
                self.spawn_mesh(coord);
            } else {
                self.pending_mesh.insert(coord);
            }

            // Coord's completion may unblock neighbours that were waiting on it
            let unblocked: Vec<IVec2> = [IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y]
                .iter()
                .map(|&d| coord + d)
                .filter(|nb| self.pending_mesh.contains(nb) && can_mesh(*nb, &self.gen_in_flight))
                .collect();
            for nb in unblocked {
                self.pending_mesh.remove(&nb);
                self.spawn_mesh(nb);
            }
        }

        // ── Stage 2 results: mesh ready ───────────────────────────────────────
        while let Ok((coord, verts, idxs)) = self.mesh_rx.try_recv() {
            self.mesh_in_flight.remove(&coord);
            self.mesh_done += 1;

            let dx = (coord.x - cam_chunk.x).abs();
            let dz = (coord.y - cam_chunk.y).abs();
            if dx > rd || dz > rd { continue; }

            self.loaded.insert(coord);
            if !verts.is_empty() {
                self.cpu_meshes.insert(coord, (verts, idxs));
                self.world_dirty = true;
            }
        }

        // ── Periodic stats ────────────────────────────────────────────────────
        let busy = !self.gen_in_flight.is_empty()
            || !self.pending_mesh.is_empty()
            || !self.mesh_in_flight.is_empty();
        if busy && self.stats_timer.elapsed().as_millis() >= 500 {
            self.stats_timer = Instant::now();
            println!(
                "[chunks] gen {}/{}  mesh {}/{}  gen_flight {}  pending {}  mesh_flight {}",
                self.gen_done, self.gen_spawned,
                self.mesh_done, self.gen_spawned,
                self.gen_in_flight.len(), self.pending_mesh.len(),
                self.mesh_in_flight.len(),
            );
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

            // Unloading chunks may remove gen_in_flight entries that were blocking
            // pending_mesh chunks — re-check for newly unblocked ones.
            let unblocked: Vec<IVec2> = self.pending_mesh
                .iter()
                .filter(|&&c| can_mesh(c, &self.gen_in_flight))
                .copied()
                .collect();
            for coord in unblocked {
                self.pending_mesh.remove(&coord);
                if !self.mesh_in_flight.contains(&coord) {
                    self.spawn_mesh(coord);
                }
            }

            // Spawn Stage 1 for new chunks, closest first
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
                self.stats_timer = Instant::now();
                self.gen_done = 0;
                self.mesh_done = 0;
                self.gen_spawned = 0;
            }
            for (coord, _) in to_spawn {
                self.gen_in_flight.insert(coord);
                self.gen_spawned += 1;
                let tx = self.gen_tx.clone();
                let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
                rayon::spawn(move || {
                    let chunk = generate(origin);
                    let fd = ChunkFaceData::extract(&chunk);
                    let _ = tx.send((coord, fd));
                });
            }
        }

        // ── Rebuild combined mesh ─────────────────────────────────────────────
        // During loading: HOST_VISIBLE upload (no queue_wait_idle stall, slightly slower GPU reads).
        // On final settle: DEVICE_LOCAL upload (one stall, then fast GPU reads forever after).
        let settled = self.gen_in_flight.is_empty()
            && self.pending_mesh.is_empty()
            && self.mesh_in_flight.is_empty();
        if self.world_dirty && (settled || self.frame_idx % REBUILD_INTERVAL == 0) {
            self.world_dirty = false;
            let mut all_verts: Vec<Vertex> = Vec::new();
            let mut all_idxs: Vec<u32> = Vec::new();
            for (verts, idxs) in self.cpu_meshes.values() {
                let base = all_verts.len() as u32;
                all_verts.extend_from_slice(verts);
                all_idxs.extend(idxs.iter().map(|&i| i + base));
            }
            if let Some(old) = self.combined.take() {
                self.defer_destroy(old);
            }
            if !all_verts.is_empty() {
                let mesh = if settled {
                    GpuMesh::from_data_device_local(&all_verts, &all_idxs, ctx, pool)
                } else {
                    GpuMesh::from_data(&all_verts, &all_idxs, ctx, pool)
                };
                if let Ok(mesh) = mesh {
                    self.combined = Some(mesh);
                }
            }
        }
    }

    pub fn iter_meshes(&self) -> impl Iterator<Item = &GpuMesh> {
        self.combined.iter()
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        for bucket in &mut self.deferred {
            for mesh in bucket.drain(..) {
                mesh.destroy(ctx);
            }
        }
        if let Some(mesh) = self.combined.take() {
            mesh.destroy(ctx);
        }
    }
}

/// A chunk can proceed to Stage 2 only when none of its 4 neighbours are still
/// in Stage 1 (gen_in_flight). Out-of-range neighbours are never in gen_in_flight,
/// so edge chunks unblock naturally.
fn can_mesh(coord: IVec2, gen_in_flight: &HashSet<IVec2>) -> bool {
    for &dir in &[IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y] {
        if gen_in_flight.contains(&(coord + dir)) {
            return false;
        }
    }
    true
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
    let mut chunk = Chunk::new(origin);
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            for y in 0..TERRAIN_HEIGHT - 1 {
                chunk.set(x, y, z, STONE);
            }
            chunk.set(x, TERRAIN_HEIGHT - 1, z, DIRT);
        }
    }
    chunk
}
