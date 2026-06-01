use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::{mpsc::{self, Receiver, Sender}, Arc};
use std::time::Instant;

use ash::vk;
use glam::{IVec2, IVec3, Vec3};

use crate::{
    meshing::greedy::mesh_chunk,
    render::{
        buffer::create_buffer,
        context::VulkanContext,
        mesh::{GpuMesh, PendingMeshUpload, Vertex},
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
/// Rayon writes directly into the persistent staging buffers and sends back the byte counts.
type RebuildResult = (usize, usize, u64); // (vb_bytes, ib_bytes, assemble_us)

/// Per-chunk draw params to commit when a pending upload completes.
enum DrawUpdate {
    /// Replace chunk_draws entirely (full rebuild changed the layout).
    Full(Vec<(IVec2, u32, u32)>),
    /// Append new entries (incremental builds only add to the end).
    Incremental(Vec<(IVec2, u32, u32)>),
}

/// Persistently mapped HOST_VISIBLE staging buffer reused across rebuilds.
/// Eliminates per-rebuild vkAllocateMemory + vkMapMemory overhead.
struct StagingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr:    *mut u8,
    cap:    usize,
}

/// Wraps a raw pointer so it can be sent to rayon tasks.
/// Safety: we guarantee exclusive access while rayon writes (pending_upload guards DMA,
/// rebuild_pending guards concurrent rayon tasks).
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
impl SendPtr {
    // Method rather than field access so closures capture `SendPtr` (Send),
    // not `*mut u8` (!Send) via Rust 2021 field-projection capture.
    fn get(&self) -> *mut u8 { self.0 }
}

impl StagingBuffer {
    fn new(ctx: &VulkanContext, cap: usize) -> anyhow::Result<Self> {
        let (buffer, memory) = create_buffer(
            ctx, cap as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ptr = unsafe {
            ctx.device.map_memory(memory, 0, cap as vk::DeviceSize, vk::MemoryMapFlags::empty())? as *mut u8
        };
        Ok(Self { buffer, memory, ptr, cap })
    }

    fn ensure_cap(&mut self, ctx: &VulkanContext, needed: usize) -> anyhow::Result<()> {
        if needed <= self.cap { return Ok(()); }
        self.destroy(ctx);
        let new_cap = needed.next_power_of_two().max(needed);
        *self = Self::new(ctx, new_cap)?;
        Ok(())
    }

    fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.unmap_memory(self.memory);
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

/// Persistent DEVICE_LOCAL render buffer — allocated once, grown only when the mesh
/// outgrows it. Eliminates per-rebuild vkAllocateMemory on the render thread.
struct DstBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    cap:    usize,
}

impl DstBuffer {
    fn new(ctx: &VulkanContext, cap: usize, usage: vk::BufferUsageFlags) -> anyhow::Result<Self> {
        let (buffer, memory) = create_buffer(
            ctx, cap as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        Ok(Self { buffer, memory, cap })
    }

    fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

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
    /// Dedicated command pool for async mesh transfers (kept alive until World is destroyed).
    transfer_pool: Option<vk::CommandPool>,
    /// In-flight GPU upload — rendering continues with the old mesh until the fence signals.
    pending_upload: Option<PendingMeshUpload>,
    /// Persistently mapped staging buffers; rayon writes into these directly.
    staging_vb: Option<StagingBuffer>,
    staging_ib: Option<StagingBuffer>,
    /// Double-buffered DEVICE_LOCAL render buffers.
    /// Uploads always write to dst_[vb|ib][dst_back]; combined reads the other slot.
    /// The two slots are distinct memory — no DMA/render overlap.
    dst_vb:   [Option<DstBuffer>; 2],
    dst_ib:   [Option<DstBuffer>; 2],
    dst_back: usize,
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
    /// Chunks whose mesh just arrived and haven't been written to staging yet.
    dirty_new:            Vec<IVec2>,
    /// When true, staging content is stale (chunks removed) — must full-rebuild.
    staging_full_rebuild: bool,
    /// Byte offset of the next free position in the persistent staging buffers.
    staging_vb_used: usize,
    staging_ib_used: usize,
    /// Per-chunk draw parameters for the CURRENT committed GPU layout.
    /// Stored as a Vec for sequential (cache-friendly) iteration in cull_draws.
    chunk_draws: Vec<(IVec2, u32, u32)>,
    /// Draw params for the in-flight upload; committed to chunk_draws when the fence fires.
    pending_draws: Option<DrawUpdate>,
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
            transfer_pool: None,
            pending_upload: None,
            staging_vb: None,
            staging_ib: None,
            dst_vb:   [None, None],
            dst_ib:   [None, None],
            dst_back: 0,
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
            dirty_new:            Vec::new(),
            staging_full_rebuild: false,
            staging_vb_used:      0,
            staging_ib_used:      0,
            chunk_draws:          Vec::new(),
            pending_draws:        None,
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
                self.dirty_new.push(coord);
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

        // ── Poll async GPU upload fence → swap buffers when done ─────────────
        if let Some(ref upload) = self.pending_upload {
            if upload.is_ready(ctx) {
                let upload      = self.pending_upload.take().unwrap();
                let tpool       = self.transfer_pool.unwrap();
                let index_count = upload.index_count;
                let _ = upload.into_mesh(ctx, tpool);
                let back = self.dst_back;
                self.combined = Some(GpuMesh::view(
                    self.dst_vb[back].as_ref().unwrap().buffer,
                    self.dst_ib[back].as_ref().unwrap().buffer,
                    index_count,
                ));
                self.dst_back = 1 - back;
                // Commit the draw params that correspond to this new layout.
                match self.pending_draws.take() {
                    Some(DrawUpdate::Full(draws))       => { self.chunk_draws = draws; }
                    Some(DrawUpdate::Incremental(adds)) => { self.chunk_draws.extend(adds); }
                    None => {}
                }
                let total_verts = self.staging_vb_used / size_of::<Vertex>();
                println!("[mesh] swapped  total_verts={total_verts}  chunks={}", self.chunk_draws.len());
            }
        }

        // ── Receive completed rayon write → submit async GPU copy ────────────
        if let Ok((vb_bytes, ib_bytes, assemble_us)) = self.rebuild_rx.try_recv() {
            self.rebuild_pending = false;
            self.staging_vb_used = vb_bytes;
            self.staging_ib_used = ib_bytes;
            if vb_bytes > 0 {
                let tpool = *self.transfer_pool.get_or_insert_with(|| unsafe {
                    ctx.device.create_command_pool(&vk::CommandPoolCreateInfo {
                        flags: vk::CommandPoolCreateFlags::TRANSIENT,
                        queue_family_index: ctx.graphics_family,
                        ..Default::default()
                    }, None).expect("transfer pool")
                });

                // Write to the back slot only; front slot is being rendered, never touched here.
                let back = self.dst_back;
                let need_grow_vb = self.dst_vb[back].as_ref().map_or(true, |b| vb_bytes > b.cap);
                let need_grow_ib = self.dst_ib[back].as_ref().map_or(true, |b| ib_bytes > b.cap);

                if need_grow_vb {
                    let cap = vb_bytes.next_power_of_two();
                    if let Some(old) = self.dst_vb[back].take() {
                        // Slot was previously the front (rendered). Defer-destroy for GPU safety.
                        self.defer_destroy(GpuMesh {
                            vertex_buffer: old.buffer, vertex_memory: old.memory,
                            index_buffer:  vk::Buffer::null(), index_memory: vk::DeviceMemory::null(),
                            index_count:   0,
                        });
                    }
                    match DstBuffer::new(ctx, cap, vk::BufferUsageFlags::VERTEX_BUFFER) {
                        Ok(b) => self.dst_vb[back] = Some(b),
                        Err(e) => { eprintln!("[mesh] dst_vb[{back}] grow: {e}"); return; }
                    }
                }
                if need_grow_ib {
                    let cap = ib_bytes.next_power_of_two();
                    if let Some(old) = self.dst_ib[back].take() {
                        self.defer_destroy(GpuMesh {
                            vertex_buffer: vk::Buffer::null(), vertex_memory: old.memory,
                            index_buffer:  old.buffer, index_memory: vk::DeviceMemory::null(),
                            index_count:   0,
                        });
                    }
                    match DstBuffer::new(ctx, cap, vk::BufferUsageFlags::INDEX_BUFFER) {
                        Ok(b) => self.dst_ib[back] = Some(b),
                        Err(e) => { eprintln!("[mesh] dst_ib[{back}] grow: {e}"); return; }
                    }
                }

                let staging_vb = self.staging_vb.as_ref().unwrap().buffer;
                let staging_ib = self.staging_ib.as_ref().unwrap().buffer;
                let dst_vb     = self.dst_vb[back].as_ref().unwrap().buffer;
                let dst_ib     = self.dst_ib[back].as_ref().unwrap().buffer;
                let t = Instant::now();
                match GpuMesh::begin_copy_to_preallocated(
                    staging_vb, vb_bytes,
                    staging_ib, ib_bytes,
                    dst_vb, dst_ib,
                    ctx, tpool,
                ) {
                    Ok(upload) => {
                        println!("[mesh] assemble={}us  submit={}us  verts={}",
                            assemble_us, t.elapsed().as_micros(),
                            vb_bytes / size_of::<Vertex>());
                        self.pending_upload = Some(upload);
                    }
                    Err(e) => eprintln!("[mesh] copy submit failed: {e}"),
                }
            }
        }

        // ── Trigger background assembly ───────────────────────────────────────
        let cooldown_ok    = self.last_rebuild.elapsed().as_millis() >= REBUILD_COOLDOWN_MS;
        // Don't start a new rebuild while DMA is still reading from staging.
        let should_rebuild = self.world_dirty && cooldown_ok && self.pending_upload.is_none();

        if should_rebuild && !self.rebuild_pending {
            // First rebuild or chunks were removed → must rewrite all of staging.
            let do_full = self.staging_full_rebuild || self.staging_vb_used == 0;

            if do_full {
                let vb_needed: usize = self.cpu_meshes.values().map(|c| c.0.len()).sum::<usize>() * size_of::<Vertex>();
                let ib_needed: usize = self.cpu_meshes.values().map(|c| c.1.len()).sum::<usize>() * size_of::<u32>();

                let vb_ok = self.staging_vb.get_or_insert_with(|| {
                    StagingBuffer::new(ctx, vb_needed.next_power_of_two()).expect("staging vb alloc")
                }).ensure_cap(ctx, vb_needed).is_ok();
                let ib_ok = self.staging_ib.get_or_insert_with(|| {
                    StagingBuffer::new(ctx, ib_needed.next_power_of_two()).expect("staging ib alloc")
                }).ensure_cap(ctx, ib_needed).is_ok();

                if vb_ok && ib_ok {
                    self.world_dirty          = false;
                    self.staging_full_rebuild = false;
                    self.last_rebuild         = Instant::now();
                    self.rebuild_pending      = true;
                    self.dirty_new.clear();

                    // Snapshot preserves a stable write order; we derive per-chunk draw params
                    // from that order on the main thread so rayon just does the memcpy work.
                    let snapshot: Vec<(IVec2, Arc<(Vec<Vertex>, Vec<u32>)>)> =
                        self.cpu_meshes.iter().map(|(&k, v)| (k, v.clone())).collect();

                    let mut new_draws = Vec::with_capacity(snapshot.len());
                    {
                        let mut first_idx = 0u32;
                        for (coord, mesh) in &snapshot {
                            let ic = mesh.1.len() as u32;
                            new_draws.push((*coord, first_idx, ic));
                            first_idx += ic;
                        }
                    }
                    self.pending_draws = Some(DrawUpdate::Full(new_draws));

                    let snapshot_data: Vec<Arc<(Vec<Vertex>, Vec<u32>)>> =
                        snapshot.into_iter().map(|(_, m)| m).collect();
                    let tx     = self.rebuild_tx.clone();
                    let vb_ptr = SendPtr(self.staging_vb.as_ref().unwrap().ptr);
                    let ib_ptr = SendPtr(self.staging_ib.as_ref().unwrap().ptr);

                    rayon::spawn(move || {
                        let t = Instant::now();
                        let mut vb_off      = 0usize;
                        let mut ib_off      = 0usize;
                        let mut base_vertex = 0u32;
                        for chunk in &snapshot_data {
                            let (verts, idxs) = chunk.as_ref();
                            let v_bytes = verts.len() * size_of::<Vertex>();
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    verts.as_ptr() as *const u8,
                                    vb_ptr.get().add(vb_off),
                                    v_bytes,
                                );
                            }
                            vb_off += v_bytes;
                            let idx_dst = unsafe { ib_ptr.get().add(ib_off) as *mut u32 };
                            for (j, &i) in idxs.iter().enumerate() {
                                unsafe { *idx_dst.add(j) = i + base_vertex; }
                            }
                            ib_off      += idxs.len() * size_of::<u32>();
                            base_vertex += verts.len() as u32;
                        }
                        let _ = tx.send((vb_off, ib_off, t.elapsed().as_micros() as u64));
                    });
                }
            } else {
                // Incremental: only write the newly arrived chunks, leave old staging data untouched.
                let mut cur_vb = self.staging_vb_used;
                let mut cur_ib = self.staging_ib_used;
                let mut cur_bv = (self.staging_vb_used / size_of::<Vertex>()) as u32;

                let dirty = &self.dirty_new;
                let cpu   = &self.cpu_meshes;
                // Include coord so we can compute draw params before handing off to rayon.
                let to_append: Vec<(IVec2, Arc<(Vec<Vertex>, Vec<u32>)>, usize, usize, u32)> =
                    dirty.iter()
                        .filter_map(|&c| cpu.get(&c).map(|m| {
                            let entry = (c, m.clone(), cur_vb, cur_ib, cur_bv);
                            cur_vb += m.0.len() * size_of::<Vertex>();
                            cur_ib += m.1.len() * size_of::<u32>();
                            cur_bv += m.0.len() as u32;
                            entry
                        }))
                        .collect();

                let new_vb_total = cur_vb;
                let new_ib_total = cur_ib;

                // If staging needs to grow the old data is gone → fall back to full rebuild.
                let needs_vb_realloc = self.staging_vb.as_ref().map_or(true, |s| new_vb_total > s.cap);
                let needs_ib_realloc = self.staging_ib.as_ref().map_or(true, |s| new_ib_total > s.cap);

                if needs_vb_realloc || needs_ib_realloc {
                    self.staging_full_rebuild = true;
                } else if to_append.is_empty() {
                    // All dirty_new chunks were unloaded before the rebuild fired.
                    self.world_dirty = false;
                    self.dirty_new.clear();
                } else {
                    self.staging_vb_used  = new_vb_total;
                    self.staging_ib_used  = new_ib_total;
                    self.world_dirty      = false;
                    self.last_rebuild     = Instant::now();
                    self.rebuild_pending  = true;
                    self.dirty_new.clear();

                    // Compute per-chunk draw params for the new chunks.
                    let inc_draws: Vec<(IVec2, u32, u32)> = to_append.iter()
                        .map(|(coord, mesh, _, ib_off, _)| {
                            let fi = (*ib_off / size_of::<u32>()) as u32;
                            let ic = mesh.1.len() as u32;
                            (*coord, fi, ic)
                        })
                        .collect();
                    self.pending_draws = Some(DrawUpdate::Incremental(inc_draws));

                    // Strip coord before handing to rayon (it only needs the data + offsets).
                    let rayon_work: Vec<(Arc<(Vec<Vertex>, Vec<u32>)>, usize, usize, u32)> =
                        to_append.into_iter().map(|(_, m, a, b, c)| (m, a, b, c)).collect();

                    let tx     = self.rebuild_tx.clone();
                    let vb_ptr = SendPtr(self.staging_vb.as_ref().unwrap().ptr);
                    let ib_ptr = SendPtr(self.staging_ib.as_ref().unwrap().ptr);

                    rayon::spawn(move || {
                        let t = Instant::now();
                        for (mesh, vb_off, ib_off, base_v) in &rayon_work {
                            let (verts, idxs) = mesh.as_ref();
                            let v_bytes = verts.len() * size_of::<Vertex>();
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    verts.as_ptr() as *const u8,
                                    vb_ptr.get().add(*vb_off),
                                    v_bytes,
                                );
                            }
                            let idx_dst = unsafe { ib_ptr.get().add(*ib_off) as *mut u32 };
                            for (j, &i) in idxs.iter().enumerate() {
                                unsafe { *idx_dst.add(j) = i + base_v; }
                            }
                        }
                        let _ = tx.send((new_vb_total, new_ib_total, t.elapsed().as_micros() as u64));
                    });
                }
            }
        }

        // ── O(RD²) load/unload scan on chunk boundary ─────────────────────────
        if cam_chunk != self.last_cam_chunk {
            self.last_cam_chunk = cam_chunk;

            let before = self.loaded.len();
            self.loaded.retain(|c| (c.x - cam_chunk.x).abs() <= rd && (c.y - cam_chunk.y).abs() <= rd);
            if self.loaded.len() != before {
                self.cpu_meshes.retain(|c, _| self.loaded.contains(c));
                self.chunk_draws.retain(|(c, _, _)| self.loaded.contains(c));
                self.staging_full_rebuild = true;
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

    /// Combined vertex and index buffer handles for the current committed GPU layout.
    pub fn render_buffers(&self) -> Option<(vk::Buffer, vk::Buffer)> {
        self.combined.as_ref().map(|m| (m.vertex_buffer, m.index_buffer))
    }

    /// Fills `out` with (first_index, index_count) for every chunk whose AABB passes
    /// all 6 frustum planes.  Clears `out` first so the caller can reuse the allocation.
    pub fn cull_draws(&self, planes: &[[f32; 4]; 6], out: &mut Vec<(u32, u32)>) {
        out.clear();
        for &(coord, fi, ic) in &self.chunk_draws {
            if chunk_in_frustum(coord, planes) {
                out.push((fi, ic));
            }
        }
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        for bucket in &mut self.deferred {
            for mesh in bucket.drain(..) { mesh.destroy(ctx); }
        }
        if let Some(upload) = self.pending_upload.take() {
            if let Some(pool) = self.transfer_pool {
                upload.abort(ctx, pool); // blocks briefly to let GPU finish
            }
        }
        // combined is a non-owning view; no need to destroy it.
        self.combined = None;
        if let Some(pool) = self.transfer_pool.take() {
            unsafe { ctx.device.destroy_command_pool(pool, None); }
        }
        if let Some(s) = self.staging_vb.take() { s.destroy(ctx); }
        if let Some(s) = self.staging_ib.take() { s.destroy(ctx); }
        for slot in &mut self.dst_vb { if let Some(b) = slot.take() { b.destroy(ctx); } }
        for slot in &mut self.dst_ib { if let Some(b) = slot.take() { b.destroy(ctx); } }
    }
}

/// AABB frustum test for a single chunk.  Returns false if the chunk is entirely
/// outside any of the 6 planes (p-vertex / positive-vertex test).
fn chunk_in_frustum(coord: IVec2, planes: &[[f32; 4]; 6]) -> bool {
    let min_x = coord.x as f32 * CHUNK_SIZE as f32;
    let min_z = coord.y as f32 * CHUNK_SIZE as f32;
    let max_x = min_x + CHUNK_SIZE as f32;
    let max_z = min_z + CHUNK_SIZE as f32;
    for &[a, b, c, d] in planes {
        let px = if a >= 0.0 { max_x } else { min_x };
        let py = if b >= 0.0 { 256.0_f32 } else { 0.0_f32 };
        let pz = if c >= 0.0 { max_z } else { min_z };
        if a * px + b * py + c * pz + d < 0.0 { return false; }
    }
    true
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
