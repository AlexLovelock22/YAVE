use std::collections::{HashMap, HashSet, VecDeque};
use std::mem::size_of;
use std::sync::{mpsc::{self, Receiver, Sender}, Arc};
use std::time::Instant;

use ash::vk;
use glam::{IVec2, IVec3, Vec3};

use crate::{
    meshing::greedy::{mesh_chunk, mesh_chunk_surface, mesh_chunk_water},
    render::{
        buffer::create_buffer,
        context::VulkanContext,
        mesh::{ArenaBuffer, ChunkSlot, IndirectBuffer, Vertex},
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
/// Max chunks uploaded per frame (bounds staging size and vkCopyBuffer count).
const MAX_UPLOAD_BATCH: usize = 64;

type GenResult  = (IVec2, ChunkFaceData, Chunk);
type MeshResult = (IVec2, [(Vec<Vertex>, Vec<u32>); 3], u64, u64);

/// One in-flight batch upload: all LOD meshes for up to MAX_UPLOAD_BATCH chunks
/// packed into a single staging buffer and submitted as one vkQueueSubmit.
struct PendingBatch {
    fence:    vk::Fence,
    cmd:      vk::CommandBuffer,
    stage_vb: vk::Buffer,
    stage_vm: vk::DeviceMemory,
    stage_ib: vk::Buffer,
    stage_im: vk::DeviceMemory,
    /// Slots committed to chunk_allocs when the fence signals.
    slots:    Vec<(IVec2, usize, ChunkSlot)>,
}

pub struct World {
    render_distance: i32,
    lod1_dist:       i32,
    lod2_dist:       i32,
    loaded:          HashSet<IVec2>,
    gen_in_flight:   HashSet<IVec2>,
    pending_mesh:    HashSet<IVec2>,
    mesh_in_flight:  HashSet<IVec2>,
    last_cam_chunk:  IVec2,
    face_data:       HashMap<IVec2, ChunkFaceData>,
    pending_chunks:  HashMap<IVec2, Chunk>,
    cpu_meshes:      HashMap<IVec2, [Arc<(Vec<Vertex>, Vec<u32>)>; 3]>,

    // ── GPU arena ────────────────────────────────────────────────────────────
    arena_vb:        Option<ArenaBuffer>,
    arena_ib:        Option<ArenaBuffer>,
    /// Per-chunk, per-LOD allocation record. None = not yet uploaded.
    chunk_allocs:    HashMap<IVec2, [Option<ChunkSlot>; 3]>,
    /// Chunks waiting to be submitted to rayon for Stage 1 gen, ordered closest-first.
    /// Drained at a throttled rate each frame so mesh tasks aren't starved.
    spawn_queue:     VecDeque<IVec2>,
    /// Chunks whose cpu_meshes are ready but not yet in a batch.
    upload_queue:    Vec<IVec2>,
    /// Single in-flight GPU upload batch (one per frame max).
    pending_batch:   Option<PendingBatch>,
    /// Deferred arena slot frees cycled by frame index (GPU safety lag).
    deferred_frees:  Vec<Vec<ChunkSlot>>,
    /// Double-buffered GPU indirect command buffers — one per in-flight frame slot.
    /// Alternated each frame so the CPU never writes to the buffer the GPU is reading.
    indirect_bufs:         [Option<IndirectBuffer>; 2],
    water_indirect_bufs:   [Option<IndirectBuffer>; 2],
    indirect_frame:        usize,
    last_draw_count:       u32,
    water_last_draw_count: u32,

    transfer_pool:   Option<vk::CommandPool>,
    gen_rx:          Receiver<GenResult>,
    gen_tx:          Sender<GenResult>,
    mesh_rx:         Receiver<MeshResult>,
    mesh_tx:         Sender<MeshResult>,

    frame_idx:         usize,
    stats_timer:       Instant,
    gen_spawned:       u32,
    gen_done:          u32,
    mesh_done:         u32,
    total_gen_us:      u64,
    total_mesh_us:     u64,
    max_chunk_verts:   usize,
    total_chunk_verts: usize,
}

impl World {
    pub fn new(render_distance: i32, lod1_dist: i32, lod2_dist: i32) -> Self {
        let (gen_tx,  gen_rx)  = mpsc::channel();
        let (mesh_tx, mesh_rx) = mpsc::channel();
        Self {
            render_distance,
            lod1_dist,
            lod2_dist,
            loaded:           HashSet::new(),
            gen_in_flight:    HashSet::new(),
            pending_mesh:     HashSet::new(),
            mesh_in_flight:   HashSet::new(),
            last_cam_chunk:   IVec2::new(i32::MAX, i32::MAX),
            face_data:        HashMap::new(),
            pending_chunks:   HashMap::new(),
            cpu_meshes:       HashMap::new(),
            arena_vb:         None,
            arena_ib:         None,
            chunk_allocs:     HashMap::new(),
            spawn_queue:      VecDeque::new(),
            upload_queue:     Vec::new(),
            pending_batch:    None,
            deferred_frees:   (0..DESTROY_LAG).map(|_| Vec::new()).collect(),
            indirect_bufs:         [None, None],
            water_indirect_bufs:   [None, None],
            indirect_frame:        0,
            last_draw_count:       0,
            water_last_draw_count: 0,
            transfer_pool:    None,
            gen_rx, gen_tx, mesh_rx, mesh_tx,
            frame_idx:         0,
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

    fn ensure_transfer_pool(&mut self, ctx: &VulkanContext) -> vk::CommandPool {
        *self.transfer_pool.get_or_insert_with(|| unsafe {
            ctx.device.create_command_pool(&vk::CommandPoolCreateInfo {
                flags: vk::CommandPoolCreateFlags::TRANSIENT,
                queue_family_index: ctx.graphics_family,
                ..Default::default()
            }, None).expect("transfer pool")
        })
    }

    fn defer_free(&mut self, slot: ChunkSlot) {
        let bucket = (self.frame_idx + DESTROY_LAG - 1) % DESTROY_LAG;
        self.deferred_frees[bucket].push(slot);
    }

    /// Drain every deferred-free bucket back into both allocators.
    /// Only call after confirming the GPU has finished (queue_wait_idle completed).
    fn flush_all_deferred_frees(&mut self) {
        let frees: Vec<ChunkSlot> = self.deferred_frees.iter_mut()
            .flat_map(|b| b.drain(..))
            .collect();
        for slot in frees {
            if let Some(ref mut a) = self.arena_vb { a.free(slot.vb_offset, slot.vb_size); }
            if let Some(ref mut a) = self.arena_ib { a.free(slot.ib_offset, slot.ib_size); }
        }
    }

    /// Allocate `size` bytes from the vertex arena, growing if needed.
    /// If the first-fit scan fails, stalls the GPU and reclaims all pending deferred frees
    /// before resorting to a buffer grow — avoids spurious grows due to fragmentation.
    fn alloc_vb(&mut self, size: usize, ctx: &VulkanContext, pool: vk::CommandPool) -> Option<usize> {
        if let Some(off) = self.arena_vb.as_mut()?.alloc(size) { return Some(off); }
        unsafe { let _ = ctx.device.queue_wait_idle(ctx.graphics_queue); }
        self.flush_all_deferred_frees();
        let a = self.arena_vb.as_mut()?;
        if let Some(off) = a.alloc(size) { return Some(off); }
        let new_cap = (a.cap + size).next_power_of_two();
        if let Err(e) = a.ensure_cap(ctx, pool, new_cap) {
            eprintln!("[arena_vb] grow: {e}"); return None;
        }
        a.alloc(size)
    }

    /// Allocate `size` bytes from the index arena, growing if needed.
    fn alloc_ib(&mut self, size: usize, ctx: &VulkanContext, pool: vk::CommandPool) -> Option<usize> {
        if let Some(off) = self.arena_ib.as_mut()?.alloc(size) { return Some(off); }
        unsafe { let _ = ctx.device.queue_wait_idle(ctx.graphics_queue); }
        self.flush_all_deferred_frees();
        let a = self.arena_ib.as_mut()?;
        if let Some(off) = a.alloc(size) { return Some(off); }
        let new_cap = (a.cap + size).next_power_of_two();
        if let Err(e) = a.ensure_cap(ctx, pool, new_cap) {
            eprintln!("[arena_ib] grow: {e}"); return None;
        }
        a.alloc(size)
    }

    fn spawn_mesh(&mut self, coord: IVec2) {
        let neighbors = build_neighbor_masks(coord, &self.face_data);
        let tx    = self.mesh_tx.clone();
        let chunk = self.pending_chunks.remove(&coord).unwrap_or_else(|| {
            let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
            generate(origin)
        });
        self.mesh_in_flight.insert(coord);
        rayon::spawn(move || {
            let t1 = Instant::now();
            let (lod0, (lod1, lod2)) = rayon::join(
                || mesh_chunk(&chunk, &neighbors),
                || rayon::join(
                    || mesh_chunk_surface(&chunk, &neighbors),
                    || mesh_chunk_water(&chunk, &neighbors), // water-only; drawn after opaque
                ),
            );
            let mesh_us = t1.elapsed().as_micros() as u64;
            let _ = tx.send((coord, [lod0, lod1, lod2], 0, mesh_us));
        });
    }

    pub fn update(&mut self, camera_pos: Vec3, ctx: &VulkanContext) {
        // ── Lazy GPU resource init ────────────────────────────────────────────
        if self.arena_vb.is_none() {
            // Scale with render distance so large worlds don't hit frequent arena-grow stalls.
            // LOD0 and LOD1 are both full-resolution (surface mesher), so budget ~5000 bytes/chunk
            // for VB and ~800 bytes/chunk for IB across both LODs combined.
            let chunks = ((2 * self.render_distance + 1) as usize).pow(2);
            let vb_cap = (chunks * 5_000).next_power_of_two().clamp(256 << 20, 512 << 20);
            let ib_cap = (chunks *   800).next_power_of_two().clamp( 64 << 20, 128 << 20);
            match (
                ArenaBuffer::new(ctx, vb_cap, vk::BufferUsageFlags::VERTEX_BUFFER, size_of::<Vertex>()),
                ArenaBuffer::new(ctx, ib_cap, vk::BufferUsageFlags::INDEX_BUFFER,  size_of::<u32>()),
            ) {
                (Ok(vb), Ok(ib)) => {
                    println!("[arena] vb={}MB ib={}MB", vb_cap >> 20, ib_cap >> 20);
                    self.arena_vb = Some(vb);
                    self.arena_ib = Some(ib);
                }
                _ => eprintln!("[arena] initial alloc failed"),
            }
        }
        if self.indirect_bufs[0].is_none() {
            let cap = ((2 * self.render_distance + 1) as usize).pow(2);
            for slot in &mut self.indirect_bufs {
                if let Ok(buf) = IndirectBuffer::new(ctx, cap) { *slot = Some(buf); }
            }
            for slot in &mut self.water_indirect_bufs {
                if let Ok(buf) = IndirectBuffer::new(ctx, cap) { *slot = Some(buf); }
            }
        }

        // ── Deferred arena slot frees ─────────────────────────────────────────
        let safe_slot = self.frame_idx % DESTROY_LAG;
        for slot in self.deferred_frees[safe_slot].drain(..) {
            if let Some(ref mut a) = self.arena_vb { a.free(slot.vb_offset, slot.vb_size); }
            if let Some(ref mut a) = self.arena_ib { a.free(slot.ib_offset, slot.ib_size); }
        }
        self.frame_idx += 1;

        // ── Throttled gen spawning ─────────────────────────────────────────────
        // Drain spawn_queue at a rate that keeps workers busy without flooding.
        // gen_in_flight includes both queued and actually-spawned chunks;
        // subtracting spawn_queue.len() gives the count actually in rayon.
        if !self.spawn_queue.is_empty() {
            let rayon_depth = self.gen_in_flight.len().saturating_sub(self.spawn_queue.len());
            let target      = (rayon::current_num_threads() * 4).max(16);
            let to_drain    = target.saturating_sub(rayon_depth).min(self.spawn_queue.len());
            for _ in 0..to_drain {
                if let Some(coord) = self.spawn_queue.pop_front() {
                    let tx     = self.gen_tx.clone();
                    let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
                    rayon::spawn(move || {
                        let chunk = generate(origin);
                        let fd    = ChunkFaceData::extract(&chunk);
                        let _ = tx.send((coord, fd, chunk));
                    });
                }
            }
        }

        let cam_chunk = world_to_chunk(camera_pos);
        let rd = self.render_distance;

        // ── Poll completed upload batch ───────────────────────────────────────
        if let Some(ref b) = self.pending_batch {
            if unsafe { ctx.device.get_fence_status(b.fence).unwrap_or(false) } {
                let b = self.pending_batch.take().unwrap();
                let pool = self.ensure_transfer_pool(ctx);
                unsafe {
                    ctx.device.destroy_fence(b.fence, None);
                    ctx.device.free_command_buffers(pool, &[b.cmd]);
                    ctx.device.destroy_buffer(b.stage_vb, None);
                    ctx.device.free_memory(b.stage_vm, None);
                    ctx.device.destroy_buffer(b.stage_ib, None);
                    ctx.device.free_memory(b.stage_im, None);
                }
                for (coord, lod, slot) in b.slots {
                    self.chunk_allocs.entry(coord).or_insert([None; 3])[lod] = Some(slot);
                }
            }
        }

        // ── Stage 1 results ───────────────────────────────────────────────────
        while let Ok((coord, fd, chunk)) = self.gen_rx.try_recv() {
            self.gen_in_flight.remove(&coord);
            self.gen_done += 1;

            if (coord.x - cam_chunk.x).abs() > rd || (coord.y - cam_chunk.y).abs() > rd { continue; }

            self.face_data.insert(coord, fd);
            self.pending_chunks.insert(coord, chunk);

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
        while let Ok((coord, lods, _gen_us, mesh_us)) = self.mesh_rx.try_recv() {
            self.mesh_in_flight.remove(&coord);
            self.mesh_done += 1;
            self.total_mesh_us += mesh_us;
            let nv = lods[0].0.len();
            self.total_chunk_verts += nv;
            if nv > self.max_chunk_verts { self.max_chunk_verts = nv; }

            if (coord.x - cam_chunk.x).abs() > rd || (coord.y - cam_chunk.y).abs() > rd { continue; }

            self.loaded.insert(coord);
            if !lods[0].0.is_empty() {
                let [m0, m1, m2] = lods;
                self.cpu_meshes.insert(coord, [Arc::new(m0), Arc::new(m1), Arc::new(m2)]);
                self.upload_queue.push(coord);
            }
        }

        // ── Periodic stats ────────────────────────────────────────────────────
        let settled = self.gen_in_flight.is_empty()
            && self.spawn_queue.is_empty()
            && self.pending_mesh.is_empty()
            && self.mesh_in_flight.is_empty();
        if !settled && self.stats_timer.elapsed().as_millis() >= 500 {
            self.stats_timer = Instant::now();
            let n = self.mesh_done.max(1) as u64;
            // println!(
            //     "[chunks] gen {}/{}  mesh {}/{}  fly {} pend {} mfly {}  \
            //      avg_mesh {}us  max_verts {} avg_verts {}  queue {}",
            //     self.gen_done, self.gen_spawned,
            //     self.mesh_done, self.gen_spawned,
            //     self.gen_in_flight.len(), self.pending_mesh.len(), self.mesh_in_flight.len(),
            //     self.total_mesh_us / n,
            //     self.max_chunk_verts,
            //     self.total_chunk_verts / self.mesh_done.max(1) as usize,
            //     self.upload_queue.len(),
            // );
        }

        // ── Submit upload batch (one per frame max) ───────────────────────────
        if self.pending_batch.is_none() && !self.upload_queue.is_empty()
            && self.arena_vb.is_some()
        {
            self.submit_upload_batch(ctx);
        }

        // ── O(RD²) load/unload scan on chunk boundary ─────────────────────────
        if cam_chunk != self.last_cam_chunk {
            self.last_cam_chunk = cam_chunk;

            let before = self.loaded.len();
            self.loaded.retain(|c| (c.x - cam_chunk.x).abs() <= rd && (c.y - cam_chunk.y).abs() <= rd);
            if self.loaded.len() != before {
                self.cpu_meshes.retain(|c, _| self.loaded.contains(c));
                self.upload_queue.retain(|c| self.loaded.contains(c));

                let unloaded: Vec<IVec2> = self.chunk_allocs.keys()
                    .filter(|c| !self.loaded.contains(*c))
                    .copied().collect();
                for coord in unloaded {
                    if let Some(allocs) = self.chunk_allocs.remove(&coord) {
                        for slot in allocs.into_iter().flatten() {
                            self.defer_free(slot);
                        }
                    }
                }
            }

            self.face_data.retain(|c, _| (c.x - cam_chunk.x).abs() <= rd + 2 && (c.y - cam_chunk.y).abs() <= rd + 2);
            self.pending_mesh.retain(|c| (c.x - cam_chunk.x).abs() <= rd && (c.y - cam_chunk.y).abs() <= rd);
            self.pending_chunks.retain(|c, _| self.pending_mesh.contains(c));
            self.gen_in_flight.retain(|c| (c.x - cam_chunk.x).abs() <= rd + 2 && (c.y - cam_chunk.y).abs() <= rd + 2);
            self.spawn_queue.retain(|c| (c.x - cam_chunk.x).abs() <= rd + 2 && (c.y - cam_chunk.y).abs() <= rd + 2);
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

            if !to_spawn.is_empty() && self.gen_in_flight.is_empty() && self.spawn_queue.is_empty() {
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
                // Add to gen_in_flight immediately so can_mesh() correctly treats this
                // chunk as unavailable for its neighbours even while it awaits rayon.
                self.gen_in_flight.insert(coord);
                self.gen_spawned += 1;
                self.spawn_queue.push_back(coord);
            }
        }
    }

    /// Build and submit a single command buffer covering all pending chunk LOD uploads.
    /// Called at most once per frame (only when no batch is in flight).
    fn submit_upload_batch(&mut self, ctx: &VulkanContext) {
        let pool = self.ensure_transfer_pool(ctx);

        // Drain up to MAX_UPLOAD_BATCH coords from upload_queue.
        let drain = self.upload_queue.len().min(MAX_UPLOAD_BATCH);
        let to_upload: Vec<IVec2> = self.upload_queue.drain(..drain).collect();

        // ── First pass: allocate arena slots and compute staging layout ────────
        struct Entry {
            coord:       IVec2,
            lod:         usize,
            slot:        ChunkSlot,
            vb_stage:    usize, // byte offset in the staging VB
            ib_stage:    usize, // byte offset in the staging IB
        }

        let mut entries: Vec<Entry> = Vec::new();
        let mut vb_stage_total = 0usize;
        let mut ib_stage_total = 0usize;

        for coord in &to_upload {
            let lods = match self.cpu_meshes.get(coord) { Some(m) => m.clone(), None => continue };
            for lod in 0..3usize {
                let (verts, idxs) = lods[lod].as_ref();
                if verts.is_empty() { continue; }

                let vb_size = verts.len() * size_of::<Vertex>();
                let ib_size = idxs.len()  * size_of::<u32>();

                // Defer-free any previously uploaded slot for this chunk/LOD.
                if let Some(old) = self.chunk_allocs.get(coord).and_then(|a| a[lod]) {
                    self.defer_free(old);
                    self.chunk_allocs.entry(*coord).and_modify(|a| a[lod] = None);
                }

                let vb_off = match self.alloc_vb(vb_size, ctx, pool) {
                    Some(o) => o,
                    None    => { eprintln!("[arena] vb alloc failed"); continue; }
                };
                let ib_off = match self.alloc_ib(ib_size, ctx, pool) {
                    Some(o) => o,
                    None    => {
                        if let Some(ref mut a) = self.arena_vb { a.free(vb_off, vb_size); }
                        eprintln!("[arena] ib alloc failed"); continue;
                    }
                };

                entries.push(Entry {
                    coord: *coord, lod,
                    slot: ChunkSlot {
                        vb_offset:   vb_off,  vb_size,
                        ib_offset:   ib_off,  ib_size,
                        index_count: idxs.len() as u32,
                        first_index: (ib_off / size_of::<u32>()) as u32,
                        vertex_base: (vb_off / size_of::<Vertex>()) as i32,
                    },
                    vb_stage: vb_stage_total,
                    ib_stage: ib_stage_total,
                });
                vb_stage_total += vb_size;
                ib_stage_total += ib_size;
            }
        }

        if entries.is_empty() { return; }

        // ── Allocate and fill staging buffers ─────────────────────────────────
        let (stage_vb, stage_vm) = match create_buffer(
            ctx, vb_stage_total as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) { Ok(r) => r, Err(e) => { eprintln!("[upload] staging vb alloc: {e}"); return; } };

        let (stage_ib, stage_im) = match create_buffer(
            ctx, ib_stage_total as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) { Ok(r) => r, Err(e) => {
            unsafe { ctx.device.destroy_buffer(stage_vb, None); ctx.device.free_memory(stage_vm, None); }
            eprintln!("[upload] staging ib alloc: {e}"); return;
        }};

        unsafe {
            let vb_ptr = ctx.device.map_memory(stage_vm, 0, vb_stage_total as vk::DeviceSize, vk::MemoryMapFlags::empty())
                .expect("map stage_vm") as *mut u8;
            let ib_ptr = ctx.device.map_memory(stage_im, 0, ib_stage_total as vk::DeviceSize, vk::MemoryMapFlags::empty())
                .expect("map stage_im") as *mut u8;

            for e in &entries {
                let (verts, idxs) = self.cpu_meshes[&e.coord][e.lod].as_ref();
                std::ptr::copy_nonoverlapping(
                    verts.as_ptr() as *const u8, vb_ptr.add(e.vb_stage), e.slot.vb_size);
                std::ptr::copy_nonoverlapping(
                    idxs.as_ptr()  as *const u8, ib_ptr.add(e.ib_stage), e.slot.ib_size);
            }

            ctx.device.unmap_memory(stage_vm);
            ctx.device.unmap_memory(stage_im);
        }

        // ── Record and submit one command buffer with all copies ───────────────
        let cmd = unsafe {
            ctx.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            }).expect("alloc cmd")[0]
        };

        let arena_vb = self.arena_vb.as_ref().unwrap().buffer;
        let arena_ib = self.arena_ib.as_ref().unwrap().buffer;

        unsafe {
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            }).expect("begin cmd");

            for e in &entries {
                ctx.device.cmd_copy_buffer(cmd, stage_vb, arena_vb, &[vk::BufferCopy {
                    src_offset: e.vb_stage as vk::DeviceSize,
                    dst_offset: e.slot.vb_offset as vk::DeviceSize,
                    size:       e.slot.vb_size as vk::DeviceSize,
                }]);
                ctx.device.cmd_copy_buffer(cmd, stage_ib, arena_ib, &[vk::BufferCopy {
                    src_offset: e.ib_stage as vk::DeviceSize,
                    dst_offset: e.slot.ib_offset as vk::DeviceSize,
                    size:       e.slot.ib_size as vk::DeviceSize,
                }]);
            }

            ctx.device.end_command_buffer(cmd).expect("end cmd");
        }

        let fence = unsafe {
            ctx.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("fence")
        };
        let cmds = [cmd];
        unsafe {
            ctx.device.queue_submit(ctx.graphics_queue, &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: cmds.as_ptr(),
                ..Default::default()
            }], fence).expect("submit batch");
        }

        let slots = entries.into_iter().map(|e| (e.coord, e.lod, e.slot)).collect();
        self.pending_batch = Some(PendingBatch { fence, cmd, stage_vb, stage_vm, stage_ib, stage_im, slots });
    }

    /// Returns the arena vertex and index buffer handles.
    pub fn render_buffers(&self) -> Option<(vk::Buffer, vk::Buffer)> {
        match (&self.arena_vb, &self.arena_ib) {
            (Some(vb), Some(ib)) => Some((vb.buffer, ib.buffer)),
            _ => None,
        }
    }

    /// Frustum-cull loaded chunks and fill separate opaque and water draw lists.
    /// LOD0/LOD1 slots are opaque geometry; LOD2 is water-only geometry.
    /// Caller must submit opaque draw before water draw for correct alpha blending.
    pub fn cull_draws(
        &self,
        planes: &[[f32; 4]; 6],
        camera_pos: Vec3,
        out_opaque: &mut Vec<vk::DrawIndexedIndirectCommand>,
        out_water:  &mut Vec<vk::DrawIndexedIndirectCommand>,
    ) {
        out_opaque.clear();
        out_water.clear();
        let cam_cx = (camera_pos.x / CHUNK_SIZE as f32).floor() as i32;
        let cam_cz = (camera_pos.z / CHUNK_SIZE as f32).floor() as i32;

        let push = |s: ChunkSlot, out: &mut Vec<vk::DrawIndexedIndirectCommand>| {
            out.push(vk::DrawIndexedIndirectCommand {
                index_count:    s.index_count,
                instance_count: 1,
                first_index:    s.first_index,
                vertex_offset:  s.vertex_base,
                first_instance: 0,
            });
        };

        for (&coord, _) in &self.cpu_meshes {
            if !chunk_in_frustum(coord, planes) { continue; }
            let allocs = match self.chunk_allocs.get(&coord) { Some(a) => a, None => continue };

            // Opaque: LOD0/LOD1 only — LOD2 is reserved for water.
            let dist = (coord.x - cam_cx).abs().max((coord.y - cam_cz).abs());
            let desired = if dist < self.lod1_dist { 0usize } else { 1 };
            if let Some(s) = [desired, 0, 1].iter().find_map(|&l| allocs[l]) {
                push(s, out_opaque);
            }

            // Water: always LOD2.
            if let Some(s) = allocs[2] {
                push(s, out_water);
            }
        }
    }

    /// Advance to the next frame slot and write `cmds` into the GPU indirect buffer.
    /// Must be called once per frame, before `indirect_draw` and `draw_frame`.
    pub fn flush_indirect(&mut self, cmds: &[vk::DrawIndexedIndirectCommand], ctx: &VulkanContext) {
        self.indirect_frame += 1;
        self.last_draw_count = cmds.len() as u32;
        let slot = self.indirect_frame % 2;
        if let Some(ref mut buf) = self.indirect_bufs[slot] {
            if let Err(e) = buf.ensure_cap(ctx, cmds.len().max(1)) {
                eprintln!("[indirect] grow: {e}"); return;
            }
            buf.write(cmds);
        }
    }

    /// Returns (indirect_buffer, draw_count) for `draw_frame`. Valid after `flush_indirect`.
    pub fn indirect_draw(&self) -> Option<(vk::Buffer, u32)> {
        let slot = self.indirect_frame % 2;
        self.indirect_bufs[slot].as_ref().map(|b| (b.buffer, self.last_draw_count))
    }

    /// Write the water draw list into the water indirect buffer (same frame slot as opaque).
    /// Call after `flush_indirect`.
    pub fn flush_indirect_water(&mut self, cmds: &[vk::DrawIndexedIndirectCommand], ctx: &VulkanContext) {
        self.water_last_draw_count = cmds.len() as u32;
        let slot = self.indirect_frame % 2;
        if let Some(ref mut buf) = self.water_indirect_bufs[slot] {
            if let Err(e) = buf.ensure_cap(ctx, cmds.len().max(1)) {
                eprintln!("[indirect water] grow: {e}"); return;
            }
            buf.write(cmds);
        }
    }

    pub fn indirect_draw_water(&self) -> Option<(vk::Buffer, u32)> {
        let slot = self.indirect_frame % 2;
        self.water_indirect_bufs[slot].as_ref().map(|b| (b.buffer, self.water_last_draw_count))
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        // Wait for any in-flight batch, then free its resources.
        if let Some(b) = self.pending_batch.take() {
            if let Some(pool) = self.transfer_pool {
                unsafe {
                    let _ = ctx.device.wait_for_fences(&[b.fence], true, u64::MAX);
                    ctx.device.destroy_fence(b.fence, None);
                    ctx.device.free_command_buffers(pool, &[b.cmd]);
                    ctx.device.destroy_buffer(b.stage_vb, None);
                    ctx.device.free_memory(b.stage_vm, None);
                    ctx.device.destroy_buffer(b.stage_ib, None);
                    ctx.device.free_memory(b.stage_im, None);
                }
            }
        }
        for bucket in &mut self.deferred_frees { bucket.clear(); }
        if let Some(ref a) = self.arena_vb { a.destroy(ctx); }
        if let Some(ref a) = self.arena_ib { a.destroy(ctx); }
        for slot in &self.indirect_bufs       { if let Some(ref b) = slot { b.destroy(ctx); } }
        for slot in &self.water_indirect_bufs { if let Some(ref b) = slot { b.destroy(ctx); } }
        if let Some(pool) = self.transfer_pool.take() {
            unsafe { ctx.device.destroy_command_pool(pool, None); }
        }
        self.arena_vb      = None;
        self.arena_ib      = None;
        self.indirect_bufs = [None, None];
    }
}

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

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = x + z * CHUNK_SIZE;
            let sy  = surface[idx] as usize;

            // Fill solid stone from y=0 up to (but not including) the surface.
            for y in 0..sy { chunk.set(x, y, z, STONE); }

            if is_ocean[idx] {
                chunk.set(x, sy, z, STONE);
                for y in (sy + 1)..=SEA_LEVEL {
                    chunk.set(x, y, z, WATER);
                }
            } else {
                chunk.set(x, sy, z, DIRT);
            }
        }
    }
    chunk
}
