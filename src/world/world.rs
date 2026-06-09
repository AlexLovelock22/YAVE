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
        block::{AIR, BlockId, DIRT, STONE, WATER, is_opaque},
        chunk::{Chunk, CHUNK_SIZE, CHUNK_HEIGHT},
        continents::{build_defs, SEA_LEVEL},
        neighbor::{ChunkFaceData, NeighborMasks},
        terrain::sample_chunk_heights,
    },
};

const DESTROY_LAG: usize = 3;
/// Chebyshev chunk radius around the player where block data is kept in memory for interaction.
const INTERACT_RADIUS: i32 = 3;
/// Maximum raycast reach in blocks.
const RAYCAST_REACH: f32 = 10.0;
/// Max chunks uploaded per frame (bounds staging size and vkCopyBuffer count).
const MAX_UPLOAD_BATCH: usize = 64;
/// Persistent staging buffer sizes.
/// AO meshing produces much larger vertex buffers than pre-AO (36B/vertex, less greedy merging).
/// 128 MiB VB / 64 chunks ≈ 2 MiB/chunk headroom — enough for any realistic terrain chunk.
const STAGING_VB_CAP: usize = 128 << 20; // 128 MiB
const STAGING_IB_CAP: usize =  32 << 20; //  32 MiB

type GenResult  = (IVec2, ChunkFaceData, Chunk);
type MeshResult = (IVec2, [(Vec<Vertex>, Vec<u32>); 3], u64, u64);

/// One in-flight batch upload: all LOD meshes for up to MAX_UPLOAD_BATCH chunks
/// packed into a single staging buffer and submitted as one vkQueueSubmit.
struct PendingBatch {
    fence: vk::Fence,
    cmd:   vk::CommandBuffer,
    /// Slots committed to chunk_allocs when the fence signals.
    slots: Vec<(IVec2, usize, ChunkSlot)>,
    /// Old slots to defer-free once the fence signals (after new slots are committed).
    /// Kept alive until then so the old GPU mesh stays visible with no 1-frame gap.
    old_slots: Vec<ChunkSlot>,
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
    // Persistent staging buffers — allocated once at first upload, kept permanently mapped.
    staging_vb:     vk::Buffer,
    staging_vm:     vk::DeviceMemory,
    staging_ib:     vk::Buffer,
    staging_im:     vk::DeviceMemory,
    staging_vb_ptr: *mut u8,
    staging_ib_ptr: *mut u8,
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

    // ── Block interaction ─────────────────────────────────────────────────────
    /// Full chunk block data for chunks within INTERACT_RADIUS + player-modified chunks.
    stored_chunks:   HashMap<IVec2, Chunk>,
    /// Tracks which chunks have player edits (never evicted from stored_chunks on distance).
    modified_set:    HashSet<IVec2>,
    /// Chunks needing remesh once their current in-flight mesh completes.
    pending_remesh:  HashSet<IVec2>,

    // ── Diagnostics (written each update(), read by app for logging) ──────────
    pub diag_gen_rx_us:    u64,  // time spent draining gen channel
    pub diag_mesh_rx_us:   u64,  // time spent draining mesh channel
    pub diag_batch_poll_us: u64, // time spent polling pending batch fence
    pub diag_upload_us:    u64,  // time spent in submit_upload_batch
    pub diag_meshes_in:    u32,  // mesh results received this frame
    pub diag_batch_chunks: u32,  // chunks in upload batch (0 if no batch)
    pub diag_batch_kb:     u32,  // staging bytes used (VB+IB) in KB
    pub diag_chunk_changed: bool, // true if player crossed a chunk boundary this frame
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
            staging_vb:     vk::Buffer::null(),
            staging_vm:     vk::DeviceMemory::null(),
            staging_ib:     vk::Buffer::null(),
            staging_im:     vk::DeviceMemory::null(),
            staging_vb_ptr: std::ptr::null_mut(),
            staging_ib_ptr: std::ptr::null_mut(),
            gen_rx, gen_tx, mesh_rx, mesh_tx,
            stored_chunks:   HashMap::new(),
            modified_set:    HashSet::new(),
            pending_remesh:  HashSet::new(),
            frame_idx:         0,
            stats_timer:       Instant::now(),
            gen_spawned:       0,
            gen_done:          0,
            mesh_done:         0,
            total_gen_us:      0,
            total_mesh_us:     0,
            max_chunk_verts:   0,
            total_chunk_verts: 0,
            diag_gen_rx_us:    0,
            diag_mesh_rx_us:   0,
            diag_batch_poll_us: 0,
            diag_upload_us:    0,
            diag_meshes_in:    0,
            diag_batch_chunks: 0,
            diag_batch_kb:     0,
            diag_chunk_changed: false,
        }
    }

    fn ensure_staging(&mut self, ctx: &VulkanContext) -> bool {
        if !self.staging_vb_ptr.is_null() { return true; }
        let (vb, vm) = match create_buffer(
            ctx, STAGING_VB_CAP as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) { Ok(r) => r, Err(e) => { eprintln!("[staging] vb: {e}"); return false; } };
        let (ib, im) = match create_buffer(
            ctx, STAGING_IB_CAP as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) { Ok(r) => r, Err(e) => {
            unsafe { ctx.device.destroy_buffer(vb, None); ctx.device.free_memory(vm, None); }
            eprintln!("[staging] ib: {e}"); return false;
        }};
        unsafe {
            let vb_ptr = ctx.device.map_memory(vm, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map staging vb") as *mut u8;
            let ib_ptr = ctx.device.map_memory(im, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map staging ib") as *mut u8;
            self.staging_vb = vb; self.staging_vm = vm;
            self.staging_ib = ib; self.staging_im = im;
            self.staging_vb_ptr = vb_ptr;
            self.staging_ib_ptr = ib_ptr;
        }
        true
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
        // Prefer stored_chunks (has player edits), fall back to pending_chunks, then regenerate.
        let stored = self.stored_chunks.get(&coord).cloned(); // release borrow before pending_chunks.remove
        let chunk = if let Some(c) = stored {
            self.pending_chunks.remove(&coord);
            c
        } else {
            let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
            self.pending_chunks.remove(&coord).unwrap_or_else(|| generate(origin))
        };
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
            // AO mesh: 36 B/vertex, ~4 K verts LOD0 for near chunks + ~1 K verts LOD1 for all.
            // LOD0 is only uploaded for the lod0 band; budget ~60 KB/chunk covers both bands.
            let vb_cap = (chunks * 60_000).next_power_of_two().clamp(512 << 20, 2048 << 20);
            let ib_cap = (chunks * 10_000).next_power_of_two().clamp(128 << 20,  512 << 20);
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

        // Reset per-frame diagnostics.
        self.diag_gen_rx_us    = 0;
        self.diag_mesh_rx_us   = 0;
        self.diag_batch_poll_us = 0;
        self.diag_upload_us    = 0;
        self.diag_meshes_in    = 0;
        self.diag_batch_chunks = 0;
        self.diag_batch_kb     = 0;
        self.diag_chunk_changed = false;

        let cam_chunk = world_to_chunk(camera_pos);
        let rd = self.render_distance;

        // ── Poll completed upload batch ───────────────────────────────────────
        let t_poll = Instant::now();
        if let Some(ref b) = self.pending_batch {
            if unsafe { ctx.device.get_fence_status(b.fence).unwrap_or(false) } {
                let b = self.pending_batch.take().unwrap();
                let pool = self.ensure_transfer_pool(ctx);
                unsafe {
                    ctx.device.destroy_fence(b.fence, None);
                    ctx.device.free_command_buffers(pool, &[b.cmd]);
                    // Staging buffers are persistent — not freed here.
                }
                // Commit new slots first — GPU copy has completed, data is ready.
                for (coord, lod, slot) in b.slots {
                    self.chunk_allocs.entry(coord).or_insert([None; 3])[lod] = Some(slot);
                }
                // Now safe to free old slots; new mesh is already live in chunk_allocs.
                for old in b.old_slots {
                    self.defer_free(old);
                }
            }
        }
        self.diag_batch_poll_us = t_poll.elapsed().as_micros() as u64;

        // ── Stage 1 results ───────────────────────────────────────────────────
        let t_gen = Instant::now();
        while let Ok((coord, fd, chunk)) = self.gen_rx.try_recv() {
            self.gen_in_flight.remove(&coord);
            self.gen_done += 1;

            if (coord.x - cam_chunk.x).abs() > rd || (coord.y - cam_chunk.y).abs() > rd { continue; }

            self.face_data.insert(coord, fd);
            // Cache block data for nearby chunks (needed for raycasting / block editing).
            // Don't overwrite an existing stored_chunk — it might have player edits.
            let interact = (coord.x - cam_chunk.x).abs().max((coord.y - cam_chunk.y).abs()) <= INTERACT_RADIUS;
            if interact && !self.stored_chunks.contains_key(&coord) {
                self.stored_chunks.insert(coord, chunk.clone());
            }
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

        // Every-frame catch: unblock pending_mesh entries that became meshable this frame.
        // Runs unconditionally (not just on camera move) so stationary-camera loads don't stall
        // when a gen-in-flight neighbour was culled by retain without completing.
        {
            let unblocked: Vec<IVec2> = self.pending_mesh.iter()
                .filter(|&&c| can_mesh(c, &self.gen_in_flight))
                .copied().collect();
            for coord in unblocked {
                self.pending_mesh.remove(&coord);
                if !self.mesh_in_flight.contains(&coord) { self.spawn_mesh(coord); }
            }
        }

        self.diag_gen_rx_us = t_gen.elapsed().as_micros() as u64;

        // ── Stage 2 results ───────────────────────────────────────────────────
        let t_mesh = Instant::now();
        while let Ok((coord, lods, _gen_us, mesh_us)) = self.mesh_rx.try_recv() {
            self.mesh_in_flight.remove(&coord);
            self.mesh_done += 1;
            self.total_mesh_us += mesh_us;
            self.diag_meshes_in += 1;
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
        self.diag_mesh_rx_us = t_mesh.elapsed().as_micros() as u64;

        // ── Pending remesh (block edits that arrived while mesh was in-flight) ─
        let remesh: Vec<IVec2> = self.pending_remesh.iter()
            .filter(|c| !self.mesh_in_flight.contains(c))
            .copied().collect();
        for coord in remesh {
            self.pending_remesh.remove(&coord);
            self.mesh_chunk_now(coord);
        }

        // ── Submit upload batch (one per frame max) ───────────────────────────
        if self.pending_batch.is_none() && !self.upload_queue.is_empty()
            && self.arena_vb.is_some()
        {
            let t_up = Instant::now();
            self.submit_upload_batch(ctx, cam_chunk);
            self.diag_upload_us = t_up.elapsed().as_micros() as u64;
        }

        // ── O(RD²) load/unload scan on chunk boundary ─────────────────────────
        if cam_chunk != self.last_cam_chunk {
            self.last_cam_chunk = cam_chunk;
            self.diag_chunk_changed = true;

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
            // Evict cached block data for chunks beyond the interact radius (keep modified chunks).
            self.stored_chunks.retain(|c, _| {
                self.modified_set.contains(c)
                    || ((c.x - cam_chunk.x).abs().max((c.y - cam_chunk.y).abs()) <= INTERACT_RADIUS)
            });
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

            // When the camera moves into the lod0 band for a chunk that only has LOD1
            // uploaded (because it was distant when first loaded), queue its LOD0 upload.
            for (&coord, allocs) in &self.chunk_allocs {
                let dist = (coord.x - cam_chunk.x).abs().max((coord.y - cam_chunk.y).abs());
                if dist < self.lod1_dist && allocs[0].is_none() && self.cpu_meshes.contains_key(&coord) {
                    self.upload_queue.push(coord);
                }
            }
        }
    }

    /// Build and submit a single command buffer covering all pending chunk LOD uploads.
    /// Called at most once per frame (only when no batch is in flight).
    fn submit_upload_batch(&mut self, ctx: &VulkanContext, cam_chunk: IVec2) {
        let pool = self.ensure_transfer_pool(ctx);

        // Drain up to MAX_UPLOAD_BATCH coords from upload_queue.
        let drain = self.upload_queue.len().min(MAX_UPLOAD_BATCH);
        let to_upload: Vec<IVec2> = self.upload_queue.drain(..drain).collect();

        // ── First pass: allocate arena slots and compute staging layout ────────
        struct Entry {
            coord:       IVec2,
            lod:         usize,
            slot:        ChunkSlot,
            old_slot:    Option<ChunkSlot>, // previous alloc, freed when fence signals
            vb_stage:    usize, // byte offset in the staging VB
            ib_stage:    usize, // byte offset in the staging IB
        }

        let mut entries: Vec<Entry> = Vec::new();
        let mut vb_stage_total = 0usize;
        let mut ib_stage_total = 0usize;

        for coord in &to_upload {
            let lods = match self.cpu_meshes.get(coord) { Some(m) => m.clone(), None => continue };
            let dist = (coord.x - cam_chunk.x).abs().max((coord.y - cam_chunk.y).abs());
            for lod in 0..3usize {
                // Skip LOD0 for distant chunks — they only render LOD1, and uploading LOD0 for
                // every chunk would drive the arena from ~600 MiB to ~2 GiB, causing repeated
                // queue_wait_idle stalls when the arena grows.  LOD0 is queued on approach
                // (see camera-move block above).
                if lod == 0 && dist >= self.lod1_dist { continue; }
                let (verts, idxs) = lods[lod].as_ref();
                if verts.is_empty() { continue; }

                let vb_size = verts.len() * size_of::<Vertex>();
                let ib_size = idxs.len()  * size_of::<u32>();

                // Capture the old slot — but do NOT nullify chunk_allocs yet.
                // The old GPU mesh stays visible until the fence signals and the new
                // slot is committed, preventing the 1-frame invisible gap.
                let old_slot = self.chunk_allocs.get(coord).and_then(|a| a[lod]);

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
                    coord: *coord, lod, old_slot,
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

        if entries.is_empty() {
            // All coords failed alloc (arena not yet ready or every LOD was empty/skipped).
            // Re-queue so they're retried next frame rather than being silently dropped.
            let remaining = std::mem::take(&mut self.upload_queue);
            self.upload_queue = to_upload.into_iter()
                .filter(|c| self.cpu_meshes.contains_key(c))
                .chain(remaining)
                .collect();
            return;
        }

        // Record batch diagnostics before doing the heavy work.
        self.diag_batch_chunks = to_upload.len() as u32;
        self.diag_batch_kb     = ((vb_stage_total + ib_stage_total) / 1024) as u32;

        // ── Fill persistent staging buffers ───────────────────────────────────
        if !self.ensure_staging(ctx) { return; }
        if vb_stage_total > STAGING_VB_CAP || ib_stage_total > STAGING_IB_CAP {
            eprintln!("[upload] staging overflow vb={vb_stage_total} ib={ib_stage_total} — re-queuing {} chunks", to_upload.len());
            // Free the arena slots we just allocated so the space isn't wasted.
            for e in &entries {
                if let Some(ref mut a) = self.arena_vb { a.free(e.slot.vb_offset, e.slot.vb_size); }
                if let Some(ref mut a) = self.arena_ib { a.free(e.slot.ib_offset, e.slot.ib_size); }
            }
            // Put the drained chunks back at the front of the queue so they're retried.
            let remaining = std::mem::take(&mut self.upload_queue);
            self.upload_queue = to_upload.into_iter()
                .filter(|c| self.cpu_meshes.contains_key(c))
                .chain(remaining)
                .collect();
            return;
        }
        unsafe {
            for e in &entries {
                let (verts, idxs) = self.cpu_meshes[&e.coord][e.lod].as_ref();
                std::ptr::copy_nonoverlapping(
                    verts.as_ptr() as *const u8, self.staging_vb_ptr.add(e.vb_stage), e.slot.vb_size);
                std::ptr::copy_nonoverlapping(
                    idxs.as_ptr()  as *const u8, self.staging_ib_ptr.add(e.ib_stage), e.slot.ib_size);
            }
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
                ctx.device.cmd_copy_buffer(cmd, self.staging_vb, arena_vb, &[vk::BufferCopy {
                    src_offset: e.vb_stage as vk::DeviceSize,
                    dst_offset: e.slot.vb_offset as vk::DeviceSize,
                    size:       e.slot.vb_size as vk::DeviceSize,
                }]);
                ctx.device.cmd_copy_buffer(cmd, self.staging_ib, arena_ib, &[vk::BufferCopy {
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

        let old_slots: Vec<ChunkSlot> = entries.iter().filter_map(|e| e.old_slot).collect();
        let slots = entries.into_iter().map(|e| (e.coord, e.lod, e.slot)).collect();
        self.pending_batch = Some(PendingBatch { fence, cmd, slots, old_slots });
    }

    // ── Block interaction ─────────────────────────────────────────────────────

    /// DDA raycast from `origin` along `dir`. Returns `(hit_block_pos, face_normal)` or `None`.
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(IVec3, IVec3)> {
        let dir = dir.normalize();
        let step = IVec3::new(
            if dir.x >= 0.0 { 1 } else { -1 },
            if dir.y >= 0.0 { 1 } else { -1 },
            if dir.z >= 0.0 { 1 } else { -1 },
        );
        let t_delta = Vec3::new(
            if dir.x.abs() > 1e-10 { 1.0 / dir.x.abs() } else { f32::INFINITY },
            if dir.y.abs() > 1e-10 { 1.0 / dir.y.abs() } else { f32::INFINITY },
            if dir.z.abs() > 1e-10 { 1.0 / dir.z.abs() } else { f32::INFINITY },
        );
        let mut block = IVec3::new(origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32);
        let mut t_max = Vec3::new(
            if dir.x >= 0.0 { (block.x as f32 + 1.0 - origin.x) * t_delta.x } else { (origin.x - block.x as f32) * t_delta.x },
            if dir.y >= 0.0 { (block.y as f32 + 1.0 - origin.y) * t_delta.y } else { (origin.y - block.y as f32) * t_delta.y },
            if dir.z >= 0.0 { (block.z as f32 + 1.0 - origin.z) * t_delta.z } else { (origin.z - block.z as f32) * t_delta.z },
        );
        let mut last_normal = IVec3::ZERO;
        let mut cached: Option<(IVec2, Chunk)> = None;

        loop {
            if block.y >= 0 && block.y < CHUNK_HEIGHT as i32 {
                let coord = IVec2::new(block.x.div_euclid(CHUNK_SIZE as i32), block.z.div_euclid(CHUNK_SIZE as i32));
                let bx = block.x.rem_euclid(CHUNK_SIZE as i32) as usize;
                let by = block.y as usize;
                let bz = block.z.rem_euclid(CHUNK_SIZE as i32) as usize;

                let id = match self.stored_chunks.get(&coord) {
                    Some(c) => c.get(bx, by, bz),
                    None => match &cached {
                        Some((cc, c)) if *cc == coord => c.get(bx, by, bz),
                        _ => {
                            let o = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
                            let c = generate(o);
                            let id = c.get(bx, by, bz);
                            cached = Some((coord, c));
                            id
                        }
                    },
                };

                if is_opaque(id) {
                    return Some((block, last_normal));
                }
            }

            // Advance to next block face
            if t_max.x <= t_max.y && t_max.x <= t_max.z {
                if t_max.x > RAYCAST_REACH { return None; }
                block.x += step.x;
                last_normal = IVec3::new(-step.x, 0, 0);
                t_max.x += t_delta.x;
            } else if t_max.y <= t_max.z {
                if t_max.y > RAYCAST_REACH { return None; }
                block.y += step.y;
                last_normal = IVec3::new(0, -step.y, 0);
                t_max.y += t_delta.y;
            } else {
                if t_max.z > RAYCAST_REACH { return None; }
                block.z += step.z;
                last_normal = IVec3::new(0, 0, -step.z);
                t_max.z += t_delta.z;
            }
        }
    }

    /// Place or remove a block at the given world position and schedule a remesh.
    pub fn set_block(&mut self, world_pos: IVec3, id: BlockId) {
        if world_pos.y < 0 || world_pos.y >= CHUNK_HEIGHT as i32 { return; }
        let coord = IVec2::new(
            world_pos.x.div_euclid(CHUNK_SIZE as i32),
            world_pos.z.div_euclid(CHUNK_SIZE as i32),
        );
        let lx = world_pos.x.rem_euclid(CHUNK_SIZE as i32) as usize;
        let ly = world_pos.y as usize;
        let lz = world_pos.z.rem_euclid(CHUNK_SIZE as i32) as usize;

        // Ensure the chunk is stored (generate if not yet cached).
        if !self.stored_chunks.contains_key(&coord) {
            let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
            self.stored_chunks.insert(coord, generate(origin));
        }

        self.stored_chunks.get_mut(&coord).unwrap().set(lx, ly, lz, id);
        self.modified_set.insert(coord);

        // Refresh face data for this chunk (needed for neighbour AO/culling).
        let fd = ChunkFaceData::extract(self.stored_chunks.get(&coord).unwrap());
        self.face_data.insert(coord, fd);

        self.remesh_chunk(coord);

        // If the modified block is on a chunk border, also remesh the adjacent chunk.
        let dirs: &[(bool, IVec2)] = &[
            (lx == 0,              IVec2::new(-1,  0)),
            (lx == CHUNK_SIZE - 1, IVec2::new( 1,  0)),
            (lz == 0,              IVec2::new( 0, -1)),
            (lz == CHUNK_SIZE - 1, IVec2::new( 0,  1)),
        ];
        for &(on_border, d) in dirs {
            if !on_border { continue; }
            let nb = coord + d;
            if self.face_data.contains_key(&nb) {
                self.remesh_chunk(nb);
            }
        }
    }

    /// Schedule a fresh remesh for `coord`, keeping the old mesh visible until the new one lands.
    fn remesh_chunk(&mut self, coord: IVec2) {
        self.upload_queue.retain(|c| *c != coord);
        if self.mesh_in_flight.contains(&coord) {
            // Rayon task already running — re-mesh once it finishes.
            self.pending_remesh.insert(coord);
        } else {
            // Mesh synchronously so player edits don't wait behind terrain gen tasks.
            self.mesh_chunk_now(coord);
        }
    }

    /// Run all three LOD meshers on the main thread and push the result to the front of the
    /// upload queue. Used for player edits so the mesh is ready before the next GPU batch,
    /// regardless of how saturated the rayon pool is with background terrain work.
    fn mesh_chunk_now(&mut self, coord: IVec2) {
        let neighbors = build_neighbor_masks(coord, &self.face_data);
        let stored = self.stored_chunks.get(&coord).cloned();
        let chunk = if let Some(c) = stored {
            self.pending_chunks.remove(&coord);
            c
        } else {
            let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
            self.pending_chunks.remove(&coord).unwrap_or_else(|| generate(origin))
        };

        let lod0 = mesh_chunk(&chunk, &neighbors);
        let lod1 = mesh_chunk_surface(&chunk, &neighbors);
        let lod2 = mesh_chunk_water(&chunk, &neighbors);

        self.loaded.insert(coord);
        self.upload_queue.retain(|c| *c != coord);
        if !lod0.0.is_empty() {
            self.cpu_meshes.insert(coord, [Arc::new(lod0), Arc::new(lod1), Arc::new(lod2)]);
            // Insert at front so this chunk is in the very next upload batch.
            self.upload_queue.insert(0, coord);
        }
    }

    /// Ensure block data exists in stored_chunks for every chunk that overlaps the player's XZ
    /// footprint. Called once per frame before physics so is_solid never misses a visible chunk.
    ///
    /// A chunk may be fully rendered but have no stored block data if it was meshed before the
    /// player was close (the raw data is discarded after meshing to save RAM). When that happens
    /// we regenerate deterministically from the same noise seed — same result, no visible change.
    pub fn prep_player_collision(&mut self, player_feet: Vec3) {
        const HALF_W: f32 = 0.32; // slightly wider than physics HALF_W for a one-block margin
        let corners = [
            (player_feet.x - HALF_W, player_feet.z - HALF_W),
            (player_feet.x + HALF_W, player_feet.z - HALF_W),
            (player_feet.x - HALF_W, player_feet.z + HALF_W),
            (player_feet.x + HALF_W, player_feet.z + HALF_W),
        ];
        let mut seen = [IVec2::ZERO; 4];
        let mut n = 0usize;
        for (x, z) in corners {
            let cx = x.floor() as i32;
            let cz = z.floor() as i32;
            let coord = IVec2::new(cx.div_euclid(CHUNK_SIZE as i32), cz.div_euclid(CHUNK_SIZE as i32));
            if seen[..n].contains(&coord) { continue; }
            seen[n] = coord; n += 1;
            if !self.stored_chunks.contains_key(&coord) && self.loaded.contains(&coord) {
                let origin = IVec3::new(coord.x * CHUNK_SIZE as i32, 0, coord.y * CHUNK_SIZE as i32);
                self.stored_chunks.insert(coord, generate(origin));
            }
        }
    }

    /// Returns true if the block at `pos` is solid (opaque). Treats out-of-world-bounds as solid
    /// below y=0 and open above the world height. Unloaded chunks are treated as air — the
    /// stored_chunks window (INTERACT_RADIUS around the player) always covers walking range.
    pub fn is_solid(&self, pos: IVec3) -> bool {
        if pos.y < 0 { return true; }
        if pos.y >= CHUNK_HEIGHT as i32 { return false; }
        let coord = IVec2::new(
            pos.x.div_euclid(CHUNK_SIZE as i32),
            pos.z.div_euclid(CHUNK_SIZE as i32),
        );
        let bx = pos.x.rem_euclid(CHUNK_SIZE as i32) as usize;
        let bz = pos.z.rem_euclid(CHUNK_SIZE as i32) as usize;
        self.stored_chunks
            .get(&coord)
            .map_or(false, |c| is_opaque(c.get(bx, pos.y as usize, bz)))
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
                }
            }
        }
        if !self.staging_vb_ptr.is_null() {
            unsafe {
                ctx.device.unmap_memory(self.staging_vm);
                ctx.device.destroy_buffer(self.staging_vb, None);
                ctx.device.free_memory(self.staging_vm, None);
                ctx.device.unmap_memory(self.staging_im);
                ctx.device.destroy_buffer(self.staging_ib, None);
                ctx.device.free_memory(self.staging_im, None);
            }
            self.staging_vb_ptr = std::ptr::null_mut();
            self.staging_ib_ptr = std::ptr::null_mut();
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

// World is only ever used from the main thread; the raw staging pointers are safe to Send.
unsafe impl Send for World {}

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
