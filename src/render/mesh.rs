use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::models::{block_model::BlockModel, face::FaceDir};
use super::{buffer::{create_buffer, create_staging_and_dst, upload_device_local, upload_via_staging}, context::VulkanContext};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],  // [u_tile,  v_tile + layer_index * 256.0]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PushConstants {
    pub mvp: [[f32; 4]; 4],
}

pub struct GpuMesh {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    pub index_count: u32,
}

/// An async DEVICE_LOCAL mesh upload in progress.
/// Poll `is_ready` each frame; call `into_mesh` once it returns true.
pub struct PendingMeshUpload {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub index_buffer:  vk::Buffer,
    pub index_memory:  vk::DeviceMemory,
    pub index_count:   u32,
    staging_vb: vk::Buffer,
    staging_vm: vk::DeviceMemory,
    staging_ib: vk::Buffer,
    staging_im: vk::DeviceMemory,
    pub fence: vk::Fence,
    cmd: vk::CommandBuffer,
}

impl PendingMeshUpload {
    /// True once the GPU DMA copy has completed.
    pub fn is_ready(&self, ctx: &VulkanContext) -> bool {
        unsafe { ctx.device.get_fence_status(self.fence).unwrap_or(false) }
    }

    /// Call after `is_ready`. Frees staging resources (if owned) and returns the finished mesh.
    pub fn into_mesh(self, ctx: &VulkanContext, pool: vk::CommandPool) -> GpuMesh {
        unsafe {
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.free_command_buffers(pool, &[self.cmd]);
            // Staging may be null when using persistent caller-owned staging buffers.
            if self.staging_vb != vk::Buffer::null() {
                ctx.device.destroy_buffer(self.staging_vb, None);
                ctx.device.free_memory(self.staging_vm, None);
                ctx.device.destroy_buffer(self.staging_ib, None);
                ctx.device.free_memory(self.staging_im, None);
            }
        }
        GpuMesh {
            vertex_buffer: self.vertex_buffer,
            vertex_memory:  self.vertex_memory,
            index_buffer:   self.index_buffer,
            index_memory:   self.index_memory,
            index_count:    self.index_count,
        }
    }

    /// Used during shutdown: blocks until transfer completes, then destroys everything.
    pub fn abort(self, ctx: &VulkanContext, pool: vk::CommandPool) {
        unsafe {
            let _ = ctx.device.wait_for_fences(&[self.fence], true, u64::MAX);
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.free_command_buffers(pool, &[self.cmd]);
            if self.staging_vb != vk::Buffer::null() {
                ctx.device.destroy_buffer(self.staging_vb, None);
                ctx.device.free_memory(self.staging_vm, None);
                ctx.device.destroy_buffer(self.staging_ib, None);
                ctx.device.free_memory(self.staging_im, None);
            }
            // Only destroy dst buffers if we own them (null memory = non-owning view).
            if self.vertex_memory != vk::DeviceMemory::null() {
                ctx.device.destroy_buffer(self.vertex_buffer, None);
                ctx.device.free_memory(self.vertex_memory, None);
                ctx.device.destroy_buffer(self.index_buffer, None);
                ctx.device.free_memory(self.index_memory, None);
            }
        }
    }
}

impl GpuMesh {
    /// Upload a single block model (used for the single-voxel demo).
    pub fn from_model(model: &BlockModel, ctx: &VulkanContext, command_pool: vk::CommandPool) -> Result<Self> {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for dir in FaceDir::ALL {
            if let Some(face) = model.face(dir) {
                let base = vertices.len() as u32;
                let normal = dir.normal();
                for i in 0..4 {
                    vertices.push(Vertex { pos: face.verts[i], normal, uv: [face.uvs[i][0], face.uvs[i][1] + face.texture as f32 * 256.0] });
                }
                indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
            }
        }

        Self::from_data(&vertices, &indices, ctx, command_pool)
    }

    /// Upload pre-built CPU-side vertex and index data (used by the chunk mesher).
    pub fn from_data(
        vertices: &[Vertex],
        indices: &[u32],
        ctx: &VulkanContext,
        command_pool: vk::CommandPool,
    ) -> Result<Self> {
        let (vertex_buffer, vertex_memory) = upload_via_staging(
            ctx, command_pool, vertices, vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let (index_buffer, index_memory) = upload_via_staging(
            ctx, command_pool, indices, vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        Ok(GpuMesh {
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            index_count: indices.len() as u32,
        })
    }

    /// Upload to DEVICE_LOCAL memory via staging. Blocks once per call but gives fast GPU reads.
    /// Use for the combined world mesh that is read every frame but rebuilt infrequently.
    pub fn from_data_device_local(
        vertices: &[Vertex],
        indices: &[u32],
        ctx: &VulkanContext,
        command_pool: vk::CommandPool,
    ) -> Result<Self> {
        let (vertex_buffer, vertex_memory) = upload_device_local(
            ctx, command_pool, vertices, vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let (index_buffer, index_memory) = upload_device_local(
            ctx, command_pool, indices, vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        Ok(GpuMesh { vertex_buffer, vertex_memory, index_buffer, index_memory, index_count: indices.len() as u32 })
    }

    /// Copies from persistent caller-owned staging buffers into persistent caller-owned
    /// DEVICE_LOCAL dst buffers. No allocation — just command recording and submission.
    /// The returned `PendingMeshUpload` does NOT own the dst buffers (memory = null),
    /// so `into_mesh` produces a non-owning view; the caller manages dst lifetimes.
    pub fn begin_copy_to_preallocated(
        staging_vb: vk::Buffer, vb_size: usize,
        staging_ib: vk::Buffer, ib_size: usize,
        dst_vb: vk::Buffer,
        dst_ib: vk::Buffer,
        ctx: &VulkanContext,
        pool: vk::CommandPool,
    ) -> Result<PendingMeshUpload> {
        let vb_bytes = vb_size as vk::DeviceSize;
        let ib_bytes = ib_size as vk::DeviceSize;

        let cmd = unsafe {
            ctx.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0]
        };
        unsafe {
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            })?;
            ctx.device.cmd_copy_buffer(cmd, staging_vb, dst_vb,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: vb_bytes }]);
            ctx.device.cmd_copy_buffer(cmd, staging_ib, dst_ib,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: ib_bytes }]);
            // No pipeline barrier here: adding VERTEX_INPUT as a dst stage would serialise
            // every subsequent render on this queue with the copy, even renders that still use
            // the old buffer.  The barrier is instead emitted in the first render command that
            // actually binds the new buffer (see World::needs_mesh_barrier / Renderer).
            ctx.device.end_command_buffer(cmd)?;

            let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            ctx.device.queue_submit(ctx.graphics_queue, &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: cmds.as_ptr(),
                ..Default::default()
            }], fence)?;

            Ok(PendingMeshUpload {
                // dst buffers are caller-owned; null memory = non-owning view on into_mesh.
                vertex_buffer: dst_vb, vertex_memory: vk::DeviceMemory::null(),
                index_buffer:  dst_ib, index_memory:  vk::DeviceMemory::null(),
                index_count: (ib_size / std::mem::size_of::<u32>()) as u32,
                staging_vb: vk::Buffer::null(), staging_vm: vk::DeviceMemory::null(),
                staging_ib: vk::Buffer::null(), staging_im: vk::DeviceMemory::null(),
                fence, cmd,
            })
        }
    }

    /// Begins an async DEVICE_LOCAL upload. Returns immediately; call `is_ready` each frame.
    /// When ready, call `into_mesh` to obtain the finished `GpuMesh`.
    pub fn begin_upload(
        vertices: &[Vertex],
        indices: &[u32],
        ctx: &VulkanContext,
        pool: vk::CommandPool,
    ) -> Result<PendingMeshUpload> {
        let vb_size = (std::mem::size_of::<Vertex>() * vertices.len()) as vk::DeviceSize;
        let ib_size = (std::mem::size_of::<u32>()    * indices.len())  as vk::DeviceSize;

        let (staging_vb, staging_vm, vertex_buffer, vertex_memory) =
            create_staging_and_dst(ctx, vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let (staging_ib, staging_im, index_buffer, index_memory) =
            create_staging_and_dst(ctx, indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

        // Record: copy both buffers then a pipeline barrier so later same-queue renders
        // see the writes (cross-submission memory visibility on a single queue).
        let cmd = unsafe {
            ctx.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0]
        };
        unsafe {
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            })?;
            ctx.device.cmd_copy_buffer(cmd, staging_vb, vertex_buffer,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: vb_size }]);
            ctx.device.cmd_copy_buffer(cmd, staging_ib, index_buffer,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: ib_size }]);
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[
                    vk::BufferMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: vertex_buffer,
                        offset: 0,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    },
                    vk::BufferMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::INDEX_READ,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: index_buffer,
                        offset: 0,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    },
                ],
                &[],
            );
            ctx.device.end_command_buffer(cmd)?;

            let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            ctx.device.queue_submit(ctx.graphics_queue, &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: cmds.as_ptr(),
                ..Default::default()
            }], fence)?;

            Ok(PendingMeshUpload {
                vertex_buffer, vertex_memory,
                index_buffer, index_memory,
                index_count: indices.len() as u32,
                staging_vb, staging_vm,
                staging_ib, staging_im,
                fence, cmd,
            })
        }
    }

    /// Non-owning view into existing GPU buffers. `destroy()` is a no-op on this.
    pub fn view(vertex_buffer: vk::Buffer, index_buffer: vk::Buffer, index_count: u32) -> Self {
        Self {
            vertex_buffer,
            vertex_memory:  vk::DeviceMemory::null(),
            index_buffer,
            index_memory:   vk::DeviceMemory::null(),
            index_count,
        }
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        if self.vertex_memory == vk::DeviceMemory::null() { return; }
        unsafe {
            ctx.device.destroy_buffer(self.vertex_buffer, None);
            ctx.device.free_memory(self.vertex_memory, None);
            ctx.device.destroy_buffer(self.index_buffer, None);
            ctx.device.free_memory(self.index_memory, None);
        }
    }
}

// ── GPU arena types ───────────────────────────────────────────────────────────

const VB_ALIGN: usize = std::mem::size_of::<Vertex>(); // 32 — vertex_base must be integer
const IB_ALIGN: usize = std::mem::size_of::<u32>();    // 4

/// Location of one LOD mesh inside the arena.
#[derive(Clone, Copy)]
pub struct ChunkSlot {
    pub vb_offset:   usize,
    pub vb_size:     usize,
    pub ib_offset:   usize,
    pub ib_size:     usize,
    pub index_count: u32,
    /// firstIndex for DrawIndexedIndirectCommand (= ib_offset / 4).
    pub first_index: u32,
    /// vertexOffset for DrawIndexedIndirectCommand (= vb_offset / VB_ALIGN as i32).
    pub vertex_base: i32,
}

/// Large DEVICE_LOCAL buffer with a first-fit free-list sub-allocator.
/// Both vertex and index arenas are this type; alignment differs.
pub struct ArenaBuffer {
    pub buffer: vk::Buffer,
    memory:     vk::DeviceMemory,
    pub cap:    usize,
    free_list:  Vec<(usize, usize)>, // (byte_offset, byte_len) sorted, coalesced
    usage:      vk::BufferUsageFlags,
    align:      usize,
}

impl ArenaBuffer {
    pub fn new(ctx: &VulkanContext, cap: usize, usage: vk::BufferUsageFlags, align: usize) -> Result<Self> {
        let (buffer, memory) = create_buffer(
            ctx, cap as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        Ok(Self { buffer, memory, cap, free_list: vec![(0, cap)], usage, align })
    }

    pub fn alloc(&mut self, size: usize) -> Option<usize> {
        if size == 0 { return Some(0); }
        let size = ((size + self.align - 1) / self.align) * self.align;
        for i in 0..self.free_list.len() {
            let (off, len) = self.free_list[i];
            if len >= size {
                if len > size {
                    self.free_list[i] = (off + size, len - size);
                } else {
                    self.free_list.remove(i);
                }
                return Some(off);
            }
        }
        None
    }

    pub fn free(&mut self, offset: usize, size: usize) {
        if size == 0 { return; }
        let size = ((size + self.align - 1) / self.align) * self.align;
        let pos = self.free_list.partition_point(|&(o, _)| o < offset);
        self.free_list.insert(pos, (offset, size));
        // Coalesce with next
        if pos + 1 < self.free_list.len() {
            let (o, s) = self.free_list[pos];
            if o + s == self.free_list[pos + 1].0 {
                let ns = self.free_list.remove(pos + 1).1;
                self.free_list[pos].1 = s + ns;
            }
        }
        // Coalesce with prev
        if pos > 0 {
            let (po, ps) = self.free_list[pos - 1];
            if po + ps == self.free_list[pos].0 {
                let cs = self.free_list.remove(pos).1;
                self.free_list[pos - 1].1 = ps + cs;
            }
        }
    }

    /// Grow to fit `needed` extra bytes. GPU-copies old content; blocks until complete.
    pub fn ensure_cap(&mut self, ctx: &VulkanContext, pool: vk::CommandPool, needed: usize) -> Result<()> {
        if needed <= self.cap { return Ok(()); }
        let new_cap = needed.next_power_of_two().max(self.cap * 2);
        let (new_buf, new_mem) = create_buffer(
            ctx, new_cap as vk::DeviceSize,
            self.usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            let cmd_info = vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let cmd = ctx.device.allocate_command_buffers(&cmd_info)?[0];
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            })?;
            ctx.device.cmd_copy_buffer(cmd, self.buffer, new_buf, &[vk::BufferCopy {
                src_offset: 0, dst_offset: 0, size: self.cap as vk::DeviceSize,
            }]);
            ctx.device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            ctx.device.queue_submit(ctx.graphics_queue, &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: cmds.as_ptr(),
                ..Default::default()
            }], vk::Fence::null())?;
            ctx.device.queue_wait_idle(ctx.graphics_queue)?;
            ctx.device.free_command_buffers(pool, &cmds);
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
        self.free(self.cap, new_cap - self.cap);
        self.buffer = new_buf;
        self.memory = new_mem;
        self.cap    = new_cap;
        Ok(())
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

/// Persistently-mapped HOST_COHERENT buffer for indirect draw commands.
/// Written by the CPU each frame after frustum culling; read by the GPU.
pub struct IndirectBuffer {
    pub buffer: vk::Buffer,
    memory:     vk::DeviceMemory,
    ptr:        *mut vk::DrawIndexedIndirectCommand,
    pub cap:    usize, // command capacity
}

unsafe impl Send for IndirectBuffer {}

impl IndirectBuffer {
    pub fn new(ctx: &VulkanContext, cap: usize) -> Result<Self> {
        let byte_size = (cap * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()) as vk::DeviceSize;
        let (buffer, memory) = create_buffer(
            ctx, byte_size,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ptr = unsafe {
            ctx.device.map_memory(memory, 0, byte_size, vk::MemoryMapFlags::empty())?
                as *mut vk::DrawIndexedIndirectCommand
        };
        Ok(Self { buffer, memory, ptr, cap })
    }

    /// Write `cmds` into the mapped buffer. Must be called before submitting the draw.
    pub fn write(&self, cmds: &[vk::DrawIndexedIndirectCommand]) {
        debug_assert!(cmds.len() <= self.cap);
        unsafe { std::ptr::copy_nonoverlapping(cmds.as_ptr(), self.ptr, cmds.len()); }
    }

    pub fn ensure_cap(&mut self, ctx: &VulkanContext, needed: usize) -> Result<()> {
        if needed <= self.cap { return Ok(()); }
        self.destroy(ctx);
        *self = Self::new(ctx, needed.next_power_of_two())?;
        Ok(())
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.unmap_memory(self.memory);
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

/// In-flight async upload of one chunk LOD mesh into the arena (staging → DEVICE_LOCAL).
pub struct ChunkUpload {
    pub fence:  vk::Fence,
    cmd:        vk::CommandBuffer,
    staging_vb: vk::Buffer,
    staging_vm: vk::DeviceMemory,
    staging_ib: vk::Buffer,
    staging_im: vk::DeviceMemory,
    pub slot:   ChunkSlot,
}

impl ChunkUpload {
    /// Allocate staging memory, fill it, and submit a copy into the arena at `slot`.
    pub fn begin(
        vertices: &[Vertex],
        indices:  &[u32],
        slot:     ChunkSlot,
        arena_vb: vk::Buffer,
        arena_ib: vk::Buffer,
        ctx:      &VulkanContext,
        pool:     vk::CommandPool,
    ) -> Result<Self> {
        let vb_bytes = slot.vb_size as vk::DeviceSize;
        let ib_bytes = slot.ib_size as vk::DeviceSize;

        let (staging_vb, staging_vm) = create_buffer(ctx, vb_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let (staging_ib, staging_im) = create_buffer(ctx, ib_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;

        unsafe {
            let p = ctx.device.map_memory(staging_vm, 0, vb_bytes, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(vertices.as_ptr() as *const u8, p, vb_bytes as usize);
            ctx.device.unmap_memory(staging_vm);

            let p = ctx.device.map_memory(staging_im, 0, ib_bytes, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(indices.as_ptr() as *const u8, p, ib_bytes as usize);
            ctx.device.unmap_memory(staging_im);
        }

        let cmd = unsafe {
            ctx.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0]
        };
        unsafe {
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            })?;
            ctx.device.cmd_copy_buffer(cmd, staging_vb, arena_vb,
                &[vk::BufferCopy { src_offset: 0, dst_offset: slot.vb_offset as vk::DeviceSize, size: vb_bytes }]);
            ctx.device.cmd_copy_buffer(cmd, staging_ib, arena_ib,
                &[vk::BufferCopy { src_offset: 0, dst_offset: slot.ib_offset as vk::DeviceSize, size: ib_bytes }]);
            ctx.device.end_command_buffer(cmd)?;
        }

        let fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let cmds = [cmd];
        unsafe {
            ctx.device.queue_submit(ctx.graphics_queue, &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: cmds.as_ptr(),
                ..Default::default()
            }], fence)?;
        }

        Ok(Self { fence, cmd, staging_vb, staging_vm, staging_ib, staging_im, slot })
    }

    pub fn is_ready(&self, ctx: &VulkanContext) -> bool {
        unsafe { ctx.device.get_fence_status(self.fence).unwrap_or(false) }
    }

    pub fn finish(self, ctx: &VulkanContext, pool: vk::CommandPool) -> ChunkSlot {
        unsafe {
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.free_command_buffers(pool, &[self.cmd]);
            ctx.device.destroy_buffer(self.staging_vb, None);
            ctx.device.free_memory(self.staging_vm, None);
            ctx.device.destroy_buffer(self.staging_ib, None);
            ctx.device.free_memory(self.staging_im, None);
        }
        self.slot
    }

    pub fn abort(self, ctx: &VulkanContext, pool: vk::CommandPool) {
        unsafe { let _ = ctx.device.wait_for_fences(&[self.fence], true, u64::MAX); }
        self.finish(ctx, pool);
    }
}
