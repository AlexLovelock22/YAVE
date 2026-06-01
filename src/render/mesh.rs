use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::models::{block_model::BlockModel, face::FaceDir};
use super::{buffer::{create_staging_and_dst, upload_device_local, upload_via_staging}, context::VulkanContext};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
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
                    vertices.push(Vertex { pos: face.verts[i], normal, uv: face.uvs[i] });
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
                        buffer: dst_vb, offset: 0, size: vk::WHOLE_SIZE,
                        ..Default::default()
                    },
                    vk::BufferMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::INDEX_READ,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: dst_ib, offset: 0, size: vk::WHOLE_SIZE,
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
