use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::models::{block_model::BlockModel, face::FaceDir};
use super::{buffer::{upload_device_local, upload_via_staging}, context::VulkanContext};

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

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_buffer(self.vertex_buffer, None);
            ctx.device.free_memory(self.vertex_memory, None);
            ctx.device.destroy_buffer(self.index_buffer, None);
            ctx.device.free_memory(self.index_memory, None);
        }
    }
}
