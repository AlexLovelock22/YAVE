use anyhow::Result;
use ash::vk;

use super::context::VulkanContext;

pub fn create_buffer(
    ctx: &VulkanContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let info = vk::BufferCreateInfo {
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let buffer = unsafe { ctx.device.create_buffer(&info, None)? };

    let reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
    let mem_type = ctx.find_memory_type(reqs.memory_type_bits, props)?;
    let alloc = vk::MemoryAllocateInfo {
        allocation_size: reqs.size,
        memory_type_index: mem_type,
        ..Default::default()
    };
    let memory = unsafe { ctx.device.allocate_memory(&alloc, None)? };
    unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0)? };

    Ok((buffer, memory))
}

/// Uploads `data` into a DEVICE_LOCAL buffer via a temporary staging buffer.
/// Blocks until the copy completes. Use for large, rarely-updated buffers (e.g. combined mesh).
pub fn upload_device_local<T: Copy>(
    ctx: &VulkanContext,
    command_pool: vk::CommandPool,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let size = (std::mem::size_of::<T>() * data.len()) as vk::DeviceSize;

    let (staging_buf, staging_mem) = create_buffer(
        ctx, size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = ctx.device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())? as *mut T;
        ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        ctx.device.unmap_memory(staging_mem);
    }

    let (dst_buf, dst_mem) = create_buffer(
        ctx, size,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Submit copy and wait — acceptable because this only runs when the world changes
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let cmd = unsafe { ctx.device.allocate_command_buffers(&alloc_info)?[0] };
    unsafe {
        ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        })?;
        ctx.device.cmd_copy_buffer(cmd, staging_buf, dst_buf, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
        ctx.device.end_command_buffer(cmd)?;
        let cmds = [cmd];
        ctx.device.queue_submit(ctx.graphics_queue, &[vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: cmds.as_ptr(),
            ..Default::default()
        }], vk::Fence::null())?;
        ctx.device.queue_wait_idle(ctx.graphics_queue)?;
        ctx.device.free_command_buffers(command_pool, &cmds);
        ctx.device.destroy_buffer(staging_buf, None);
        ctx.device.free_memory(staging_mem, None);
    }

    Ok((dst_buf, dst_mem))
}

/// Creates a HOST_VISIBLE staging buffer filled with `data` and a matching DEVICE_LOCAL
/// destination buffer. Returns (staging_buf, staging_mem, dst_buf, dst_mem).
/// The caller is responsible for submitting the copy command and freeing staging resources.
pub fn create_staging_and_dst<T: Copy>(
    ctx: &VulkanContext,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, vk::Buffer, vk::DeviceMemory)> {
    let size = (std::mem::size_of::<T>() * data.len()) as vk::DeviceSize;

    let (staging_buf, staging_mem) = create_buffer(
        ctx, size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = ctx.device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())? as *mut T;
        ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        ctx.device.unmap_memory(staging_mem);
    }

    let (dst_buf, dst_mem) = create_buffer(
        ctx, size,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    Ok((staging_buf, staging_mem, dst_buf, dst_mem))
}

/// Uploads `data` into a host-visible buffer (map + memcpy, no staging copy, no GPU stall).
pub fn upload_via_staging<T: Copy>(
    ctx: &VulkanContext,
    _command_pool: vk::CommandPool,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let size = (std::mem::size_of::<T>() * data.len()) as vk::DeviceSize;

    let (buffer, memory) = create_buffer(
        ctx,
        size,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = ctx.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())? as *mut T;
        ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        ctx.device.unmap_memory(memory);
    }

    Ok((buffer, memory))
}
