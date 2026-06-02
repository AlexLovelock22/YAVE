use anyhow::Result;
use ash::vk;
use bytemuck::bytes_of;

use super::{
    context::VulkanContext,
    mesh::PushConstants,
    pipeline::{self, Pipeline},
    swapchain::{self, Swapchain},
    texture::TextureArray,
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    pub ctx: VulkanContext,
    pub swapchain: Swapchain,
    pub pipeline: Pipeline,
    texture: TextureArray,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    // Per-frame: one image_available semaphore + one fence
    image_available: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    // Per-swapchain-image: one render_finished semaphore.
    // Indexed by the acquired image index so we never reuse a semaphore that
    // may still be consumed by an in-flight presentation of that same image.
    render_finished: Vec<vk::Semaphore>,
    current_frame: usize,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let size = window.inner_size();
        let ctx = VulkanContext::new(window)?;

        let pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: ctx.graphics_family,
            ..Default::default()
        };
        let command_pool =
            unsafe { ctx.device.create_command_pool(&pool_info, None)? };

        let formats = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_formats(ctx.physical_device, ctx.surface)?
        };
        let color_format = super::swapchain::choose_surface_format(&formats);

        let texture  = TextureArray::new(&ctx, command_pool)?;
        let pipeline = Pipeline::new(&ctx, color_format, texture.desc_layout)?;
        let swapchain =
            Swapchain::new(&ctx, pipeline.render_pass, size.width, size.height)?;

        let cmd_alloc = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };
        let command_buffers = unsafe { ctx.device.allocate_command_buffers(&cmd_alloc)? };

        let sem_info = vk::SemaphoreCreateInfo { ..Default::default() };
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let image_available = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| unsafe { ctx.device.create_semaphore(&sem_info, None).map_err(Into::into) })
            .collect::<Result<Vec<_>>>()?;
        let in_flight = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| unsafe { ctx.device.create_fence(&fence_info, None).map_err(Into::into) })
            .collect::<Result<Vec<_>>>()?;
        let render_finished = (0..swapchain.images.len())
            .map(|_| unsafe { ctx.device.create_semaphore(&sem_info, None).map_err(Into::into) })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            ctx,
            swapchain,
            pipeline,
            texture,
            command_pool,
            command_buffers,
            image_available,
            in_flight,
            render_finished,
            current_frame: 0,
        })
    }

    pub fn draw_frame(
        &mut self,
        render_bufs: Option<(vk::Buffer, vk::Buffer)>,
        indirect: Option<(vk::Buffer, u32)>,
        push: PushConstants,
    ) -> Result<()> {
        let fences = [self.in_flight[self.current_frame]];
        unsafe { self.ctx.device.wait_for_fences(&fences, true, u64::MAX)? };

        let acquire_result = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.image_available[self.current_frame],
                vk::Fence::null(),
            )
        };

        let image_index = match acquire_result {
            Ok((idx, _)) => idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(()),
            Err(e) => return Err(e.into()),
        };

        unsafe { self.ctx.device.reset_fences(&fences)? };

        let cmd = self.command_buffers[self.current_frame];
        unsafe {
            self.ctx
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?
        };

        record_command_buffer(
            &self.ctx,
            cmd,
            &self.swapchain,
            &self.pipeline,
            render_bufs,
            indirect,
            push,
            self.texture.desc_set,
            image_index as usize,
        )?;

        let wait_semaphores = [self.image_available[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished[image_index as usize]];
        let cmds = [cmd];
        let submit = vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: cmds.len() as u32,
            p_command_buffers: cmds.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };
        unsafe {
            self.ctx
                .device
                .queue_submit(self.ctx.graphics_queue, &[submit], self.in_flight[self.current_frame])?
        };

        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: signal_semaphores.len() as u32,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: swapchains.len() as u32,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };
        match unsafe {
            self.swapchain
                .loader
                .queue_present(self.ctx.present_queue, &present_info)
        } {
            Ok(_)
            | Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
            | Err(vk::Result::SUBOPTIMAL_KHR) => {}
            Err(e) => return Err(e.into()),
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.swapchain
            .recreate(&self.ctx, self.pipeline.render_pass, width, height)
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for &s in &self.image_available {
                self.ctx.device.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished {
                self.ctx.device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight {
                self.ctx.device.destroy_fence(f, None);
            }
            swapchain::destroy(&self.ctx, &self.swapchain);
            pipeline::destroy(&self.ctx, &self.pipeline);
            self.texture.destroy(&self.ctx);
            self.ctx.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn record_command_buffer(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    sc: &Swapchain,
    pipeline: &Pipeline,
    render_bufs: Option<(vk::Buffer, vk::Buffer)>,
    indirect: Option<(vk::Buffer, u32)>,
    push: PushConstants,
    desc_set: vk::DescriptorSet,
    image_index: usize,
) -> Result<()> {
    let begin = vk::CommandBufferBeginInfo { ..Default::default() };
    unsafe { ctx.device.begin_command_buffer(cmd, &begin)? };

    let clear_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.05, 0.05, 0.07, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];
    let rp_begin = vk::RenderPassBeginInfo {
        render_pass: pipeline.render_pass,
        framebuffer: sc.framebuffers[image_index],
        render_area: vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: sc.extent,
        },
        clear_value_count: clear_values.len() as u32,
        p_clear_values: clear_values.as_ptr(),
        ..Default::default()
    };

    unsafe {
        ctx.device
            .cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
        ctx.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout,
            0, &[desc_set], &[],
        );

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: sc.extent.width as f32,
            height: sc.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        ctx.device.cmd_set_viewport(cmd, 0, &[viewport]);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: sc.extent,
        };
        ctx.device.cmd_set_scissor(cmd, 0, &[scissor]);

        // Push constant is the same for all meshes (world-space vertices, one camera MVP)
        ctx.device.cmd_push_constants(
            cmd,
            pipeline.layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytes_of(&push),
        );

        if let (Some((vb, ib)), Some((ind_buf, draw_count))) = (render_bufs, indirect) {
            if draw_count > 0 {
                // Ensure all preceding staging→arena DMA copies are visible to the
                // vertex/index fetch stage and that the CPU write to the indirect buffer
                // is visible to the DRAW_INDIRECT stage.
                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::DRAW_INDIRECT,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
                        dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                            | vk::AccessFlags::INDEX_READ
                            | vk::AccessFlags::INDIRECT_COMMAND_READ,
                        ..Default::default()
                    }],
                    &[],
                    &[],
                );
                ctx.device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
                ctx.device.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);
                ctx.device.cmd_draw_indexed_indirect(
                    cmd,
                    ind_buf,
                    0,
                    draw_count,
                    std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
            }
        }
        ctx.device.cmd_end_render_pass(cmd);
        ctx.device.end_command_buffer(cmd)?;
    }
    Ok(())
}
