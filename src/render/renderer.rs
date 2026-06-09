use anyhow::Result;
use ash::vk;
use bytemuck::bytes_of;
use glam::Vec3;

use super::{
    context::VulkanContext,
    mesh::PushConstants,
    pipeline::{self, Pipeline},
    swapchain::{self, Swapchain},
    texture::TextureArray,
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    pub ctx:      VulkanContext,
    pub swapchain: Swapchain,
    pub pipeline:  Pipeline,
    texture: TextureArray,

    // ── Per-swapchain-image framebuffers ─────────────────────────────────────
    scene_framebuffers: Vec<vk::Framebuffer>,

    // ── Command recording ────────────────────────────────────────────────────
    command_pool:    vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // ── Frame synchronisation ─────────────────────────────────────────────────
    image_available: Vec<vk::Semaphore>,
    in_flight:       Vec<vk::Fence>,
    render_finished: Vec<vk::Semaphore>,
    current_frame:   usize,

    // ── Block target outline (small HOST_VISIBLE buffers) ─────────────────────
    outline_vb:     vk::Buffer,
    outline_vb_mem: vk::DeviceMemory,
    outline_vb_ptr: *mut u8,
    outline_ib:     vk::Buffer,
    outline_ib_mem: vk::DeviceMemory,

    // ── Screen-space crosshair (static HOST_VISIBLE buffers) ─────────────────
    crosshair_vb:     vk::Buffer,
    crosshair_vb_mem: vk::DeviceMemory,
    crosshair_ib:     vk::Buffer,
    crosshair_ib_mem: vk::DeviceMemory,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let size = window.inner_size();
        let ctx  = VulkanContext::new(window)?;

        let pool_info = vk::CommandPoolCreateInfo {
            flags:              vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: ctx.graphics_family,
            ..Default::default()
        };
        let command_pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };

        let formats = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_formats(ctx.physical_device, ctx.surface)?
        };
        let swapchain_fmt = super::swapchain::choose_surface_format(&formats);

        let texture  = TextureArray::new(&ctx, command_pool)?;
        let pipeline = Pipeline::new(&ctx, swapchain_fmt, texture.desc_layout)?;
        let swapchain = Swapchain::new(&ctx, size.width, size.height)?;

        let scene_framebuffers = create_framebuffers(&ctx.device, &pipeline, &swapchain)?;

        let cmd_alloc = vk::CommandBufferAllocateInfo {
            command_pool,
            level:               vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };
        let command_buffers = unsafe { ctx.device.allocate_command_buffers(&cmd_alloc)? };

        let sem_info   = vk::SemaphoreCreateInfo::default();
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

        // Outline: 8 vertices × 12 bytes = 96 bytes VB; 24 indices × 4 bytes = 96 bytes IB.
        let (outline_vb, outline_vb_mem, outline_vb_ptr) =
            alloc_host_buffer(&ctx, 96, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let (outline_ib, outline_ib_mem, outline_ib_ptr) =
            alloc_host_buffer(&ctx, 96, vk::BufferUsageFlags::INDEX_BUFFER)?;
        // Write constant edge indices once.
        let outline_indices: [u32; 24] = [
            0,1, 1,2, 2,3, 3,0,  // bottom face
            4,5, 5,6, 6,7, 7,4,  // top face
            0,4, 1,5, 2,6, 3,7,  // vertical edges
        ];
        unsafe {
            std::ptr::copy_nonoverlapping(
                outline_indices.as_ptr() as *const u8,
                outline_ib_ptr,
                96,
            );
        }

        // Crosshair: 8 vertices × 8 bytes (vec2) = 64 bytes VB; 12 indices × 4 bytes = 48 bytes IB.
        // arm_len and bar_thick are in square-NDC-y units; vertex shader divides x by aspect.
        const ARM: f32 = 0.04;
        const THK: f32 = 0.003;
        let crosshair_verts: [[f32; 2]; 8] = [
            [-ARM, -THK], [ ARM, -THK], [ ARM,  THK], [-ARM,  THK],  // horizontal bar
            [-THK, -ARM], [ THK, -ARM], [ THK,  ARM], [-THK,  ARM],  // vertical bar
        ];
        let crosshair_indices: [u32; 12] = [0,1,2, 0,2,3, 4,5,6, 4,6,7];

        let (crosshair_vb, crosshair_vb_mem, crosshair_vb_ptr) =
            alloc_host_buffer(&ctx, 64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let (crosshair_ib, crosshair_ib_mem, crosshair_ib_ptr) =
            alloc_host_buffer(&ctx, 48, vk::BufferUsageFlags::INDEX_BUFFER)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                crosshair_verts.as_ptr() as *const u8,
                crosshair_vb_ptr,
                64,
            );
            std::ptr::copy_nonoverlapping(
                crosshair_indices.as_ptr() as *const u8,
                crosshair_ib_ptr,
                48,
            );
        }

        Ok(Self {
            ctx,
            swapchain,
            pipeline,
            texture,
            scene_framebuffers,
            command_pool,
            command_buffers,
            image_available,
            in_flight,
            render_finished,
            current_frame: 0,
            outline_vb,
            outline_vb_mem,
            outline_vb_ptr,
            outline_ib,
            outline_ib_mem,
            crosshair_vb,
            crosshair_vb_mem,
            crosshair_ib,
            crosshair_ib_mem,
        })
    }

    /// Wait for the in-flight fence and acquire the next swapchain image.
    /// Returns `Ok(None)` if the swapchain is out of date (caller should skip the frame).
    pub fn begin_frame(&mut self) -> Result<Option<u32>> {
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
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        unsafe { self.ctx.device.reset_fences(&fences)? };
        Ok(Some(image_index))
    }

    pub fn end_frame(
        &mut self,
        image_index:    u32,
        render_bufs:    Option<(vk::Buffer, vk::Buffer)>,
        indirect:       Option<(vk::Buffer, u32)>,
        water_indirect: Option<(vk::Buffer, u32)>,
        push:           PushConstants,
        outline_pos:    Option<Vec3>,
        aspect:         f32,
    ) -> Result<()> {
        let cmd = self.command_buffers[self.current_frame];
        unsafe {
            self.ctx.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?
        };

        // Update outline vertex buffer with the 8 expanded cube corners.
        if let Some(p) = outline_pos {
            let e = 0.005_f32;
            let corners: [[f32; 3]; 8] = [
                [p.x - e,       p.y - e,       p.z - e      ],
                [p.x + 1.0 + e, p.y - e,       p.z - e      ],
                [p.x + 1.0 + e, p.y + 1.0 + e, p.z - e      ],
                [p.x - e,       p.y + 1.0 + e, p.z - e      ],
                [p.x - e,       p.y - e,       p.z + 1.0 + e],
                [p.x + 1.0 + e, p.y - e,       p.z + 1.0 + e],
                [p.x + 1.0 + e, p.y + 1.0 + e, p.z + 1.0 + e],
                [p.x - e,       p.y + 1.0 + e, p.z + 1.0 + e],
            ];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    corners.as_ptr() as *const u8,
                    self.outline_vb_ptr,
                    96,
                );
            }
        }

        let outline_bufs = outline_pos.map(|_| (self.outline_vb, self.outline_ib));

        record(
            &self.ctx,
            cmd,
            &self.swapchain,
            &self.pipeline,
            self.scene_framebuffers[image_index as usize],
            render_bufs,
            indirect,
            water_indirect,
            push,
            self.texture.desc_set,
            outline_bufs,
            (self.crosshair_vb, self.crosshair_ib),
            aspect,
        )?;

        let wait_semaphores   = [self.image_available[self.current_frame]];
        let wait_stages       = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished[image_index as usize]];
        let cmds = [cmd];
        let submit = vk::SubmitInfo {
            wait_semaphore_count:   wait_semaphores.len() as u32,
            p_wait_semaphores:      wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask:  wait_stages.as_ptr(),
            command_buffer_count:   cmds.len() as u32,
            p_command_buffers:      cmds.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores:    signal_semaphores.as_ptr(),
            ..Default::default()
        };
        unsafe {
            self.ctx.device.queue_submit(
                self.ctx.graphics_queue,
                &[submit],
                self.in_flight[self.current_frame],
            )?
        };

        let swapchains    = [self.swapchain.swapchain];
        let image_indices = [image_index];
        let present_info  = vk::PresentInfoKHR {
            wait_semaphore_count: signal_semaphores.len() as u32,
            p_wait_semaphores:    signal_semaphores.as_ptr(),
            swapchain_count:      swapchains.len() as u32,
            p_swapchains:         swapchains.as_ptr(),
            p_image_indices:      image_indices.as_ptr(),
            ..Default::default()
        };
        match unsafe {
            self.swapchain.loader.queue_present(self.ctx.present_queue, &present_info)
        } {
            Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {}
            Err(e) => return Err(e.into()),
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe { self.ctx.device.device_wait_idle()? };

        self.swapchain.recreate(&self.ctx, width, height)?;

        destroy_framebuffers(&self.ctx.device, &self.scene_framebuffers);
        self.scene_framebuffers = create_framebuffers(&self.ctx.device, &self.pipeline, &self.swapchain)?;

        Ok(())
    }

    pub fn command_pool(&self) -> vk::CommandPool { self.command_pool }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            destroy_framebuffers(&self.ctx.device, &self.scene_framebuffers);
            for &s in &self.image_available { self.ctx.device.destroy_semaphore(s, None); }
            for &s in &self.render_finished { self.ctx.device.destroy_semaphore(s, None); }
            for &f in &self.in_flight      { self.ctx.device.destroy_fence(f, None); }
            swapchain::destroy(&self.ctx, &self.swapchain);
            pipeline::destroy(&self.ctx, &self.pipeline);
            self.texture.destroy(&self.ctx);
            self.ctx.device.destroy_command_pool(self.command_pool, None);
            self.ctx.device.destroy_buffer(self.outline_vb, None);
            self.ctx.device.free_memory(self.outline_vb_mem, None);
            self.ctx.device.destroy_buffer(self.outline_ib, None);
            self.ctx.device.free_memory(self.outline_ib_mem, None);
            self.ctx.device.destroy_buffer(self.crosshair_vb, None);
            self.ctx.device.free_memory(self.crosshair_vb_mem, None);
            self.ctx.device.destroy_buffer(self.crosshair_ib, None);
            self.ctx.device.free_memory(self.crosshair_ib_mem, None);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-visible buffer allocation
// ─────────────────────────────────────────────────────────────────────────────

fn alloc_host_buffer(
    ctx:   &VulkanContext,
    size:  u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8)> {
    let buf_info = vk::BufferCreateInfo {
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let buf = unsafe { ctx.device.create_buffer(&buf_info, None)? };
    let reqs = unsafe { ctx.device.get_buffer_memory_requirements(buf) };
    let mem_type = ctx.find_memory_type(
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let alloc = vk::MemoryAllocateInfo {
        allocation_size:   reqs.size,
        memory_type_index: mem_type,
        ..Default::default()
    };
    let mem = unsafe { ctx.device.allocate_memory(&alloc, None)? };
    unsafe { ctx.device.bind_buffer_memory(buf, mem, 0)? };
    let ptr = unsafe {
        ctx.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
    } as *mut u8;
    Ok((buf, mem, ptr))
}

// ─────────────────────────────────────────────────────────────────────────────
// Framebuffer management
// ─────────────────────────────────────────────────────────────────────────────

fn create_framebuffers(
    device:   &ash::Device,
    pipeline: &Pipeline,
    sc:       &Swapchain,
) -> Result<Vec<vk::Framebuffer>> {
    let e = sc.extent;
    sc.image_views.iter()
        .map(|&iv| {
            let atts = [iv, sc.depth_view];
            unsafe {
                device.create_framebuffer(&vk::FramebufferCreateInfo {
                    render_pass:      pipeline.geom_render_pass,
                    attachment_count: atts.len() as u32,
                    p_attachments:    atts.as_ptr(),
                    width: e.width, height: e.height, layers: 1,
                    ..Default::default()
                }, None).map_err(Into::into)
            }
        })
        .collect()
}

fn destroy_framebuffers(device: &ash::Device, fbs: &[vk::Framebuffer]) {
    unsafe {
        for &fb in fbs { device.destroy_framebuffer(fb, None); }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Command recording
// ─────────────────────────────────────────────────────────────────────────────

fn record(
    ctx:            &VulkanContext,
    cmd:            vk::CommandBuffer,
    sc:             &Swapchain,
    pipeline:       &Pipeline,
    scene_fb:       vk::Framebuffer,
    render_bufs:    Option<(vk::Buffer, vk::Buffer)>,
    indirect:       Option<(vk::Buffer, u32)>,
    water_indirect: Option<(vk::Buffer, u32)>,
    push:           PushConstants,
    tex_set:        vk::DescriptorSet,
    outline:        Option<(vk::Buffer, vk::Buffer)>,
    crosshair:      (vk::Buffer, vk::Buffer),
    aspect:         f32,
) -> Result<()> {
    let d  = &ctx.device;
    let ex = sc.extent;

    unsafe { d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())? };

    let clear_vals = [
        vk::ClearValue { color: vk::ClearColorValue { float32: [0.4, 0.65, 1.0, 1.0] } },
        vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
    ];
    let rp_begin = vk::RenderPassBeginInfo {
        render_pass:  pipeline.geom_render_pass,
        framebuffer:  scene_fb,
        render_area:  vk::Rect2D { offset: vk::Offset2D::default(), extent: ex },
        clear_value_count: clear_vals.len() as u32,
        p_clear_values:    clear_vals.as_ptr(),
        ..Default::default()
    };
    unsafe {
        d.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
        d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout,
            0, &[tex_set], &[]);
        set_viewport_scissor(d, cmd, ex);
        d.cmd_push_constants(cmd, pipeline.layout, vk::ShaderStageFlags::VERTEX, 0,
            bytes_of(&push));

        if let (Some((vb, ib)), Some((ind_buf, draw_count))) = (render_bufs, indirect) {
            if draw_count > 0 {
                // Ensure staging DMA copies and indirect-buffer CPU writes are visible.
                d.cmd_pipeline_barrier(
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
                    &[], &[],
                );
                d.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
                d.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);
                d.cmd_draw_indexed_indirect(
                    cmd, ind_buf, 0, draw_count,
                    std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                // Water sub-pass: transparent surfaces with stencil guard.
                if let Some((water_buf, water_count)) = water_indirect {
                    if water_count > 0 {
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS,
                            pipeline.water_pipeline);
                        d.cmd_draw_indexed_indirect(
                            cmd, water_buf, 0, water_count,
                            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        );
                    }
                }
            }
        }
        // Draw block target outline last (depth-test on, no depth write).
        if let Some((ovb, oib)) = outline {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::HOST_WRITE,
                    dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags::INDEX_READ,
                    ..Default::default()
                }],
                &[], &[],
            );
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.outline_pipeline);
            d.cmd_push_constants(cmd, pipeline.outline_layout, vk::ShaderStageFlags::VERTEX,
                0, bytes_of(&push));
            d.cmd_bind_vertex_buffers(cmd, 0, &[ovb], &[0]);
            d.cmd_bind_index_buffer(cmd, oib, 0, vk::IndexType::UINT32);
            d.cmd_draw_indexed(cmd, 24, 1, 0, 0, 0);
        }

        // Crosshair: always-visible inversion-blend + at screen center.
        let (cvb, cib) = crosshair;
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.crosshair_pipeline);
        d.cmd_push_constants(cmd, pipeline.crosshair_layout, vk::ShaderStageFlags::VERTEX,
            0, &aspect.to_le_bytes());
        d.cmd_bind_vertex_buffers(cmd, 0, &[cvb], &[0]);
        d.cmd_bind_index_buffer(cmd, cib, 0, vk::IndexType::UINT32);
        d.cmd_draw_indexed(cmd, 12, 1, 0, 0, 0);

        d.cmd_end_render_pass(cmd);
    }

    unsafe { d.end_command_buffer(cmd)? };
    Ok(())
}

fn set_viewport_scissor(d: &ash::Device, cmd: vk::CommandBuffer, ex: vk::Extent2D) {
    let viewport = vk::Viewport {
        x: 0.0, y: 0.0,
        width:  ex.width  as f32,
        height: ex.height as f32,
        min_depth: 0.0, max_depth: 1.0,
    };
    let scissor = vk::Rect2D { offset: vk::Offset2D::default(), extent: ex };
    unsafe {
        d.cmd_set_viewport(cmd, 0, &[viewport]);
        d.cmd_set_scissor(cmd,  0, &[scissor]);
    }
}
