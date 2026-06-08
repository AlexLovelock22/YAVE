use anyhow::Result;
use ash::vk;
use bytemuck::bytes_of;
use glam::Mat4;

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

    // ── Per-swapchain-extent framebuffers ────────────────────────────────
    scene_framebuffer:      vk::Framebuffer,
    ssao_framebuffer:       vk::Framebuffer,
    blur_framebuffer:       vk::Framebuffer,
    composite_framebuffers: Vec<vk::Framebuffer>,

    // ── SSAO / blur / composite descriptor resources ──────────────────────
    sampler:            vk::Sampler,
    desc_pool:          vk::DescriptorPool,
    ssao_desc_set:      vk::DescriptorSet,
    blur_desc_set:      vk::DescriptorSet,
    composite_desc_set: vk::DescriptorSet,

    // ── Command recording ────────────────────────────────────────────────
    command_pool:    vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // ── Frame synchronisation ─────────────────────────────────────────────
    image_available: Vec<vk::Semaphore>,
    in_flight:       Vec<vk::Fence>,
    render_finished: Vec<vk::Semaphore>,
    current_frame:   usize,
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

        let sampler = create_sampler(&ctx.device)?;

        let (desc_pool, ssao_desc_set, blur_desc_set, composite_desc_set) =
            allocate_desc_sets(&ctx.device, &pipeline)?;

        write_desc_sets(&ctx.device, sampler, &swapchain,
            ssao_desc_set, blur_desc_set, composite_desc_set);

        let (scene_framebuffer, ssao_framebuffer, blur_framebuffer, composite_framebuffers) =
            create_framebuffers(&ctx.device, &pipeline, &swapchain)?;

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

        Ok(Self {
            ctx,
            swapchain,
            pipeline,
            texture,
            scene_framebuffer,
            ssao_framebuffer,
            blur_framebuffer,
            composite_framebuffers,
            sampler,
            desc_pool,
            ssao_desc_set,
            blur_desc_set,
            composite_desc_set,
            command_pool,
            command_buffers,
            image_available,
            in_flight,
            render_finished,
            current_frame: 0,
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

    /// Record all three render passes and submit.
    pub fn end_frame(
        &mut self,
        image_index:    u32,
        render_bufs:    Option<(vk::Buffer, vk::Buffer)>,
        indirect:       Option<(vk::Buffer, u32)>,
        water_indirect: Option<(vk::Buffer, u32)>,
        push:           PushConstants,
        proj:           Mat4,   // projection-only matrix (with Y-flip) for the SSAO pass
    ) -> Result<()> {
        let cmd = self.command_buffers[self.current_frame];
        unsafe {
            self.ctx.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?
        };

        record(
            &self.ctx,
            cmd,
            &self.swapchain,
            &self.pipeline,
            self.scene_framebuffer,
            self.ssao_framebuffer,
            self.blur_framebuffer,
            self.composite_framebuffers[image_index as usize],
            render_bufs,
            indirect,
            water_indirect,
            push,
            proj,
            self.texture.desc_set,
            self.ssao_desc_set,
            self.blur_desc_set,
            self.composite_desc_set,
        )?;

        let wait_semaphores  = [self.image_available[self.current_frame]];
        let wait_stages      = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
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

        // Recreate swapchain + offscreen images.
        self.swapchain.recreate(&self.ctx, width, height)?;

        // Rebuild framebuffers that reference the new image views.
        destroy_framebuffers(&self.ctx.device, &self.composite_framebuffers,
            self.scene_framebuffer, self.ssao_framebuffer, self.blur_framebuffer);
        let (scene_fb, ssao_fb, blur_fb, composite_fbs) =
            create_framebuffers(&self.ctx.device, &self.pipeline, &self.swapchain)?;
        self.scene_framebuffer      = scene_fb;
        self.ssao_framebuffer       = ssao_fb;
        self.blur_framebuffer       = blur_fb;
        self.composite_framebuffers = composite_fbs;

        // Update SSAO/blur/composite descriptor sets with new image views.
        write_desc_sets(&self.ctx.device, self.sampler, &self.swapchain,
            self.ssao_desc_set, self.blur_desc_set, self.composite_desc_set);

        Ok(())
    }

    pub fn command_pool(&self) -> vk::CommandPool { self.command_pool }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            destroy_framebuffers(&self.ctx.device, &self.composite_framebuffers,
                self.scene_framebuffer, self.ssao_framebuffer, self.blur_framebuffer);
            self.ctx.device.destroy_descriptor_pool(self.desc_pool, None);
            self.ctx.device.destroy_sampler(self.sampler, None);
            for &s in &self.image_available { self.ctx.device.destroy_semaphore(s, None); }
            for &s in &self.render_finished { self.ctx.device.destroy_semaphore(s, None); }
            for &f in &self.in_flight      { self.ctx.device.destroy_fence(f, None); }
            swapchain::destroy(&self.ctx, &self.swapchain);
            pipeline::destroy(&self.ctx, &self.pipeline);
            self.texture.destroy(&self.ctx);
            self.ctx.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Framebuffer management
// ─────────────────────────────────────────────────────────────────────────────

fn create_framebuffers(
    device:   &ash::Device,
    pipeline: &Pipeline,
    sc:       &Swapchain,
) -> Result<(vk::Framebuffer, vk::Framebuffer, vk::Framebuffer, Vec<vk::Framebuffer>)> {
    let e = sc.extent;

    // Geometry pass framebuffer: scene_color + depth
    let scene_atts = [sc.scene_color_view, sc.depth_view];
    let scene_fb = unsafe {
        device.create_framebuffer(&vk::FramebufferCreateInfo {
            render_pass:      pipeline.geom_render_pass,
            attachment_count: scene_atts.len() as u32,
            p_attachments:    scene_atts.as_ptr(),
            width: e.width, height: e.height, layers: 1,
            ..Default::default()
        }, None)?
    };

    // SSAO pass framebuffer: raw AO image
    let ssao_atts = [sc.ao_view];
    let ssao_fb = unsafe {
        device.create_framebuffer(&vk::FramebufferCreateInfo {
            render_pass:      pipeline.ssao_render_pass,
            attachment_count: ssao_atts.len() as u32,
            p_attachments:    ssao_atts.as_ptr(),
            width: e.width, height: e.height, layers: 1,
            ..Default::default()
        }, None)?
    };

    // Blur pass framebuffer: blurred AO image
    let blur_atts = [sc.ao_blurred_view];
    let blur_fb = unsafe {
        device.create_framebuffer(&vk::FramebufferCreateInfo {
            render_pass:      pipeline.blur_render_pass,
            attachment_count: blur_atts.len() as u32,
            p_attachments:    blur_atts.as_ptr(),
            width: e.width, height: e.height, layers: 1,
            ..Default::default()
        }, None)?
    };

    // Composite pass framebuffers: one per swapchain image
    let composite_fbs = sc.image_views.iter()
        .map(|&iv| {
            let atts = [iv];
            unsafe {
                device.create_framebuffer(&vk::FramebufferCreateInfo {
                    render_pass:      pipeline.composite_render_pass,
                    attachment_count: atts.len() as u32,
                    p_attachments:    atts.as_ptr(),
                    width: e.width, height: e.height, layers: 1,
                    ..Default::default()
                }, None).map_err(Into::into)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((scene_fb, ssao_fb, blur_fb, composite_fbs))
}

fn destroy_framebuffers(
    device:        &ash::Device,
    composite_fbs: &[vk::Framebuffer],
    scene_fb:      vk::Framebuffer,
    ssao_fb:       vk::Framebuffer,
    blur_fb:       vk::Framebuffer,
) {
    unsafe {
        for &fb in composite_fbs { device.destroy_framebuffer(fb, None); }
        device.destroy_framebuffer(blur_fb,   None);
        device.destroy_framebuffer(ssao_fb,   None);
        device.destroy_framebuffer(scene_fb,  None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampler
// ─────────────────────────────────────────────────────────────────────────────

fn create_sampler(device: &ash::Device) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo {
        mag_filter:     vk::Filter::NEAREST,
        min_filter:     vk::Filter::NEAREST,
        mipmap_mode:    vk::SamplerMipmapMode::NEAREST,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        compare_enable: vk::FALSE,
        compare_op:     vk::CompareOp::ALWAYS,
        max_lod:        0.0,
        border_color:   vk::BorderColor::FLOAT_OPAQUE_BLACK,
        ..Default::default()
    };
    unsafe { device.create_sampler(&info, None).map_err(Into::into) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor sets
// ─────────────────────────────────────────────────────────────────────────────

fn allocate_desc_sets(
    device:   &ash::Device,
    pipeline: &Pipeline,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet, vk::DescriptorSet, vk::DescriptorSet)> {
    let pool_sizes = [vk::DescriptorPoolSize {
        ty:               vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 5, // 1 (SSAO depth) + 2 (blur: ao + depth) + 2 (composite: scene + ao)
    }];
    let pool = unsafe {
        device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo {
            max_sets:        3,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes:    pool_sizes.as_ptr(),
            ..Default::default()
        }, None)?
    };

    let layouts = [
        pipeline.ssao_desc_layout,
        pipeline.blur_desc_layout,
        pipeline.composite_desc_layout,
    ];
    let sets = unsafe {
        device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
            descriptor_pool:      pool,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts:        layouts.as_ptr(),
            ..Default::default()
        })?
    };

    Ok((pool, sets[0], sets[1], sets[2]))
}

fn write_desc_sets(
    device:        &ash::Device,
    sampler:       vk::Sampler,
    sc:            &Swapchain,
    ssao_set:      vk::DescriptorSet,
    blur_set:      vk::DescriptorSet,
    composite_set: vk::DescriptorSet,
) {
    // SSAO set: binding 0 = depth (depth-only view)
    let depth_info = vk::DescriptorImageInfo {
        sampler,
        image_view:   sc.depth_sample_view,
        image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };
    // Blur set: binding 0 = raw AO, binding 1 = depth (for bilateral weighting)
    let raw_ao_info = vk::DescriptorImageInfo {
        sampler,
        image_view:   sc.ao_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let depth_blur_info = vk::DescriptorImageInfo {
        sampler,
        image_view:   sc.depth_sample_view,
        image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };
    // Composite set: binding 0 = scene colour, binding 1 = blurred AO
    let scene_info = vk::DescriptorImageInfo {
        sampler,
        image_view:   sc.scene_color_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let ao_blurred_info = vk::DescriptorImageInfo {
        sampler,
        image_view:   sc.ao_blurred_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };

    let writes = [
        vk::WriteDescriptorSet {
            dst_set:          ssao_set,
            dst_binding:      0,
            descriptor_count: 1,
            descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info:     &depth_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set:          blur_set,
            dst_binding:      0,
            descriptor_count: 1,
            descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info:     &raw_ao_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set:          blur_set,
            dst_binding:      1,
            descriptor_count: 1,
            descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info:     &depth_blur_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set:          composite_set,
            dst_binding:      0,
            descriptor_count: 1,
            descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info:     &scene_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set:          composite_set,
            dst_binding:      1,
            descriptor_count: 1,
            descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info:     &ao_blurred_info,
            ..Default::default()
        },
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Command recording
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn record(
    ctx:            &VulkanContext,
    cmd:            vk::CommandBuffer,
    sc:             &Swapchain,
    pipeline:       &Pipeline,
    scene_fb:       vk::Framebuffer,
    ssao_fb:        vk::Framebuffer,
    blur_fb:        vk::Framebuffer,
    composite_fb:   vk::Framebuffer,
    render_bufs:    Option<(vk::Buffer, vk::Buffer)>,
    indirect:       Option<(vk::Buffer, u32)>,
    water_indirect: Option<(vk::Buffer, u32)>,
    push:           PushConstants,
    proj:           Mat4,
    tex_set:        vk::DescriptorSet,
    ssao_set:       vk::DescriptorSet,
    blur_set:       vk::DescriptorSet,
    composite_set:  vk::DescriptorSet,
) -> Result<()> {
    let d  = &ctx.device;
    let ex = sc.extent;

    unsafe { d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())? };

    // ── Pass 1: Geometry ─────────────────────────────────────────────────
    {
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
            d.cmd_end_render_pass(cmd);
        }
    }

    // ── Pass 2: SSAO ─────────────────────────────────────────────────────
    {
        let clear_vals = [vk::ClearValue { color: vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 0.0] } }];
        let rp_begin = vk::RenderPassBeginInfo {
            render_pass:  pipeline.ssao_render_pass,
            framebuffer:  ssao_fb,
            render_area:  vk::Rect2D { offset: vk::Offset2D::default(), extent: ex },
            clear_value_count: clear_vals.len() as u32,
            p_clear_values:    clear_vals.as_ptr(),
            ..Default::default()
        };

        // SSAO push constants: proj (64 bytes) + inv_proj (64 bytes)
        let inv_proj  = proj.inverse();
        let proj_arr  = proj.to_cols_array();
        let iproj_arr = inv_proj.to_cols_array();
        let mut ssao_pc = [0u8; 128];
        ssao_pc[..64].copy_from_slice(bytemuck::bytes_of(&proj_arr));
        ssao_pc[64..].copy_from_slice(bytemuck::bytes_of(&iproj_arr));

        unsafe {
            d.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.ssao_pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                pipeline.ssao_layout, 0, &[ssao_set], &[]);
            set_viewport_scissor(d, cmd, ex);
            d.cmd_push_constants(cmd, pipeline.ssao_layout,
                vk::ShaderStageFlags::FRAGMENT, 0, &ssao_pc);
            d.cmd_draw(cmd, 3, 1, 0, 0); // fullscreen triangle
            d.cmd_end_render_pass(cmd);
        }
    }

    // ── Pass 3: Bilateral blur ───────────────────────────────────────────
    {
        let rp_begin = vk::RenderPassBeginInfo {
            render_pass:       pipeline.blur_render_pass,
            framebuffer:       blur_fb,
            render_area:       vk::Rect2D { offset: vk::Offset2D::default(), extent: ex },
            clear_value_count: 0,
            ..Default::default()
        };
        unsafe {
            d.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.blur_pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                pipeline.blur_layout, 0, &[blur_set], &[]);
            set_viewport_scissor(d, cmd, ex);
            d.cmd_draw(cmd, 3, 1, 0, 0);
            d.cmd_end_render_pass(cmd);
        }
    }

    // ── Pass 4: Composite ─────────────────────────────────────────────────
    {
        let rp_begin = vk::RenderPassBeginInfo {
            render_pass:       pipeline.composite_render_pass,
            framebuffer:       composite_fb,
            render_area:       vk::Rect2D { offset: vk::Offset2D::default(), extent: ex },
            clear_value_count: 0,
            ..Default::default()
        };
        unsafe {
            d.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.composite_pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                pipeline.composite_layout, 0, &[composite_set], &[]);
            set_viewport_scissor(d, cmd, ex);
            d.cmd_draw(cmd, 3, 1, 0, 0); // fullscreen triangle
            d.cmd_end_render_pass(cmd);
        }
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
