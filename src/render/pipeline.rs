use anyhow::Result;
use ash::vk;
use std::mem;

use super::context::VulkanContext;
use super::mesh::Vertex;

/// Render passes and graphics pipelines used by the renderer.
pub struct Pipeline {
    // ── render pass ─────────────────────────────────────────────────────────
    pub geom_render_pass: vk::RenderPass,

    // ── geometry (terrain + water) ───────────────────────────────────────────
    pub layout:         vk::PipelineLayout,
    pub pipeline:       vk::Pipeline,
    pub water_pipeline: vk::Pipeline,

    // ── block target outline (LINE_LIST, no descriptor sets) ─────────────────
    pub outline_layout:   vk::PipelineLayout,
    pub outline_pipeline: vk::Pipeline,

    // ── screen-space crosshair (TRIANGLE_LIST, inversion blend, no depth test) ─
    pub crosshair_layout:   vk::PipelineLayout,
    pub crosshair_pipeline: vk::Pipeline,
}

impl Pipeline {
    pub fn new(
        ctx:             &VulkanContext,
        swapchain_fmt:   vk::Format,
        tex_desc_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        let dev = &ctx.device;

        let geom_render_pass = create_geom_render_pass(dev, swapchain_fmt)?;
        let (layout, pipeline) = create_geom_pipeline(dev, geom_render_pass, tex_desc_layout, true)?;
        let water_pipeline = create_water_pipeline(dev, geom_render_pass, layout)?;
        let (outline_layout, outline_pipeline) = create_outline_pipeline(dev, geom_render_pass)?;
        let (crosshair_layout, crosshair_pipeline) = create_crosshair_pipeline(dev, geom_render_pass)?;

        Ok(Self {
            geom_render_pass,
            layout,
            pipeline,
            water_pipeline,
            outline_layout,
            outline_pipeline,
            crosshair_layout,
            crosshair_pipeline,
        })
    }
}

pub fn destroy(ctx: &VulkanContext, p: &Pipeline) {
    unsafe {
        let d = &ctx.device;
        d.destroy_pipeline(p.crosshair_pipeline,          None);
        d.destroy_pipeline_layout(p.crosshair_layout,     None);
        d.destroy_pipeline(p.outline_pipeline,            None);
        d.destroy_pipeline_layout(p.outline_layout,       None);
        d.destroy_pipeline(p.water_pipeline,          None);
        d.destroy_pipeline(p.pipeline,                None);
        d.destroy_pipeline_layout(p.layout,           None);
        d.destroy_render_pass(p.geom_render_pass,     None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render pass
// ─────────────────────────────────────────────────────────────────────────────

fn create_geom_render_pass(device: &ash::Device, color_format: vk::Format) -> Result<vk::RenderPass> {
    let color_att = vk::AttachmentDescription {
        format:           color_format,
        samples:          vk::SampleCountFlags::TYPE_1,
        load_op:          vk::AttachmentLoadOp::CLEAR,
        store_op:         vk::AttachmentStoreOp::STORE,
        stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:   vk::ImageLayout::UNDEFINED,
        final_layout:     vk::ImageLayout::PRESENT_SRC_KHR,
        ..Default::default()
    };
    let depth_att = vk::AttachmentDescription {
        format:           vk::Format::D32_SFLOAT_S8_UINT,
        samples:          vk::SampleCountFlags::TYPE_1,
        load_op:          vk::AttachmentLoadOp::CLEAR,
        store_op:         vk::AttachmentStoreOp::DONT_CARE,
        stencil_load_op:  vk::AttachmentLoadOp::CLEAR,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:   vk::ImageLayout::UNDEFINED,
        final_layout:     vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ..Default::default()
    };

    let color_ref = [vk::AttachmentReference {
        attachment: 0,
        layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_ref = vk::AttachmentReference {
        attachment: 1,
        layout:     vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    let subpass = vk::SubpassDescription {
        pipeline_bind_point:        vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count:     color_ref.len() as u32,
        p_color_attachments:        color_ref.as_ptr(),
        p_depth_stencil_attachment: &depth_ref,
        ..Default::default()
    };

    // Wait for swapchain image acquisition before writing color/depth.
    let dep = vk::SubpassDependency {
        src_subpass:     vk::SUBPASS_EXTERNAL,
        dst_subpass:     0,
        src_stage_mask:  vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                       | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        dst_stage_mask:  vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                       | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                       | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ..Default::default()
    };

    let atts = [color_att, depth_att];
    let subs = [subpass];
    let deps = [dep];
    let info = vk::RenderPassCreateInfo {
        attachment_count: atts.len() as u32,
        p_attachments:    atts.as_ptr(),
        subpass_count:    subs.len() as u32,
        p_subpasses:      subs.as_ptr(),
        dependency_count: deps.len() as u32,
        p_dependencies:   deps.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_render_pass(&info, None).map_err(Into::into) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipelines
// ─────────────────────────────────────────────────────────────────────────────

fn create_geom_pipeline(
    device:      &ash::Device,
    render_pass: vk::RenderPass,
    desc_layout: vk::DescriptorSetLayout,
    depth_write: bool,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/voxel.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/voxel.frag.spv"));
    let vert_module = create_shader_module(device, vert_spv)?;
    let frag_module = create_shader_module(device, frag_spv)?;

    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::VERTEX,
            module: vert_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
    ];

    let binding = vk::VertexInputBindingDescription {
        binding:    0,
        stride:     mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    };
    let attrs = [
        vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0  },
        vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 12 },
        vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R32G32_SFLOAT,    offset: 24 },
        vk::VertexInputAttributeDescription { location: 3, binding: 0, format: vk::Format::R32_SFLOAT,       offset: 32 },
    ];
    let bindings = [binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count:   bindings.len() as u32,
        p_vertex_binding_descriptions:      bindings.as_ptr(),
        vertex_attribute_description_count: attrs.len() as u32,
        p_vertex_attribute_descriptions:    attrs.as_ptr(),
        ..Default::default()
    };
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        scissor_count:  1,
        ..Default::default()
    };
    let rasterizer = vk::PipelineRasterizationStateCreateInfo {
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode:    vk::CullModeFlags::BACK,
        front_face:   vk::FrontFace::COUNTER_CLOCKWISE,
        line_width:   1.0,
        ..Default::default()
    };
    let multisampling = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable:  vk::TRUE,
        depth_write_enable: if depth_write { vk::TRUE } else { vk::FALSE },
        depth_compare_op:   vk::CompareOp::LESS,
        ..Default::default()
    };
    let blend_att = vk::PipelineColorBlendAttachmentState {
        color_write_mask:        vk::ColorComponentFlags::RGBA,
        blend_enable:            vk::TRUE,
        src_color_blend_factor:  vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor:  vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op:          vk::BlendOp::ADD,
        src_alpha_blend_factor:  vk::BlendFactor::ONE,
        dst_alpha_blend_factor:  vk::BlendFactor::ZERO,
        alpha_blend_op:          vk::BlendOp::ADD,
    };
    let blend_atts = [blend_att];
    let color_blending = vk::PipelineColorBlendStateCreateInfo {
        attachment_count: blend_atts.len() as u32,
        p_attachments:    blend_atts.as_ptr(),
        ..Default::default()
    };
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state  = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states:    dynamic_states.as_ptr(),
        ..Default::default()
    };

    let push_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset:      0,
        size:        64, // mat4 MVP
    };
    let set_layouts = [desc_layout];
    let layout_info = vk::PipelineLayoutCreateInfo {
        set_layout_count:          set_layouts.len() as u32,
        p_set_layouts:             set_layouts.as_ptr(),
        push_constant_range_count: 1,
        p_push_constant_ranges:    &push_range,
        ..Default::default()
    };
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count:             stages.len() as u32,
        p_stages:                stages.as_ptr(),
        p_vertex_input_state:    &vertex_input,
        p_input_assembly_state:  &input_assembly,
        p_viewport_state:        &viewport_state,
        p_rasterization_state:   &rasterizer,
        p_multisample_state:     &multisampling,
        p_depth_stencil_state:   &depth_stencil,
        p_color_blend_state:     &color_blending,
        p_dynamic_state:         &dynamic_state,
        layout,
        render_pass,
        subpass:                 0,
        base_pipeline_index:     -1,
        ..Default::default()
    };
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?[0]
    };

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    Ok((layout, pipeline))
}

fn create_water_pipeline(
    device:      &ash::Device,
    render_pass: vk::RenderPass,
    layout:      vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/voxel.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/voxel.frag.spv"));
    let vert_module = create_shader_module(device, vert_spv)?;
    let frag_module = create_shader_module(device, frag_spv)?;

    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::VERTEX,
            module: vert_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
    ];

    let binding = vk::VertexInputBindingDescription {
        binding:    0,
        stride:     mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    };
    let attrs = [
        vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0  },
        vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 12 },
        vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R32G32_SFLOAT,    offset: 24 },
        vk::VertexInputAttributeDescription { location: 3, binding: 0, format: vk::Format::R32_SFLOAT,       offset: 32 },
    ];
    let bindings = [binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count:   bindings.len() as u32,
        p_vertex_binding_descriptions:      bindings.as_ptr(),
        vertex_attribute_description_count: attrs.len() as u32,
        p_vertex_attribute_descriptions:    attrs.as_ptr(),
        ..Default::default()
    };
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        scissor_count:  1,
        ..Default::default()
    };
    let rasterizer = vk::PipelineRasterizationStateCreateInfo {
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode:    vk::CullModeFlags::NONE,
        front_face:   vk::FrontFace::COUNTER_CLOCKWISE,
        line_width:   1.0,
        ..Default::default()
    };
    let multisampling = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let water_stencil = vk::StencilOpState {
        fail_op:       vk::StencilOp::KEEP,
        pass_op:       vk::StencilOp::INCREMENT_AND_CLAMP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op:    vk::CompareOp::EQUAL,
        compare_mask:  0xFF,
        write_mask:    0xFF,
        reference:     0,
    };
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable:   vk::TRUE,
        depth_write_enable:  vk::TRUE,
        depth_compare_op:    vk::CompareOp::LESS_OR_EQUAL,
        stencil_test_enable: vk::TRUE,
        front:               water_stencil,
        back:                water_stencil,
        ..Default::default()
    };
    let blend_att = vk::PipelineColorBlendAttachmentState {
        color_write_mask:       vk::ColorComponentFlags::RGBA,
        blend_enable:           vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op:         vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op:         vk::BlendOp::ADD,
    };
    let blend_atts = [blend_att];
    let color_blending = vk::PipelineColorBlendStateCreateInfo {
        attachment_count: blend_atts.len() as u32,
        p_attachments:    blend_atts.as_ptr(),
        ..Default::default()
    };
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state  = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states:    dynamic_states.as_ptr(),
        ..Default::default()
    };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count:           stages.len() as u32,
        p_stages:              stages.as_ptr(),
        p_vertex_input_state:  &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_viewport_state:      &viewport_state,
        p_rasterization_state: &rasterizer,
        p_multisample_state:   &multisampling,
        p_depth_stencil_state: &depth_stencil,
        p_color_blend_state:   &color_blending,
        p_dynamic_state:       &dynamic_state,
        layout,
        render_pass,
        subpass:               0,
        base_pipeline_index:   -1,
        ..Default::default()
    };
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?[0]
    };

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    Ok(pipeline)
}

fn create_crosshair_pipeline(
    device:      &ash::Device,
    render_pass: vk::RenderPass,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/crosshair.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/crosshair.frag.spv"));
    let vert_module = create_shader_module(device, vert_spv)?;
    let frag_module = create_shader_module(device, frag_spv)?;

    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::VERTEX,
            module: vert_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
    ];

    // XY position only, 8 bytes per vertex.
    let binding = vk::VertexInputBindingDescription {
        binding:    0,
        stride:     8,
        input_rate: vk::VertexInputRate::VERTEX,
    };
    let attrs = [vk::VertexInputAttributeDescription {
        location: 0, binding: 0,
        format: vk::Format::R32G32_SFLOAT,
        offset: 0,
    }];
    let bindings = [binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count:   1,
        p_vertex_binding_descriptions:      bindings.as_ptr(),
        vertex_attribute_description_count: 1,
        p_vertex_attribute_descriptions:    attrs.as_ptr(),
        ..Default::default()
    };
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        scissor_count:  1,
        ..Default::default()
    };
    let rasterizer = vk::PipelineRasterizationStateCreateInfo {
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode:    vk::CullModeFlags::NONE,
        front_face:   vk::FrontFace::COUNTER_CLOCKWISE,
        line_width:   1.0,
        ..Default::default()
    };
    let multisampling = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    // No depth test: always renders on top of everything.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable:  vk::FALSE,
        depth_write_enable: vk::FALSE,
        ..Default::default()
    };
    // Inversion blend: result = 1 - dst, so the crosshair is always visible.
    let blend_att = vk::PipelineColorBlendAttachmentState {
        color_write_mask:        vk::ColorComponentFlags::RGBA,
        blend_enable:            vk::TRUE,
        src_color_blend_factor:  vk::BlendFactor::ONE_MINUS_DST_COLOR,
        dst_color_blend_factor:  vk::BlendFactor::ZERO,
        color_blend_op:          vk::BlendOp::ADD,
        src_alpha_blend_factor:  vk::BlendFactor::ZERO,
        dst_alpha_blend_factor:  vk::BlendFactor::ONE,
        alpha_blend_op:          vk::BlendOp::ADD,
    };
    let blend_atts = [blend_att];
    let color_blending = vk::PipelineColorBlendStateCreateInfo {
        attachment_count: 1,
        p_attachments:    blend_atts.as_ptr(),
        ..Default::default()
    };
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state  = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: 2,
        p_dynamic_states:    dynamic_states.as_ptr(),
        ..Default::default()
    };

    let push_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset:      0,
        size:        4, // float aspect
    };
    let layout_info = vk::PipelineLayoutCreateInfo {
        set_layout_count:          0,
        push_constant_range_count: 1,
        p_push_constant_ranges:    &push_range,
        ..Default::default()
    };
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count:             2,
        p_stages:                stages.as_ptr(),
        p_vertex_input_state:    &vertex_input,
        p_input_assembly_state:  &input_assembly,
        p_viewport_state:        &viewport_state,
        p_rasterization_state:   &rasterizer,
        p_multisample_state:     &multisampling,
        p_depth_stencil_state:   &depth_stencil,
        p_color_blend_state:     &color_blending,
        p_dynamic_state:         &dynamic_state,
        layout,
        render_pass,
        subpass:                 0,
        base_pipeline_index:     -1,
        ..Default::default()
    };
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?[0]
    };

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    Ok((layout, pipeline))
}

fn create_outline_pipeline(
    device:      &ash::Device,
    render_pass: vk::RenderPass,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/outline.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/outline.frag.spv"));
    let vert_module = create_shader_module(device, vert_spv)?;
    let frag_module = create_shader_module(device, frag_spv)?;

    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::VERTEX,
            module: vert_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
    ];

    // Just XYZ position, 12 bytes per vertex.
    let binding = vk::VertexInputBindingDescription {
        binding:    0,
        stride:     12,
        input_rate: vk::VertexInputRate::VERTEX,
    };
    let attrs = [vk::VertexInputAttributeDescription {
        location: 0, binding: 0,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    }];
    let bindings = [binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count:   1,
        p_vertex_binding_descriptions:      bindings.as_ptr(),
        vertex_attribute_description_count: 1,
        p_vertex_attribute_descriptions:    attrs.as_ptr(),
        ..Default::default()
    };
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::LINE_LIST,
        ..Default::default()
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        scissor_count:  1,
        ..Default::default()
    };
    let rasterizer = vk::PipelineRasterizationStateCreateInfo {
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode:    vk::CullModeFlags::NONE,
        front_face:   vk::FrontFace::COUNTER_CLOCKWISE,
        line_width:   2.0,
        ..Default::default()
    };
    let multisampling = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    // Depth-test on so the outline is occluded by other geometry; no depth write.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable:  vk::TRUE,
        depth_write_enable: vk::FALSE,
        depth_compare_op:   vk::CompareOp::LESS_OR_EQUAL,
        ..Default::default()
    };
    let blend_att = vk::PipelineColorBlendAttachmentState {
        color_write_mask: vk::ColorComponentFlags::RGBA,
        blend_enable:     vk::FALSE,
        ..Default::default()
    };
    let blend_atts = [blend_att];
    let color_blending = vk::PipelineColorBlendStateCreateInfo {
        attachment_count: 1,
        p_attachments:    blend_atts.as_ptr(),
        ..Default::default()
    };
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state  = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: 2,
        p_dynamic_states:    dynamic_states.as_ptr(),
        ..Default::default()
    };

    let push_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset:      0,
        size:        64, // mat4 MVP
    };
    let layout_info = vk::PipelineLayoutCreateInfo {
        set_layout_count:          0,
        push_constant_range_count: 1,
        p_push_constant_ranges:    &push_range,
        ..Default::default()
    };
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count:             2,
        p_stages:                stages.as_ptr(),
        p_vertex_input_state:    &vertex_input,
        p_input_assembly_state:  &input_assembly,
        p_viewport_state:        &viewport_state,
        p_rasterization_state:   &rasterizer,
        p_multisample_state:     &multisampling,
        p_depth_stencil_state:   &depth_stencil,
        p_color_blend_state:     &color_blending,
        p_dynamic_state:         &dynamic_state,
        layout,
        render_pass,
        subpass:                 0,
        base_pipeline_index:     -1,
        ..Default::default()
    };
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?[0]
    };

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    Ok((layout, pipeline))
}

fn create_shader_module(device: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    let spv_u32: Vec<u32> = spv
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let info = vk::ShaderModuleCreateInfo {
        code_size: spv_u32.len() * 4,
        p_code:    spv_u32.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_shader_module(&info, None).map_err(Into::into) }
}
