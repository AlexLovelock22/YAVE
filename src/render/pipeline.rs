use anyhow::Result;
use ash::vk;
use std::mem;

use super::context::VulkanContext;
use super::mesh::Vertex;

/// All render passes and graphics pipelines used by the renderer.
///
/// Four render passes (recorded in this order every frame):
///   1. geom_render_pass  – scene geometry → offscreen R8G8B8A8 + D32_SFLOAT_S8_UINT
///   2. ssao_render_pass  – SSAO post-process → offscreen R8_UNORM raw AO
///   3. blur_render_pass  – bilateral blur of raw AO → R8_UNORM blurred AO
///   4. composite_render_pass – composite scene × blurred AO → swapchain image
pub struct Pipeline {
    // ── render passes ───────────────────────────────────────────────────
    pub geom_render_pass:      vk::RenderPass,
    pub ssao_render_pass:      vk::RenderPass,
    pub blur_render_pass:      vk::RenderPass,
    pub composite_render_pass: vk::RenderPass,

    // ── geometry (terrain + water) ───────────────────────────────────────
    pub layout:        vk::PipelineLayout,
    pub pipeline:      vk::Pipeline,
    pub water_pipeline: vk::Pipeline,

    // ── SSAO ─────────────────────────────────────────────────────────────
    pub ssao_desc_layout: vk::DescriptorSetLayout,
    pub ssao_layout:      vk::PipelineLayout,
    pub ssao_pipeline:    vk::Pipeline,

    // ── bilateral blur ───────────────────────────────────────────────────
    pub blur_desc_layout: vk::DescriptorSetLayout,
    pub blur_layout:      vk::PipelineLayout,
    pub blur_pipeline:    vk::Pipeline,

    // ── composite ────────────────────────────────────────────────────────
    pub composite_desc_layout: vk::DescriptorSetLayout,
    pub composite_layout:      vk::PipelineLayout,
    pub composite_pipeline:    vk::Pipeline,
}

impl Pipeline {
    pub fn new(
        ctx:          &VulkanContext,
        swapchain_fmt: vk::Format,
        tex_desc_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        let dev = &ctx.device;

        // ── render passes ─────────────────────────────────────────────
        let geom_render_pass      = create_geom_render_pass(dev)?;
        let ssao_render_pass      = create_ssao_render_pass(dev)?;
        let composite_render_pass = create_composite_render_pass(dev, swapchain_fmt)?;

        // ── geometry pipelines ────────────────────────────────────────
        let (layout, pipeline) = create_geom_pipeline(dev, geom_render_pass, tex_desc_layout, true)?;
        let water_pipeline     = create_water_pipeline(dev, geom_render_pass, layout)?;

        // ── SSAO ──────────────────────────────────────────────────────
        let ssao_desc_layout = create_ssao_desc_layout(dev)?;
        let ssao_layout      = create_ssao_pipeline_layout(dev, ssao_desc_layout)?;
        let ssao_pipeline    = create_fullscreen_pipeline(
            dev, ssao_render_pass, ssao_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/ssao.frag.spv")),
            vk::Format::R8_UNORM,
        )?;

        // ── blur ──────────────────────────────────────────────────────
        let blur_render_pass  = create_blur_render_pass(dev)?;
        let blur_desc_layout  = create_blur_desc_layout(dev)?;
        let blur_layout       = create_simple_pipeline_layout(dev, blur_desc_layout)?;
        let blur_pipeline     = create_fullscreen_pipeline(
            dev, blur_render_pass, blur_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/blur.frag.spv")),
            vk::Format::R8_UNORM,
        )?;

        // ── composite ─────────────────────────────────────────────────
        let composite_desc_layout = create_composite_desc_layout(dev)?;
        let composite_layout      = create_simple_pipeline_layout(dev, composite_desc_layout)?;
        let composite_pipeline    = create_fullscreen_pipeline(
            dev, composite_render_pass, composite_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/composite.frag.spv")),
            swapchain_fmt,
        )?;

        Ok(Self {
            geom_render_pass,
            ssao_render_pass,
            blur_render_pass,
            composite_render_pass,
            layout,
            pipeline,
            water_pipeline,
            ssao_desc_layout,
            ssao_layout,
            ssao_pipeline,
            blur_desc_layout,
            blur_layout,
            blur_pipeline,
            composite_desc_layout,
            composite_layout,
            composite_pipeline,
        })
    }
}

pub fn destroy(ctx: &VulkanContext, p: &Pipeline) {
    unsafe {
        let d = &ctx.device;
        d.destroy_pipeline(p.water_pipeline,      None);
        d.destroy_pipeline(p.pipeline,            None);
        d.destroy_pipeline(p.ssao_pipeline,       None);
        d.destroy_pipeline(p.blur_pipeline,       None);
        d.destroy_pipeline(p.composite_pipeline,  None);
        d.destroy_pipeline_layout(p.layout,            None);
        d.destroy_pipeline_layout(p.ssao_layout,       None);
        d.destroy_pipeline_layout(p.blur_layout,       None);
        d.destroy_pipeline_layout(p.composite_layout,  None);
        d.destroy_descriptor_set_layout(p.ssao_desc_layout,       None);
        d.destroy_descriptor_set_layout(p.blur_desc_layout,       None);
        d.destroy_descriptor_set_layout(p.composite_desc_layout,  None);
        d.destroy_render_pass(p.geom_render_pass,      None);
        d.destroy_render_pass(p.ssao_render_pass,      None);
        d.destroy_render_pass(p.blur_render_pass,      None);
        d.destroy_render_pass(p.composite_render_pass, None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render passes
// ─────────────────────────────────────────────────────────────────────────────

/// Geometry pass: renders terrain into an offscreen R8G8B8A8 + D32S8 framebuffer.
/// Both attachments transition to shader-read layouts at the end so the SSAO
/// and composite passes can sample them without explicit barriers.
fn create_geom_render_pass(device: &ash::Device) -> Result<vk::RenderPass> {
    let color_att = vk::AttachmentDescription {
        format:           vk::Format::R8G8B8A8_UNORM,
        samples:          vk::SampleCountFlags::TYPE_1,
        load_op:          vk::AttachmentLoadOp::CLEAR,
        store_op:         vk::AttachmentStoreOp::STORE,
        stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:   vk::ImageLayout::UNDEFINED,
        final_layout:     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ..Default::default()
    };
    let depth_att = vk::AttachmentDescription {
        format:           vk::Format::D32_SFLOAT_S8_UINT,
        samples:          vk::SampleCountFlags::TYPE_1,
        load_op:          vk::AttachmentLoadOp::CLEAR,
        store_op:         vk::AttachmentStoreOp::STORE,
        stencil_load_op:  vk::AttachmentLoadOp::CLEAR,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:   vk::ImageLayout::UNDEFINED,
        // DEPTH_STENCIL_READ_ONLY_OPTIMAL: can be both depth/stencil attachment (read-only)
        // and sampled image — exactly what the SSAO pass needs.
        final_layout:     vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
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
        pipeline_bind_point:       vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count:    color_ref.len() as u32,
        p_color_attachments:       color_ref.as_ptr(),
        p_depth_stencil_attachment: &depth_ref,
        ..Default::default()
    };

    // Wait for prior frame's SSAO reads before clearing and writing depth/color.
    let dep_in = vk::SubpassDependency {
        src_subpass:    vk::SUBPASS_EXTERNAL,
        dst_subpass:    0,
        src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        src_access_mask: vk::AccessFlags::SHADER_READ,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ..Default::default()
    };
    // Ensure geometry writes are complete before SSAO fragment shader reads.
    let dep_out = vk::SubpassDependency {
        src_subpass:    0,
        dst_subpass:    vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };

    let atts  = [color_att, depth_att];
    let subs  = [subpass];
    let deps  = [dep_in, dep_out];
    let info  = vk::RenderPassCreateInfo {
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

/// SSAO pass: renders a fullscreen quad to the R8_UNORM AO texture.
fn create_ssao_render_pass(device: &ash::Device) -> Result<vk::RenderPass> {
    let ao_att = vk::AttachmentDescription {
        format:           vk::Format::R8_UNORM,
        samples:          vk::SampleCountFlags::TYPE_1,
        load_op:          vk::AttachmentLoadOp::DONT_CARE,
        store_op:         vk::AttachmentStoreOp::STORE,
        stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:   vk::ImageLayout::UNDEFINED,
        final_layout:     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ..Default::default()
    };

    let ao_ref = [vk::AttachmentReference {
        attachment: 0,
        layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let subpass = vk::SubpassDescription {
        pipeline_bind_point:    vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: ao_ref.len() as u32,
        p_color_attachments:    ao_ref.as_ptr(),
        ..Default::default()
    };

    // Wait for geometry pass depth writes before SSAO reads depth.
    let dep_in = vk::SubpassDependency {
        src_subpass:    vk::SUBPASS_EXTERNAL,
        dst_subpass:    0,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };
    // Ensure AO writes are done before composite reads.
    let dep_out = vk::SubpassDependency {
        src_subpass:    0,
        dst_subpass:    vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };

    let atts = [ao_att];
    let subs = [subpass];
    let deps = [dep_in, dep_out];
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

/// Blur pass: bilateral Gaussian blur of the raw AO → blurred AO R8_UNORM.
fn create_blur_render_pass(device: &ash::Device) -> Result<vk::RenderPass> {
    let ao_att = vk::AttachmentDescription {
        format:           vk::Format::R8_UNORM,
        samples:          vk::SampleCountFlags::TYPE_1,
        load_op:          vk::AttachmentLoadOp::DONT_CARE,
        store_op:         vk::AttachmentStoreOp::STORE,
        stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:   vk::ImageLayout::UNDEFINED,
        final_layout:     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ..Default::default()
    };
    let ao_ref = [vk::AttachmentReference {
        attachment: 0,
        layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpass = vk::SubpassDescription {
        pipeline_bind_point:    vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: ao_ref.len() as u32,
        p_color_attachments:    ao_ref.as_ptr(),
        ..Default::default()
    };
    // Wait for SSAO pass to finish writing raw AO before blur fragment shader reads it.
    let dep_in = vk::SubpassDependency {
        src_subpass:     vk::SUBPASS_EXTERNAL,
        dst_subpass:     0,
        src_stage_mask:  vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask:  vk::PipelineStageFlags::FRAGMENT_SHADER,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };
    // Ensure blurred AO writes are done before composite reads.
    let dep_out = vk::SubpassDependency {
        src_subpass:     0,
        dst_subpass:     vk::SUBPASS_EXTERNAL,
        src_stage_mask:  vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask:  vk::PipelineStageFlags::FRAGMENT_SHADER,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };
    let atts = [ao_att];
    let subs = [subpass];
    let deps = [dep_in, dep_out];
    let info = vk::RenderPassCreateInfo {
        attachment_count: atts.len() as u32, p_attachments:    atts.as_ptr(),
        subpass_count:    subs.len() as u32, p_subpasses:      subs.as_ptr(),
        dependency_count: deps.len() as u32, p_dependencies:   deps.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_render_pass(&info, None).map_err(Into::into) }
}

/// Composite pass: reads scene color + AO, writes final image to a swapchain framebuffer.
fn create_composite_render_pass(device: &ash::Device, color_format: vk::Format) -> Result<vk::RenderPass> {
    let color_att = vk::AttachmentDescription {
        format:           color_format,
        samples:          vk::SampleCountFlags::TYPE_1,
        load_op:          vk::AttachmentLoadOp::DONT_CARE,
        store_op:         vk::AttachmentStoreOp::STORE,
        stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:   vk::ImageLayout::UNDEFINED,
        final_layout:     vk::ImageLayout::PRESENT_SRC_KHR,
        ..Default::default()
    };

    let color_ref = [vk::AttachmentReference {
        attachment: 0,
        layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let subpass = vk::SubpassDescription {
        pipeline_bind_point:    vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: color_ref.len() as u32,
        p_color_attachments:    color_ref.as_ptr(),
        ..Default::default()
    };

    let dep_in = vk::SubpassDependency {
        src_subpass:    vk::SUBPASS_EXTERNAL,
        dst_subpass:    0,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };
    let dep_out = vk::SubpassDependency {
        src_subpass:    0,
        dst_subpass:    vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::empty(),
        ..Default::default()
    };

    let atts = [color_att];
    let subs = [subpass];
    let deps = [dep_in, dep_out];
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
// Descriptor set layouts
// ─────────────────────────────────────────────────────────────────────────────

fn create_ssao_desc_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    // binding 0: depth sampler
    let binding = vk::DescriptorSetLayoutBinding {
        binding:              0,
        descriptor_type:      vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count:     1,
        stage_flags:          vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: std::ptr::null(),
        ..Default::default()
    };
    let info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: 1,
        p_bindings:    &binding,
        ..Default::default()
    };
    unsafe { device.create_descriptor_set_layout(&info, None).map_err(Into::into) }
}

fn create_blur_desc_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    // binding 0: raw AO sampler, binding 1: depth sampler (for bilateral weighting)
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding:              0,
            descriptor_type:      vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count:     1,
            stage_flags:          vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding:              1,
            descriptor_type:      vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count:     1,
            stage_flags:          vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        },
    ];
    let info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: bindings.len() as u32,
        p_bindings:    bindings.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_descriptor_set_layout(&info, None).map_err(Into::into) }
}

fn create_composite_desc_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    // binding 0: scene color sampler, binding 1: AO sampler
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding:          0,
            descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags:      vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding:          1,
            descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags:      vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        },
    ];
    let info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: bindings.len() as u32,
        p_bindings:    bindings.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_descriptor_set_layout(&info, None).map_err(Into::into) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline layouts
// ─────────────────────────────────────────────────────────────────────────────

/// SSAO pipeline layout: one descriptor set (depth sampler) + 128-byte push constant (proj + inv_proj).
fn create_ssao_pipeline_layout(
    device: &ash::Device,
    desc_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset:      0,
        size:        128, // mat4 proj (64) + mat4 inv_proj (64)
    };
    let set_layouts = [desc_layout];
    let info = vk::PipelineLayoutCreateInfo {
        set_layout_count:          set_layouts.len() as u32,
        p_set_layouts:             set_layouts.as_ptr(),
        push_constant_range_count: 1,
        p_push_constant_ranges:    &push_range,
        ..Default::default()
    };
    unsafe { device.create_pipeline_layout(&info, None).map_err(Into::into) }
}

/// Simple pipeline layout for the composite pass: one descriptor set, no push constants.
fn create_simple_pipeline_layout(
    device: &ash::Device,
    desc_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let set_layouts = [desc_layout];
    let info = vk::PipelineLayoutCreateInfo {
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts:    set_layouts.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_pipeline_layout(&info, None).map_err(Into::into) }
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

/// Shared fullscreen-quad pipeline builder for SSAO and composite passes.
/// Vertex shader is always fullscreen.vert; caller supplies the fragment SPIR-V and color format.
fn create_fullscreen_pipeline(
    device:       &ash::Device,
    render_pass:  vk::RenderPass,
    layout:       vk::PipelineLayout,
    frag_spv:     &[u8],
    color_format: vk::Format,
) -> Result<vk::Pipeline> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));
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

    // No vertex buffers — vertex positions are generated in the vertex shader.
    let vertex_input   = vk::PipelineVertexInputStateCreateInfo::default();
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
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable:  vk::FALSE,
        depth_write_enable: vk::FALSE,
        ..Default::default()
    };
    // One color attachment, no blending — just overwrite.
    let blend_att = vk::PipelineColorBlendAttachmentState {
        color_write_mask: vk::ColorComponentFlags::RGBA,
        blend_enable:     vk::FALSE,
        ..Default::default()
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

    // Suppress unused-variable warning in the signature — format is matched by the render pass.
    let _ = color_format;

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
