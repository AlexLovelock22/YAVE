use anyhow::Result;
use ash::{vk, Device};

use super::context::VulkanContext;

pub struct Swapchain {
    pub loader:      ash::khr::swapchain::Device,
    pub swapchain:   vk::SwapchainKHR,
    pub images:      Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub extent:      vk::Extent2D,
    pub format:      vk::Format,

    // ── Depth buffer (D32_SFLOAT_S8_UINT) ────────────────────────────────────
    pub depth_image:  vk::Image,
    pub depth_memory: vk::DeviceMemory,
    pub depth_view:   vk::ImageView,
}

impl Swapchain {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32) -> Result<Self> {
        build(ctx, width, height, vk::SwapchainKHR::null())
    }

    pub fn recreate(&mut self, ctx: &VulkanContext, width: u32, height: u32) -> Result<()> {
        unsafe { ctx.device.device_wait_idle()? };
        let old = self.swapchain;
        destroy_images(ctx, self);
        let new = build(ctx, width, height, old)?;
        unsafe { self.loader.destroy_swapchain(old, None) };
        *self = new;
        Ok(())
    }
}

pub fn destroy(ctx: &VulkanContext, sc: &Swapchain) {
    destroy_images(ctx, sc);
    unsafe { sc.loader.destroy_swapchain(sc.swapchain, None) };
}

fn destroy_images(ctx: &VulkanContext, sc: &Swapchain) {
    unsafe {
        let d = &ctx.device;
        for &iv in &sc.image_views {
            d.destroy_image_view(iv, None);
        }
        d.destroy_image_view(sc.depth_view, None);
        d.destroy_image(sc.depth_image, None);
        d.free_memory(sc.depth_memory, None);
    }
}

fn build(
    ctx:           &VulkanContext,
    width:         u32,
    height:        u32,
    old_swapchain: vk::SwapchainKHR,
) -> Result<Swapchain> {
    let surface_caps = unsafe {
        ctx.surface_loader
            .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)?
    };
    let formats = unsafe {
        ctx.surface_loader
            .get_physical_device_surface_formats(ctx.physical_device, ctx.surface)?
    };
    let present_modes = unsafe {
        ctx.surface_loader
            .get_physical_device_surface_present_modes(ctx.physical_device, ctx.surface)?
    };

    let fmt          = choose_format(&formats);
    let present_mode = choose_present_mode(&present_modes);
    let extent       = choose_extent(&surface_caps, width, height);

    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let (sharing_mode, queue_families) = if ctx.graphics_family != ctx.present_family {
        (vk::SharingMode::CONCURRENT, vec![ctx.graphics_family, ctx.present_family])
    } else {
        (vk::SharingMode::EXCLUSIVE, vec![])
    };

    let sc_info = vk::SwapchainCreateInfoKHR {
        surface:                  ctx.surface,
        min_image_count:          image_count,
        image_format:             fmt.format,
        image_color_space:        fmt.color_space,
        image_extent:             extent,
        image_array_layers:       1,
        image_usage:              vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode:       sharing_mode,
        queue_family_index_count: queue_families.len() as u32,
        p_queue_family_indices:   queue_families.as_ptr(),
        pre_transform:            surface_caps.current_transform,
        composite_alpha:          vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped:                  vk::TRUE,
        old_swapchain,
        ..Default::default()
    };

    let loader    = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
    let swapchain = unsafe { loader.create_swapchain(&sc_info, None)? };
    let images    = unsafe { loader.get_swapchain_images(swapchain)? };

    let image_views = images.iter()
        .map(|&img| make_view(&ctx.device, img, fmt.format, vk::ImageAspectFlags::COLOR))
        .collect::<Result<Vec<_>>>()?;

    let (depth_image, depth_memory, depth_view) = create_depth_image(ctx, extent)?;

    Ok(Swapchain {
        loader,
        swapchain,
        images,
        image_views,
        extent,
        format: fmt.format,
        depth_image,
        depth_memory,
        depth_view,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Image helpers
// ─────────────────────────────────────────────────────────────────────────────

fn create_depth_image(
    ctx:    &VulkanContext,
    extent: vk::Extent2D,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let format = vk::Format::D32_SFLOAT_S8_UINT;
    let image = unsafe {
        ctx.device.create_image(&vk::ImageCreateInfo {
            image_type:  vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D { width: extent.width, height: extent.height, depth: 1 },
            mip_levels:   1,
            array_layers: 1,
            samples:      vk::SampleCountFlags::TYPE_1,
            tiling:       vk::ImageTiling::OPTIMAL,
            usage:        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        }, None)?
    };
    let mem_reqs = unsafe { ctx.device.get_image_memory_requirements(image) };
    let mem_type = ctx.find_memory_type(
        mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let memory = unsafe {
        ctx.device.allocate_memory(&vk::MemoryAllocateInfo {
            allocation_size:   mem_reqs.size,
            memory_type_index: mem_type,
            ..Default::default()
        }, None)?
    };
    unsafe { ctx.device.bind_image_memory(image, memory, 0)? };

    let view = make_view(&ctx.device, image, format,
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)?;

    Ok((image, memory, view))
}

fn make_view(
    device: &Device,
    image:  vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo {
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask:      aspect,
            base_mip_level:   0,
            level_count:      1,
            base_array_layer: 0,
            layer_count:      1,
        },
        ..Default::default()
    };
    unsafe { device.create_image_view(&info, None).map_err(Into::into) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format / present-mode selection
// ─────────────────────────────────────────────────────────────────────────────

pub fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::Format {
    choose_format(formats).format
}

fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats.iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .copied()
        .unwrap_or(formats[0])
}

fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        vk::PresentModeKHR::IMMEDIATE
    } else if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

fn choose_extent(caps: &vk::SurfaceCapabilitiesKHR, width: u32, height: u32) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width:  width.clamp(caps.min_image_extent.width,  caps.max_image_extent.width),
            height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    }
}
