use std::path::Path;

use anyhow::Result;
use ash::vk;

use super::{buffer::create_buffer, context::VulkanContext};

// Layer order must match face_tex() in world/block.rs
const LAYER_FILES: [(&str, [u8; 4]); 4] = [
    ("Stone.png",      [128, 128, 128, 255]),
    ("Dirt.png",       [101,  67,  33, 255]),
    ("Grass_Top.png",  [ 85, 139,  48, 255]),
    ("Grass_Side.png", [ 69, 108,  45, 255]),
];
const LAYER_COUNT: u32 = LAYER_FILES.len() as u32;

pub struct TextureArray {
    image:           vk::Image,
    memory:          vk::DeviceMemory,
    view:            vk::ImageView,
    sampler:         vk::Sampler,
    pub desc_layout: vk::DescriptorSetLayout,
    desc_pool:       vk::DescriptorPool,
    pub desc_set:    vk::DescriptorSet,
}

impl TextureArray {
    pub fn new(ctx: &VulkanContext, pool: vk::CommandPool) -> Result<Self> {
        let tex_dir = Path::new("Textures");
        std::fs::create_dir_all(tex_dir)?;

        // Generate solid-color placeholders for any missing files.
        for (filename, rgba) in &LAYER_FILES {
            let path = tex_dir.join(filename);
            if !path.exists() {
                let img = image::RgbaImage::from_pixel(24, 24, image::Rgba(*rgba));
                img.save(&path)?;
                println!("[texture] created placeholder {filename}");
            }
        }

        // Load all layers; all must be the same size.
        let layers: Vec<image::RgbaImage> = LAYER_FILES.iter()
            .map(|(f, _)| -> Result<_> {
                Ok(image::open(tex_dir.join(f))?.into_rgba8())
            })
            .collect::<Result<_>>()?;

        let width  = layers[0].width();
        let height = layers[0].height();
        for (i, l) in layers.iter().enumerate() {
            assert!(l.width() == width && l.height() == height,
                "Texture {} has different size than layer 0", LAYER_FILES[i].0);
        }

        let layer_bytes = (width * height * 4) as usize;
        let total_bytes = layer_bytes * LAYER_COUNT as usize;

        // Staging buffer — all layers packed sequentially.
        let (staging_buf, staging_mem) = create_buffer(
            ctx,
            total_bytes as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = ctx.device.map_memory(
                staging_mem, 0, total_bytes as vk::DeviceSize, vk::MemoryMapFlags::empty(),
            )? as *mut u8;
            for (i, layer) in layers.iter().enumerate() {
                std::ptr::copy_nonoverlapping(layer.as_raw().as_ptr(), ptr.add(i * layer_bytes), layer_bytes);
            }
            ctx.device.unmap_memory(staging_mem);
        }

        // Device-local 2D array image.
        let image = unsafe {
            ctx.device.create_image(&vk::ImageCreateInfo {
                image_type:     vk::ImageType::TYPE_2D,
                format:         vk::Format::R8G8B8A8_SRGB,
                extent:         vk::Extent3D { width, height, depth: 1 },
                mip_levels:     1,
                array_layers:   LAYER_COUNT,
                samples:        vk::SampleCountFlags::TYPE_1,
                tiling:         vk::ImageTiling::OPTIMAL,
                usage:          vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode:   vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            }, None)?
        };

        let reqs     = unsafe { ctx.device.get_image_memory_requirements(image) };
        let mem_type = ctx.find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let memory   = unsafe {
            let m = ctx.device.allocate_memory(&vk::MemoryAllocateInfo {
                allocation_size:   reqs.size,
                memory_type_index: mem_type,
                ..Default::default()
            }, None)?;
            ctx.device.bind_image_memory(image, m, 0)?;
            m
        };

        let full_range = vk::ImageSubresourceRange {
            aspect_mask:      vk::ImageAspectFlags::COLOR,
            base_mip_level:   0,
            level_count:      1,
            base_array_layer: 0,
            layer_count:      LAYER_COUNT,
        };

        // Upload: layout transitions + one buffer-to-image copy per layer.
        let cmd = unsafe {
            ctx.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool:        pool,
                level:               vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0]
        };
        unsafe {
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            })?;

            ctx.device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout:             vk::ImageLayout::UNDEFINED,
                    new_layout:             vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_access_mask:        vk::AccessFlags::empty(),
                    dst_access_mask:        vk::AccessFlags::TRANSFER_WRITE,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: full_range,
                    ..Default::default()
                }],
            );

            let copies: Vec<vk::BufferImageCopy> = (0..LAYER_COUNT).map(|layer| {
                vk::BufferImageCopy {
                    buffer_offset:       (layer as usize * layer_bytes) as vk::DeviceSize,
                    buffer_row_length:   0,
                    buffer_image_height: 0,
                    image_subresource:   vk::ImageSubresourceLayers {
                        aspect_mask:      vk::ImageAspectFlags::COLOR,
                        mip_level:        0,
                        base_array_layer: layer,
                        layer_count:      1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width, height, depth: 1 },
                }
            }).collect();

            ctx.device.cmd_copy_buffer_to_image(
                cmd, staging_buf, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copies,
            );

            ctx.device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout:             vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout:             vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access_mask:        vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask:        vk::AccessFlags::SHADER_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: full_range,
                    ..Default::default()
                }],
            );

            ctx.device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            ctx.device.queue_submit(ctx.graphics_queue, &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: cmds.as_ptr(),
                ..Default::default()
            }], vk::Fence::null())?;
            ctx.device.queue_wait_idle(ctx.graphics_queue)?;
            ctx.device.free_command_buffers(pool, &cmds);

            ctx.device.destroy_buffer(staging_buf, None);
            ctx.device.free_memory(staging_mem, None);
        }

        let view = unsafe {
            ctx.device.create_image_view(&vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                format:    vk::Format::R8G8B8A8_SRGB,
                components: vk::ComponentMapping::default(),
                subresource_range: full_range,
                ..Default::default()
            }, None)?
        };

        let sampler = unsafe {
            ctx.device.create_sampler(&vk::SamplerCreateInfo {
                mag_filter:     vk::Filter::NEAREST,
                min_filter:     vk::Filter::NEAREST,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                mip_lod_bias:   0.0,
                max_lod:        0.0,
                ..Default::default()
            }, None)?
        };

        // Descriptor set layout — one combined image sampler in fragment stage.
        let binding = vk::DescriptorSetLayoutBinding {
            binding:              0,
            descriptor_type:      vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count:     1,
            stage_flags:          vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        };
        let desc_layout = unsafe {
            ctx.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo {
                binding_count: 1,
                p_bindings:    &binding,
                ..Default::default()
            }, None)?
        };

        let pool_size = vk::DescriptorPoolSize {
            ty:               vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };
        let desc_pool = unsafe {
            ctx.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo {
                max_sets:        1,
                pool_size_count: 1,
                p_pool_sizes:    &pool_size,
                ..Default::default()
            }, None)?
        };

        let layouts  = [desc_layout];
        let desc_set = unsafe {
            ctx.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool:      desc_pool,
                descriptor_set_count: 1,
                p_set_layouts:        layouts.as_ptr(),
                ..Default::default()
            })?[0]
        };

        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view:   view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        unsafe {
            ctx.device.update_descriptor_sets(&[vk::WriteDescriptorSet {
                dst_set:           desc_set,
                dst_binding:       0,
                dst_array_element: 0,
                descriptor_count:  1,
                descriptor_type:   vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info:      &image_info,
                ..Default::default()
            }], &[]);
        }

        Ok(Self { image, memory, view, sampler, desc_layout, desc_pool, desc_set })
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_descriptor_pool(self.desc_pool, None);
            ctx.device.destroy_descriptor_set_layout(self.desc_layout, None);
            ctx.device.destroy_sampler(self.sampler, None);
            ctx.device.destroy_image_view(self.view, None);
            ctx.device.destroy_image(self.image, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}
