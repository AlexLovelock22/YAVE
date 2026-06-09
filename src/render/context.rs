use anyhow::{anyhow, Result};
use ash::{vk, Device, Entry, Instance};
use std::ffi::CString;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_family: u32,
    pub present_family: u32,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
}

impl VulkanContext {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let raw_display = window.display_handle()?.as_raw();
        let raw_window = window.window_handle()?.as_raw();

        let app_name = CString::new("YAVE").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: app_name.as_ptr(),
            api_version: vk::API_VERSION_1_2,
            ..Default::default()
        };

        // Use ash-window to get platform-specific required extensions
        let mut instance_extensions =
            ash_window::enumerate_required_extensions(raw_display)?.to_vec();
        #[cfg(debug_assertions)]
        instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let layers: Vec<*const i8> = if cfg!(debug_assertions) {
            vec![validation_layer.as_ptr()]
        } else {
            vec![]
        };

        let instance_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: instance_extensions.len() as u32,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, raw_display, raw_window, None)?
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device =
            pick_physical_device(&instance, &surface_loader, surface, &physical_devices)
                .ok_or_else(|| anyhow!("No suitable Vulkan physical device found"))?;

        let (graphics_family, present_family) =
            find_queue_families(&instance, &surface_loader, surface, physical_device)
                .ok_or_else(|| anyhow!("Could not find required queue families"))?;

        let queue_priorities = [1.0_f32];
        let unique_families: Vec<u32> = if graphics_family == present_family {
            vec![graphics_family]
        } else {
            vec![graphics_family, present_family]
        };
        let queue_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&family| vk::DeviceQueueCreateInfo {
                queue_family_index: family,
                queue_count: 1,
                p_queue_priorities: queue_priorities.as_ptr(),
                ..Default::default()
            })
            .collect();

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let phys_features = unsafe { instance.get_physical_device_features(physical_device) };
        let enabled_features = vk::PhysicalDeviceFeatures {
            wide_lines: phys_features.wide_lines,
            ..Default::default()
        };
        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_infos.len() as u32,
            p_queue_create_infos: queue_infos.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features: &enabled_features,
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family, 0) };

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            graphics_family,
            present_family,
            surface_loader,
            surface,
        })
    }

    pub fn find_memory_type(
        &self,
        type_filter: u32,
        props: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
            {
                return Ok(i);
            }
        }
        Err(anyhow!("No suitable memory type found"))
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_physical_device(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    devices: &[vk::PhysicalDevice],
) -> Option<vk::PhysicalDevice> {
    let mut fallback = None;
    for &dev in devices {
        let props = unsafe { instance.get_physical_device_properties(dev) };
        if find_queue_families(instance, surface_loader, surface, dev).is_none() {
            continue;
        }
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            return Some(dev);
        }
        fallback = Some(dev);
    }
    fallback
}

fn find_queue_families(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    device: vk::PhysicalDevice,
) -> Option<(u32, u32)> {
    let families =
        unsafe { instance.get_physical_device_queue_family_properties(device) };
    let mut graphics = None;
    let mut present = None;
    for (i, family) in families.iter().enumerate() {
        let i = i as u32;
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics = Some(i);
        }
        let present_support = unsafe {
            surface_loader
                .get_physical_device_surface_support(device, i, surface)
                .unwrap_or(false)
        };
        if present_support {
            present = Some(i);
        }
        if graphics.is_some() && present.is_some() {
            break;
        }
    }
    Some((graphics?, present?))
}
