mod app;
mod camera;
mod meshing;
mod models;
mod render;
mod settings;
mod world;

use std::sync::Arc;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use app::App;

#[derive(Default)]
struct AppRunner {
    app: Option<App>,
}

impl ApplicationHandler for AppRunner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("YAVE")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("Failed to create window: {e:?}");
                event_loop.exit();
                return;
            }
        };
        match App::new(Arc::clone(&window)) {
            Ok(app) => self.app = Some(app),
            Err(e) => {
                eprintln!("Failed to initialise app: {e:?}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(app) = &mut self.app else { return };
        if !app.on_window_event(&event) {
            event_loop.exit();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(app) = &mut self.app {
            app.on_device_event(&event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(app) = &self.app {
            app.window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    // Leave half the cores free for the render thread so chunk generation doesn't starve it
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(cores.saturating_sub(2).max(1))
        .build_global();

    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut AppRunner::default())?;
    Ok(())
}
