use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use winit::{
    event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

use crate::{
    camera::Camera,
    render::{mesh::PushConstants, renderer::Renderer},
    world::world::World,
};

pub struct App {
    pub window: Arc<Window>,
    renderer: Renderer,
    camera: Camera,
    world: World,
    keys_held: HashSet<KeyCode>,
    last_frame: Instant,
    cursor_grabbed: bool,
    fps_frame_count: u32,
    fps_last_print: Instant,
}

impl App {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let renderer = Renderer::new(&window)?;
        let aspect = size.width as f32 / size.height as f32;
        let camera = Camera::new(aspect);

        let RD = 40;
        let mut world = World::new(RD);
        world.update(camera.position, &renderer.ctx, renderer.command_pool());

        Ok(Self {
            window,
            renderer,
            camera,
            world,
            keys_held: HashSet::new(),
            last_frame: Instant::now(),
            cursor_grabbed: false,
            fps_frame_count: 0,
            fps_last_print: Instant::now(),
        })
    }

    fn set_cursor_grab(&mut self, grab: bool) {
        self.cursor_grabbed = grab;
        if grab {
            let _ = self.window.set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        }
        self.window.set_cursor_visible(!grab);
    }

    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CloseRequested => return false,
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render() {
                    eprintln!("render error: {e:?}");
                    return false;
                }
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.camera.aspect = size.width as f32 / size.height as f32;
                    let _ = self.renderer.resize(size.width, size.height);
                }
            }
            // Any click grabs the cursor; Escape releases it
            WindowEvent::MouseInput { state: ElementState::Pressed, .. } => {
                self.set_cursor_grab(true);
            }
            WindowEvent::Focused(false) => {
                self.set_cursor_grab(false);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.camera.adjust_speed(scroll);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape && event.state == ElementState::Pressed {
                        self.set_cursor_grab(false);
                    }
                    match event.state {
                        ElementState::Pressed => { self.keys_held.insert(key); }
                        ElementState::Released => { self.keys_held.remove(&key); }
                    }
                }
            }
            _ => {}
        }
        true
    }

    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.cursor_grabbed {
                self.camera.look(*dx as f32, *dy as f32);
            }
        }
    }

    fn render(&mut self) -> Result<()> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.fps_frame_count += 1;
        let fps_elapsed = self.fps_last_print.elapsed();
        if fps_elapsed.as_millis() >= 200 {
            let fps = self.fps_frame_count as f64 / fps_elapsed.as_secs_f64();
            println!("FPS: {:.0}", fps);
            self.fps_frame_count = 0;
            self.fps_last_print = Instant::now();
        }

        self.camera.apply_movement(&self.keys_held, dt);
        self.world.update(self.camera.position, &self.renderer.ctx, self.renderer.command_pool());
        let meshes: Vec<&_> = self.world.iter_meshes().collect();
        let push = PushConstants { mvp: self.camera.view_proj().to_cols_array_2d() };
        self.renderer.draw_frame(&meshes, push)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe { let _ = self.renderer.ctx.device.device_wait_idle(); }
        self.world.destroy(&self.renderer.ctx);
    }
}
