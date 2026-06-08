use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use winit::{
    event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

use ash::vk;

use crate::{
    camera::Camera,
    render::{mesh::PushConstants, renderer::Renderer},
    settings,
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
    frame_index: u64,
    /// Per-frame durations (µs) accumulated for the current 1-second window.
    frame_times_us: Vec<u32>,
    fps_window_timer: Instant,
    draw_buf:       Vec<vk::DrawIndexedIndirectCommand>,
    draw_buf_water: Vec<vk::DrawIndexedIndirectCommand>,
    title_timer: Instant,
    /// End time of the per-frame burst window triggered by crossing a chunk boundary.
    chunk_burst_until: Instant,
    /// Throttle for the normal (non-burst) periodic log line (~2/sec).
    slow_log_timer: Instant,
}

impl App {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let renderer = Renderer::new(&window)?;
        let aspect = size.width as f32 / size.height as f32;
        let camera = Camera::new(aspect);

        let s = settings::load();
        let mut world = World::new(s.render_distance(), s.lod1_dist(), s.lod2_dist());
        world.update(camera.position, &renderer.ctx);

        Ok(Self {
            window,
            renderer,
            camera,
            world,
            keys_held: HashSet::new(),
            last_frame: Instant::now(),
            cursor_grabbed: false,
            frame_index: 0,
            frame_times_us: Vec::new(),
            fps_window_timer: Instant::now(),
            draw_buf:       Vec::new(),
            draw_buf_water: Vec::new(),
            title_timer: Instant::now(),
            chunk_burst_until: Instant::now(),
            slow_log_timer: Instant::now(),
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
        const MIN_FRAME: Duration = Duration::from_nanos(666_667); // 1500 FPS cap
        let elapsed = self.last_frame.elapsed();
        if elapsed < MIN_FRAME {
            std::thread::sleep(MIN_FRAME - elapsed);
        }

        let now = Instant::now();
        let frame_dur = now.duration_since(self.last_frame);
        let dt = frame_dur.as_secs_f32().min(0.1);
        let frame_us = frame_dur.as_micros() as u64;
        self.last_frame = now;

        self.frame_index += 1;
        self.frame_times_us.push(frame_us.min(u32::MAX as u64) as u32);

        if self.fps_window_timer.elapsed().as_secs_f64() >= 1.0 {
            self.fps_window_timer = Instant::now();
            let times = &mut self.frame_times_us;
            if !times.is_empty() {
                let count = times.len();
                let sum: u64 = times.iter().map(|&t| t as u64).sum();
                let avg_us = sum / count as u64;
                let min_us = *times.iter().min().unwrap() as u64;
                let max_us = *times.iter().max().unwrap() as u64;
                times.sort_unstable();
                let p1_us  = times[count * 1  / 100] as u64;
                let p99_us = times[count * 99 / 100] as u64;
                println!(
                    "── 1s  frames={}  avg={:.0}fps({}µs)  best={:.0}fps  worst={:.0}fps  p1={}µs  p99={}µs ──",
                    count,
                    1_000_000.0 / avg_us.max(1) as f64, avg_us,
                    1_000_000.0 / min_us.max(1) as f64,
                    1_000_000.0 / max_us.max(1) as f64,
                    p1_us, p99_us,
                );
                times.clear();
            }
        }

        self.camera.apply_movement(&self.keys_held, dt);

        let t_update = Instant::now();
        self.world.update(self.camera.position, &self.renderer.ctx);
        let update_us = t_update.elapsed().as_micros() as u64;

        // Frustum cull while the GPU may still be finishing the previous-previous frame —
        // this CPU work overlaps with GPU work for free.
        let t_cull = Instant::now();
        let planes = self.camera.frustum_planes();
        self.world.cull_draws(&planes, self.camera.position, &mut self.draw_buf, &mut self.draw_buf_water);

        // begin_frame waits for the in-flight fence BEFORE we write the indirect buffer.
        // Without this ordering the CPU can overwrite an indirect buffer slot that the GPU
        // is still reading, causing random chunks to flicker every other frame.
        let t_gpu = Instant::now();
        let image_index = match self.renderer.begin_frame()? {
            Some(idx) => idx,
            None => return Ok(()),
        };

        self.world.flush_indirect(&self.draw_buf, &self.renderer.ctx);
        self.world.flush_indirect_water(&self.draw_buf_water, &self.renderer.ctx);
        let cull_us = t_cull.elapsed().as_micros() as u64;

        if self.title_timer.elapsed().as_millis() >= 100 {
            self.title_timer = Instant::now();
            let fps = 1_000_000.0 / frame_us.max(1) as f64;
            // index_count * 2/3 = vertex count (every quad is 4 verts / 6 indices).
            let verts: u32 = self.draw_buf.iter().map(|c| c.index_count * 2 / 3).sum();
            self.window.set_title(&format!("YAVE  |  {:.0} fps  |  {} verts  |  {} chunks", fps, verts, self.draw_buf.len()));
        }

        let push = PushConstants { mvp: self.camera.view_proj().to_cols_array_2d() };
        let r = self.renderer.end_frame(image_index, self.world.render_buffers(), self.world.indirect_draw(), self.world.indirect_draw_water(), push, self.camera.proj_matrix());
        let gpu_us = t_gpu.elapsed().as_micros() as u64;

        let total_us = update_us + cull_us + gpu_us;
        let d = &self.world;

        // Extend burst window whenever the player crosses a chunk boundary.
        if d.diag_chunk_changed {
            self.chunk_burst_until = now + Duration::from_secs(1);
        }

        // Log every frame during a chunk-crossing burst; otherwise throttle to ~2/sec.
        let in_burst = now < self.chunk_burst_until;
        let slow_log = self.slow_log_timer.elapsed() >= Duration::from_millis(500);
        if slow_log {
            self.slow_log_timer = now;
        }

        if in_burst || slow_log {
            let verts: u32 = self.draw_buf.iter().map(|c| c.index_count * 2 / 3).sum();
            // println!(
            //     "[f{:>6}] total={:>5}µs | update={:>4}µs \
            //      (poll={:>3}µs gen_rx={:>3}µs mesh_rx={:>3}µs upload={:>4}µs) \
            //      | cull={:>3}µs gpu={:>4}µs \
            //      | draws={:>3} water={:>3} verts={:>6} \
            //      | mesh_in={} batch={}chunks/{}KB",
            //     self.frame_index,
            //     total_us, update_us,
            //     d.diag_batch_poll_us, d.diag_gen_rx_us,
            //     d.diag_mesh_rx_us, d.diag_upload_us,
            //     cull_us, gpu_us,
            //     self.draw_buf.len(), self.draw_buf_water.len(), verts,
            //     d.diag_meshes_in, d.diag_batch_chunks, d.diag_batch_kb,
            // );
        }

        r
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe { let _ = self.renderer.ctx.device.device_wait_idle(); }
        self.world.destroy(&self.renderer.ctx);
    }
}
