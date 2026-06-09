use std::collections::HashSet;

use glam::{Mat4, Vec3};
use winit::keyboard::KeyCode;

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub speed: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::new(2048.0, 200.0, 2048.0),
            yaw: 0.0,
            pitch: -0.3,
            fov_y: 90_f32.to_radians(),
            aspect,
            speed: 20.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }

    fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    /// Extracts 6 world-space frustum planes from the VP matrix (Gribb-Hartmann method).
    /// Each plane is [a, b, c, d] where ax+by+cz+d >= 0 for points inside the frustum.
    pub fn frustum_planes(&self) -> [[f32; 4]; 6] {
        let m = self.view_proj();
        let r = [m.row(0), m.row(1), m.row(2), m.row(3)];
        let p = |a: glam::Vec4, b: glam::Vec4| -> [f32; 4] { (a + b).into() };
        let q = |a: glam::Vec4, b: glam::Vec4| -> [f32; 4] { (a - b).into() };
        [
            p(r[3], r[0]),  // left
            q(r[3], r[0]),  // right
            p(r[3], r[1]),  // bottom
            q(r[3], r[1]),  // top
            r[2].into(),    // near  (Vulkan Z: 0..1)
            q(r[3], r[2]),  // far
        ]
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }

    /// Projection matrix only (includes Vulkan Y-flip). Used by the SSAO pass.
    pub fn proj_matrix(&self) -> Mat4 {
        let proj   = Mat4::perspective_rh(self.fov_y, self.aspect, 0.1, 10000.0);
        let flip_y = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        flip_y * proj
    }

    pub fn look(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * 0.002;
        self.pitch = (self.pitch - delta_y * 0.005)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.001, std::f32::consts::FRAC_PI_2 - 0.001);
    }

    pub fn adjust_speed(&mut self, delta: f32) {
        self.speed = (self.speed + delta * 2.0).clamp(1.0, 200.0);
    }

    pub fn apply_movement(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        // Flatten forward onto the XZ plane so W/S never changes height
        let flat_forward = Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos());
        let right = self.right();
        let spd = self.speed * dt;

        if keys.contains(&KeyCode::KeyW) { self.position += flat_forward * spd; }
        if keys.contains(&KeyCode::KeyS) { self.position -= flat_forward * spd; }
        if keys.contains(&KeyCode::KeyA) { self.position += right * spd; }
        if keys.contains(&KeyCode::KeyD) { self.position -= right * spd; }
        if keys.contains(&KeyCode::Space) { self.position.y += spd; }
        if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            self.position.y -= spd;
        }
    }
}
