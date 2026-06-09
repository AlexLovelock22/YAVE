use std::collections::HashSet;

use glam::{IVec3, Mat4, Vec3};
use winit::keyboard::KeyCode;

const EYE_HEIGHT: f32 = 1.62;
const GRAVITY:    f32 = -25.0;
const MAX_FALL:   f32 = -40.0;
const JUMP_VEL:   f32 = 8.0;
const HALF_W:     f32 = 0.3;   // half-width of player AABB
const PLAYER_H:   f32 = 1.8;   // full height of player AABB
const WALK_ACCEL: f32 = 60.0;
const WALK_DECEL: f32 = 40.0;
const FLY_ACCEL:  f32 = 50.0;
const FLY_DECEL:  f32 = 35.0;

pub struct Camera {
    pub position:  Vec3,
    pub yaw:       f32,
    pub pitch:     f32,
    pub fov_y:     f32,
    pub aspect:    f32,
    pub speed:     f32,
    pub velocity:  Vec3,
    pub on_ground: bool,
    pub flying:    bool,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            position:  Vec3::new(2048.0, 200.0, 2048.0),
            yaw:       0.0,
            pitch:     -0.3,
            fov_y:     90_f32.to_radians(),
            aspect,
            speed:     20.0,
            velocity:  Vec3::ZERO,
            on_ground: false,
            flying:    true,
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
            r[2].into(),    // near
            q(r[3], r[2]),  // far
        ]
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }

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

    pub fn apply_movement(
        &mut self,
        keys:     &HashSet<KeyCode>,
        dt:       f32,
        is_solid: impl Fn(IVec3) -> bool,
    ) {
        if self.flying {
            self.apply_fly(keys, dt);
        } else {
            self.apply_walk(keys, dt, is_solid);
        }
    }

    fn apply_fly(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        let forward = self.forward();
        let right   = self.right();
        let mut wish = Vec3::ZERO;
        if keys.contains(&KeyCode::KeyW) { wish += forward; }
        if keys.contains(&KeyCode::KeyS) { wish -= forward; }
        if keys.contains(&KeyCode::KeyA) { wish += right; }
        if keys.contains(&KeyCode::KeyD) { wish -= right; }
        if keys.contains(&KeyCode::Space) { wish.y += 1.0; }
        if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            wish.y -= 1.0;
        }

        let target = if wish.length_squared() > 0.0 { wish.normalize() * self.speed } else { Vec3::ZERO };
        let accel  = if wish.length_squared() > 0.0 { FLY_ACCEL } else { FLY_DECEL };
        self.velocity.x = nudge(self.velocity.x, target.x, accel * dt);
        self.velocity.y = nudge(self.velocity.y, target.y, accel * dt);
        self.velocity.z = nudge(self.velocity.z, target.z, accel * dt);

        self.position += self.velocity * dt;
    }

    fn apply_walk(
        &mut self,
        keys:     &HashSet<KeyCode>,
        dt:       f32,
        is_solid: impl Fn(IVec3) -> bool,
    ) {
        // Desired horizontal velocity from WASD, flat (no pitch)
        let fwd   = Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos());
        let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let mut wish = Vec3::ZERO;
        if keys.contains(&KeyCode::KeyW) { wish += fwd; }
        if keys.contains(&KeyCode::KeyS) { wish -= fwd; }
        if keys.contains(&KeyCode::KeyA) { wish += right; }
        if keys.contains(&KeyCode::KeyD) { wish -= right; }

        let has_input = wish.length_squared() > 0.0;
        let target_h  = if has_input { wish.normalize() * self.speed } else { Vec3::ZERO };
        let accel_h   = if has_input { WALK_ACCEL } else { WALK_DECEL };
        self.velocity.x = nudge(self.velocity.x, target_h.x, accel_h * dt);
        self.velocity.z = nudge(self.velocity.z, target_h.z, accel_h * dt);

        // Gravity accumulates each frame and is cancelled by ground collision.
        self.velocity.y = (self.velocity.y + GRAVITY * dt).max(MAX_FALL);

        // Jump on space while grounded.
        if keys.contains(&KeyCode::Space) && self.on_ground {
            self.velocity.y  = JUMP_VEL;
            self.on_ground   = false;
        }

        // Axis-separated collision response
        let delta = self.velocity * dt;
        let mut feet = self.position - Vec3::new(0.0, EYE_HEIGHT, 0.0);
        self.on_ground = false;

        // ── X ───────────────────────────────────────────────────────────────────
        let nx = feet.x + delta.x;
        if !overlaps(&is_solid, nx, feet.y, feet.z) {
            feet.x = nx;
        } else {
            self.velocity.x = 0.0;
        }

        // ── Y ───────────────────────────────────────────────────────────────────
        let ny = feet.y + delta.y;
        if !overlaps(&is_solid, feet.x, ny, feet.z) {
            feet.y = ny;
        } else {
            if self.velocity.y < 0.0 {
                self.on_ground = true;
                // Snap feet to the exact top of the block we landed on so the player
                // doesn't float above the surface across multiple frames.
                feet.y = ny.ceil();
            }
            self.velocity.y = 0.0;
        }

        // ── Z ───────────────────────────────────────────────────────────────────
        let nz = feet.z + delta.z;
        if !overlaps(&is_solid, feet.x, feet.y, nz) {
            feet.z = nz;
        } else {
            self.velocity.z = 0.0;
        }

        self.position = feet + Vec3::new(0.0, EYE_HEIGHT, 0.0);
    }
}

/// Move `v` toward `target` by at most `max_delta`.
fn nudge(v: f32, target: f32, max_delta: f32) -> f32 {
    let diff = target - v;
    if diff.abs() <= max_delta { target } else { v + diff.signum() * max_delta }
}

/// Returns true if the player AABB (HALF_W × PLAYER_H × HALF_W box, feet at (fx, fy, fz))
/// overlaps any solid block.
fn overlaps(is_solid: &impl Fn(IVec3) -> bool, fx: f32, fy: f32, fz: f32) -> bool {
    let x0 = (fx - HALF_W).floor() as i32;
    let x1 = (fx + HALF_W - 0.001).floor() as i32;
    let y0 = fy.floor() as i32;
    let y1 = (fy + PLAYER_H - 0.001).floor() as i32;
    let z0 = (fz - HALF_W).floor() as i32;
    let z1 = (fz + HALF_W - 0.001).floor() as i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            for z in z0..=z1 {
                if is_solid(IVec3::new(x, y, z)) { return true; }
            }
        }
    }
    false
}
