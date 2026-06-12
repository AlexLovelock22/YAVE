#![allow(dead_code)]

// Cell-based rock hardness system.
// Currently unused for cliff generation — replaced by gradient-projected 1D noise in terrain.rs.
// Kept here for future non-cliff uses: mineral veins, biome variation, beach type, etc.
//
// Hard cells are snapped to the nearest coastline (c = 0) at build time.
// This guarantees the peak of every cliff dome sits exactly where land meets
// ocean, so the cliff boost always fires with maximum strength.
//
// Cells whose initial RNG center is further than SNAP_RANGE_C from the coast
// are suppressed: deep-inland cells are reserved for future non-cliff uses
// (mineral veins, biomes, etc.) and deep-ocean cells have no land to cliff.
//
// The snap walk uses numerical gradient descent on the continent field, taking
// up to SNAP_STEPS × SNAP_STEP_SIZE = 1 200 blocks to reach c ≈ 0.

use crate::world::continents::{continentalness_defs, ContinentDef};

pub const HARDNESS_CELL_SIZE: i32 = 400;

const MIN_RADIUS: f32 = 550.0;
const MAX_RADIUS: f32 = 850.0;

const HARD_FRACTION: f32 = 0.55;
const JITTER:        f32 = 0.4;

// Only cells whose initial center has |c| < SNAP_RANGE_C are cliff-eligible.
// At the continent gradient rate this is roughly ±200–400 blocks from the shore.
const SNAP_RANGE_C:   f32 = 0.12;
// Gradient-descent tuning for the coast walk.
const SNAP_STEP_SIZE: f32 = 40.0;
const SNAP_GRAD_D:    f32 = 25.0;
const SNAP_STEPS:     usize = 30;

#[derive(Clone, Copy)]
pub struct HardnessDef {
    pub cx:   f32,
    pub cz:   f32,
    radius:   f32,
    hard:     bool,
}

/// Build the 3×3 cell neighbourhood for the chunk containing `(wx, wz)`.
/// `cont_defs` is used to snap hard-cell centres onto the nearest coastline.
pub fn build_hardness_defs(
    wx:        i32,
    wz:        i32,
    cont_defs: &[ContinentDef; 9],
) -> [HardnessDef; 9] {
    let base_cx = wx.div_euclid(HARDNESS_CELL_SIZE) - 1;
    let base_cz = wz.div_euclid(HARDNESS_CELL_SIZE) - 1;
    std::array::from_fn(|i| {
        let mut def = hardness_for_cell(
            base_cx + (i / 3) as i32,
            base_cz + (i % 3) as i32,
        );
        if def.hard {
            let c0 = continentalness_defs(cont_defs, def.cx, def.cz);
            if c0.abs() <= SNAP_RANGE_C {
                // Near-coastal: walk to c = 0 so the dome peaks on the shore.
                let (sx, sz) = snap_to_coast(def.cx, def.cz, cont_defs);
                def.cx = sx;
                def.cz = sz;
            } else {
                // Too far inland or too deep ocean — no cliff contribution.
                def.hard = false;
            }
        }
        def
    })
}

/// Returns hardness in [0, 1]: 1.0 = fully hard rock, 0.0 = fully soft.
pub fn hardness_at(defs: &[HardnessDef; 9], fx: f32, fz: f32) -> f32 {
    defs.iter()
        .filter(|d| d.hard)
        .map(|d| {
            let dx = fx - d.cx;
            let dz = fz - d.cz;
            let t = ((dx * dx + dz * dz).sqrt() / d.radius).clamp(0.0, 1.0);
            let s = t * t * (3.0 - 2.0 * t);
            1.0 - s
        })
        .sum::<f32>()
        .clamp(0.0, 1.0)
}

// ── Internals ─────────────────────────────────────────────────────────────────

fn hardness_for_cell(cx: i32, cz: i32) -> HardnessDef {
    let half  = HARDNESS_CELL_SIZE as f32 * 0.5;
    let off_x = (rng01(cx, cz, 0) - 0.5) * 2.0 * half * JITTER;
    let off_z = (rng01(cx, cz, 1) - 0.5) * 2.0 * half * JITTER;
    HardnessDef {
        cx:     (cx as f32 + 0.5) * HARDNESS_CELL_SIZE as f32 + off_x,
        cz:     (cz as f32 + 0.5) * HARDNESS_CELL_SIZE as f32 + off_z,
        radius: lerp(MIN_RADIUS, MAX_RADIUS, rng01(cx, cz, 2)),
        hard:   rng01(cx, cz, 3) < HARD_FRACTION,
    }
}

/// Walk `(px, pz)` toward continentalness = 0 using gradient descent.
/// Stops when |c| < 0.02 or after SNAP_STEPS iterations.
fn snap_to_coast(
    mut px: f32,
    mut pz: f32,
    cont_defs: &[ContinentDef; 9],
) -> (f32, f32) {
    for _ in 0..SNAP_STEPS {
        let c = continentalness_defs(cont_defs, px, pz);
        if c.abs() < 0.02 {
            break;
        }
        // Numerical gradient of the continent field.
        let d  = SNAP_GRAD_D;
        let gx = continentalness_defs(cont_defs, px + d, pz)
               - continentalness_defs(cont_defs, px - d, pz);
        let gz = continentalness_defs(cont_defs, px, pz + d)
               - continentalness_defs(cont_defs, px, pz - d);
        let len = (gx * gx + gz * gz).sqrt().max(1e-6);
        // Land (c < 0): walk in +gradient direction (toward ocean = higher c).
        // Ocean (c > 0): walk in -gradient direction (toward land = lower c).
        let sign: f32 = if c < 0.0 { 1.0 } else { -1.0 };
        px += sign * (gx / len) * SNAP_STEP_SIZE;
        pz += sign * (gz / len) * SNAP_STEP_SIZE;
    }
    (px, pz)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t * (b - a) }

fn rng01(cx: i32, cz: i32, stream: u64) -> f32 {
    let mut h: u64 = (cx as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (cz as u64).wrapping_mul(0x6c62272e07bb0142)
        ^ stream.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 30; h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27; h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    (h >> 11) as f32 / (1u64 << 53) as f32
}
