use std::f32::consts::TAU;

use crate::world::noise::simplex2d;

pub const CELL_SIZE: i32 = 4_000;
pub const SEA_LEVEL: usize = 120;

// ── Tuning ────────────────────────────────────────────────────────────────────

const MIN_RADIUS: f32 = 2_000.0;
const MAX_RADIUS: f32 = 3_500.0;
const MIN_ASPECT: f32 = 0.8;
const MAX_ASPECT: f32 = 1.5;

// A continent's "value" (dome) is summed across the 3×3 cell neighbourhood.
// Land when sum >= CONT_THRESHOLD.
pub const CONT_THRESHOLD: f32 = 0.18;

// Large-scale domain warp (bulges, flat edges, the "African lean").
const WARP1_AMP_FACTOR: f32    = 0.35;
const WARP1_PERIOD_FACTOR: f32 = 1.5;

// Medium-scale domain warp (bays, capes, peninsulas).
const WARP2_AMP_FACTOR: f32    = 0.08;
const WARP2_PERIOD_FACTOR: f32 = 0.5;

// Coastline jaggedness: multi-octave noise added to the normalised distance.
const COAST_FREQ_FACTOR: f32 = 5.0;
const COAST_AMP: f32         = 0.14;
const COAST_OCT: [f32; 6]    = [0.50, 0.35, 0.20, 0.12, 0.07, 0.04];

// ── Data ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct ContinentDef {
    cx: f32, cz: f32,
    ra: f32, rb: f32,   // semi-axes in local frame
    cos_a: f32, sin_a: f32,
    warp1_freq: f32, warp1_amp: f32, warp1_phase: f32,
    warp2_freq: f32, warp2_amp: f32, warp2_phase: f32,
    coast_freq: f32, coast_phase: f32,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Build the 9 continent defs for the 3×3 cell neighbourhood of `(wx, wz)`.
/// Call once per chunk (all columns in a 32-block chunk share the same neighbourhood
/// because CELL_SIZE=4000 >> CHUNK_SIZE=32) and reuse across columns.
pub fn build_defs(wx: i32, wz: i32) -> [ContinentDef; 9] {
    let base_cx = wx.div_euclid(CELL_SIZE) - 1;
    let base_cz = wz.div_euclid(CELL_SIZE) - 1;
    std::array::from_fn(|i| {
        continent_for_cell(base_cx + (i / 3) as i32, base_cz + (i % 3) as i32)
    })
}

/// True if `(wx, wz)` is land, given pre-built neighbourhood defs.
#[inline]
pub fn is_land_defs(defs: &[ContinentDef; 9], wx: i32, wz: i32) -> bool {
    continentalness_defs(defs, wx as f32, wz as f32) <= 0.0
}

/// Fallback single-call version (allocates 9 defs internally — use build_defs for bulk).
pub fn is_land(wx: i32, wz: i32) -> bool {
    let defs = build_defs(wx, wz);
    is_land_defs(&defs, wx, wz)
}

// ── Internals ─────────────────────────────────────────────────────────────────

pub fn continentalness_defs(defs: &[ContinentDef; 9], fx: f32, fz: f32) -> f32 {
    let land: f32 = defs.iter().map(|c| continent_sample(c, fx, fz)).sum();
    CONT_THRESHOLD - land
}

pub fn continent_for_cell(cx: i32, cz: i32) -> ContinentDef {
    let base   = lerp(MIN_RADIUS, MAX_RADIUS, rng01(cx, cz, 0));
    let aspect = lerp(MIN_ASPECT, MAX_ASPECT,  rng01(cx, cz, 1));
    let angle  = rng01(cx, cz, 2) * TAU;

    let ra = base;
    let rb = base * aspect;

    let max_off = CELL_SIZE as f32 / 8.0;
    let off_x = (rng01(cx, cz, 6) - 0.5) * 2.0 * max_off;
    let off_z = (rng01(cx, cz, 7) - 0.5) * 2.0 * max_off;

    ContinentDef {
        cx: (cx as f32 + 0.5) * CELL_SIZE as f32 + off_x,
        cz: (cz as f32 + 0.5) * CELL_SIZE as f32 + off_z,
        ra,
        rb,
        cos_a: angle.cos(),
        sin_a: angle.sin(),
        warp1_freq:  1.0 / (base * WARP1_PERIOD_FACTOR),
        warp1_amp:   base * WARP1_AMP_FACTOR,
        warp1_phase: rng01(cx, cz, 3) * 100.0,
        warp2_freq:  1.0 / (base * WARP2_PERIOD_FACTOR),
        warp2_amp:   base * WARP2_AMP_FACTOR,
        warp2_phase: rng01(cx, cz, 4) * 100.0,
        coast_freq:  COAST_FREQ_FACTOR / base,
        coast_phase: rng01(cx, cz, 5) * 100.0,
    }
}

#[inline]
fn continent_sample(c: &ContinentDef, fx: f32, fz: f32) -> f32 {
    let dx = fx - c.cx;
    let dz = fz - c.cz;

    // Rotate to ellipse-local frame
    let lx =  dx * c.cos_a + dz * c.sin_a;
    let lz = -dx * c.sin_a + dz * c.cos_a;

    // Large-scale domain warp
    let w1x = simplex2d((fx + c.warp1_phase) * c.warp1_freq,         fz * c.warp1_freq        ) * c.warp1_amp;
    let w1z = simplex2d((fx + c.warp1_phase) * c.warp1_freq + 3.7,   fz * c.warp1_freq + 1.9  ) * c.warp1_amp;

    // Medium-scale domain warp
    let w2x = simplex2d((fx + c.warp2_phase) * c.warp2_freq,         fz * c.warp2_freq        ) * c.warp2_amp;
    let w2z = simplex2d((fx + c.warp2_phase) * c.warp2_freq + 7.3,   fz * c.warp2_freq + 5.1  ) * c.warp2_amp;

    // Transform the warp back into local frame and apply
    let wx_total = w1x + w2x;
    let wz_total = w1z + w2z;
    let wlx = lx + wx_total * c.cos_a + wz_total * c.sin_a;
    let wlz = lz - wx_total * c.sin_a + wz_total * c.cos_a;

    // Normalised ellipse distance
    let ex = wlx / c.ra;
    let ez = wlz / c.rb;
    let t = (ex * ex + ez * ez).sqrt();

    // Coastline noise (6 octaves)
    let cf = c.coast_freq;
    let px = fx + c.coast_phase;
    let coast =
        simplex2d(px * cf,        fz * cf       ) * COAST_OCT[0] +
        simplex2d(px * cf * 2.0,  fz * cf * 2.0 ) * COAST_OCT[1] +
        simplex2d(px * cf * 4.0,  fz * cf * 4.0 ) * COAST_OCT[2] +
        simplex2d(px * cf * 8.0,  fz * cf * 8.0 ) * COAST_OCT[3] +
        simplex2d(px * cf * 16.0, fz * cf * 16.0) * COAST_OCT[4] +
        simplex2d(px * cf * 32.0, fz * cf * 32.0) * COAST_OCT[5];

    let t = (t + coast * COAST_AMP).clamp(0.0, 1.0);

    // Dome: (1 - smoothstep(0,1,t))³
    let s = t * t * (3.0 - 2.0 * t); // smoothstep
    let d = 1.0 - s;
    d * d * d
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t * (b - a) }

/// Splitmix64-style hash → f32 in [0, 1).
fn rng01(cx: i32, cz: i32, stream: u64) -> f32 {
    let mut h: u64 = (cx as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (cz as u64).wrapping_mul(0x6c62272e07bb0142)
        ^ stream.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 30; h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27; h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    (h >> 11) as f32 / (1u64 << 53) as f32
}
