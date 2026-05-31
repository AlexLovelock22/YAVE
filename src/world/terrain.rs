use crate::world::{
    chunk::CHUNK_SIZE,
    continents::{continentalness_defs, ContinentDef, SEA_LEVEL},
    noise::simplex2d,
};

const HEIGHT_SCALE: f32 = 50.0;

const INLAND_VAR_FREQ1: f32 = 1.0 / 1_500.0;
const INLAND_VAR_AMP1:  f32 = 4.0;
const INLAND_VAR_FREQ2: f32 = 1.0 /   500.0;
const INLAND_VAR_AMP2:  f32 = 1.5;

const HARDNESS_FREQ:  f32 = 1.0 / 700.0;
const HARDNESS_PHASE: f32 = 271.3;

// ── Grid interpolation ────────────────────────────────────────────────────────
//
// The continent noise is expensive (90 simplex calls per column).
// We evaluate it on a 9×9 coarse grid and bilinearly interpolate to the full
// 32×32 chunk — 81 samples instead of 1024, a 12× speedup with negligible error
// because continentalness is smooth at 4-block scale.

const GRID_STRIDE: usize = 4;
const GRID_DIM:    usize = CHUNK_SIZE / GRID_STRIDE + 1; // = 9

// ── Public API ────────────────────────────────────────────────────────────────

pub struct TerrainColumn {
    pub surface_y: usize,
    pub is_ocean:  bool,
}

/// Fills `out_surface` and `out_ocean` for all 1024 columns in a chunk.
/// Returns the highest surface Y (used to cap the fill loop in world.rs).
pub fn sample_chunk_heights(
    defs:        &[ContinentDef; 9],
    origin_x:    i32,
    origin_z:    i32,
    out_surface: &mut [u16],
    out_ocean:   &mut [bool],
) -> usize {
    let mut c_grid  = [0.0f32; GRID_DIM * GRID_DIM];
    let mut n1_grid = [0.0f32; GRID_DIM * GRID_DIM];
    let mut n2_grid = [0.0f32; GRID_DIM * GRID_DIM];

    for gz in 0..GRID_DIM {
        for gx in 0..GRID_DIM {
            let wx = origin_x + (gx * GRID_STRIDE) as i32;
            let wz = origin_z + (gz * GRID_STRIDE) as i32;
            let (fx, fz) = (wx as f32, wz as f32);
            let gi = gx + gz * GRID_DIM;

            c_grid[gi]  = continentalness_defs(defs, fx, fz);
            n1_grid[gi] = simplex2d(fx * INLAND_VAR_FREQ1, fz * INLAND_VAR_FREQ1);
            n2_grid[gi] = simplex2d(fx * INLAND_VAR_FREQ2, fz * INLAND_VAR_FREQ2 + 17.3);
        }
    }

    let mut max_y = SEA_LEVEL;

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let gx0 = x / GRID_STRIDE;
            let gz0 = z / GRID_STRIDE;
            let gx1 = (gx0 + 1).min(GRID_DIM - 1);
            let gz1 = (gz0 + 1).min(GRID_DIM - 1);
            let tx = (x % GRID_STRIDE) as f32 / GRID_STRIDE as f32;
            let tz = (z % GRID_STRIDE) as f32 / GRID_STRIDE as f32;

            let c  = bilerp(&c_grid,  gx0, gz0, gx1, gz1, tx, tz);
            let n1 = bilerp(&n1_grid, gx0, gz0, gx1, gz1, tx, tz);
            let n2 = bilerp(&n2_grid, gx0, gz0, gx1, gz1, tx, tz);

            let idx = x + z * CHUNK_SIZE;
            let (sy, ocean) = surface_from_cv(c, n1, n2);
            out_surface[idx] = sy as u16;
            out_ocean[idx]   = ocean;
            if !ocean && sy > max_y { max_y = sy; }
        }
    }

    max_y
}

/// Single-column version kept for any code that needs ad-hoc sampling.
pub fn sample_column(defs: &[ContinentDef; 9], wx: i32, wz: i32) -> TerrainColumn {
    let (fx, fz) = (wx as f32, wz as f32);
    let c  = continentalness_defs(defs, fx, fz);
    let n1 = simplex2d(fx * INLAND_VAR_FREQ1, fz * INLAND_VAR_FREQ1);
    let n2 = simplex2d(fx * INLAND_VAR_FREQ2, fz * INLAND_VAR_FREQ2 + 17.3);
    let (surface_y, is_ocean) = surface_from_cv(c, n1, n2);
    TerrainColumn { surface_y, is_ocean }
}

pub fn rock_hardness_raw(x: f32, z: f32) -> f32 {
    let h1 = simplex2d(x * HARDNESS_FREQ + HARDNESS_PHASE, z * HARDNESS_FREQ);
    let h2 = simplex2d(x * HARDNESS_FREQ * 2.5,            z * HARDNESS_FREQ * 2.5 + 43.7);
    (h1 * 0.7 + h2 * 0.3) * 0.5 + 0.5
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn surface_from_cv(c: f32, n1: f32, n2: f32) -> (usize, bool) {
    let noise = n1 * INLAND_VAR_AMP1 + n2 * INLAND_VAR_AMP2;
    let sy = (SEA_LEVEL as f32 - c * HEIGHT_SCALE + noise).round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
}

#[inline]
fn bilerp(
    grid: &[f32],
    gx0: usize, gz0: usize,
    gx1: usize, gz1: usize,
    tx: f32, tz: f32,
) -> f32 {
    let v00 = grid[gx0 + gz0 * GRID_DIM];
    let v10 = grid[gx1 + gz0 * GRID_DIM];
    let v01 = grid[gx0 + gz1 * GRID_DIM];
    let v11 = grid[gx1 + gz1 * GRID_DIM];
    lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), tz)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t * (b - a) }
