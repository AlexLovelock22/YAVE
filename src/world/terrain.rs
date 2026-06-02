use crate::world::{
    chunk::CHUNK_SIZE,
    continents::{continentalness_defs, ContinentDef, CONT_THRESHOLD, SEA_LEVEL},
    noise::simplex2d,
};

// ── Height ranges ─────────────────────────────────────────────────────────────
//
// The continent map blends two independent height ranges using the same noise:
//   • Ocean floor: noise drives depth variation across the seabed
//   • Land terrain: noise drives hills and mountains above sea level
// The continental factor (0 = deep ocean, 1 = deep inland) blends between them,
// so continental areas naturally rise out of the ocean floor.

const OCEAN_FLOOR_MIN: f32 = 30.0;   // deepest open ocean
const OCEAN_FLOOR_MAX: f32 = 105.0;  // shallow shelf / near coastline

const LAND_HEIGHT_MIN: f32 = 115.0;  // low coastal plains
const LAND_HEIGHT_MAX: f32 = 235.0;  // mountain peaks

// Continentalness c: negative = land (dome above threshold), positive = ocean.
// Maximum expected magnitude on each side.
const LAND_DEPTH:  f32 = 0.82;        // |c| at deepest inland point
// CONT_THRESHOLD (0.18) is the max c value in open ocean

// ── Grid interpolation ────────────────────────────────────────────────────────
//
// Continent noise is expensive (90 simplex calls/column). Evaluate on a coarse
// grid and bilinearly interpolate — 81 samples instead of 1024 per chunk.

const GRID_STRIDE: usize = 4;
const GRID_DIM:    usize = CHUNK_SIZE / GRID_STRIDE + 1; // 9

// Noise frequencies — two octaves for terrain variation.
const NOISE_FREQ1: f32 = 1.0 / 1_500.0;
const NOISE_FREQ2: f32 = 1.0 /   500.0;

// ── Public API ────────────────────────────────────────────────────────────────

pub struct TerrainColumn {
    pub surface_y: usize,
    pub is_ocean:  bool,
}

/// Fill `out_surface` and `out_ocean` for all 1024 columns in one chunk.
/// Returns the highest non-ocean surface Y seen (caps mesh iteration).
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
            n1_grid[gi] = simplex2d(fx * NOISE_FREQ1, fz * NOISE_FREQ1);
            n2_grid[gi] = simplex2d(fx * NOISE_FREQ2, fz * NOISE_FREQ2 + 17.3);
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

pub fn sample_column(defs: &[ContinentDef; 9], wx: i32, wz: i32) -> TerrainColumn {
    let (fx, fz) = (wx as f32, wz as f32);
    let c  = continentalness_defs(defs, fx, fz);
    let n1 = simplex2d(fx * NOISE_FREQ1, fz * NOISE_FREQ1);
    let n2 = simplex2d(fx * NOISE_FREQ2, fz * NOISE_FREQ2 + 17.3);
    let (surface_y, is_ocean) = surface_from_cv(c, n1, n2);
    TerrainColumn { surface_y, is_ocean }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Blend ocean-floor and land-terrain height ranges using the continental factor.
/// The same noise drives variation in both ranges, so the ocean floor and land
/// share the same "shape" — continents are literally the seabed pushed upward.
fn surface_from_cv(c: f32, n1: f32, n2: f32) -> (usize, bool) {
    // Map two noise octaves to [0, 1].
    let noise_01 = (n1 * 0.7 + n2 * 0.3).clamp(-1.0, 1.0) * 0.5 + 0.5;

    let ocean_h = lerp(OCEAN_FLOOR_MIN, OCEAN_FLOOR_MAX, noise_01);
    let land_h  = lerp(LAND_HEIGHT_MIN, LAND_HEIGHT_MAX, noise_01);

    let cf = cont_factor(c);
    let sy = lerp(ocean_h, land_h, cf).round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
}

/// Continental factor: 0.0 = deep ocean, 1.0 = deep inland.
/// Splits on c=0 (the coast) and maps each side to [0, 0.5] or [0.5, 1].
/// This keeps the midpoint (0.5) exactly at the coastline so the blended
/// height hits SEA_LEVEL when noise_01 ≈ 0.5.
fn cont_factor(c: f32) -> f32 {
    if c >= 0.0 {
        // Ocean side: c from 0 → CONT_THRESHOLD
        let t = smoothstep(0.0, CONT_THRESHOLD, c);
        0.5 * (1.0 - t)
    } else {
        // Land side: c from 0 → -LAND_DEPTH
        let t = smoothstep(0.0, -LAND_DEPTH, c);
        0.5 + 0.5 * t
    }
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
