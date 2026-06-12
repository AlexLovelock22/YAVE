use crate::world::{
    chunk::CHUNK_SIZE,
    continents::{continentalness_defs, ContinentDef, CONT_THRESHOLD, SEA_LEVEL},
    noise::simplex2d,
};

// ── Height ranges ─────────────────────────────────────────────────────────────

const OCEAN_FLOOR_MIN: f32 = 30.0;
const OCEAN_FLOOR_MAX: f32 = 105.0;
const LAND_HEIGHT_MIN: f32 = 115.0;
const LAND_HEIGHT_MAX: f32 = 235.0;
const LAND_DEPTH:      f32 = 0.82;

// ── Terrain noise ─────────────────────────────────────────────────────────────

const TERRAIN_FREQ: f32 = 1.0 / 4_000.0;
const TERRAIN_SEED: f32 = 0.0;

// ── Grid interpolation ────────────────────────────────────────────────────────

const GRID_STRIDE: usize = 4;
const GRID_DIM:    usize = CHUNK_SIZE / GRID_STRIDE + 1; // 9

// ── Cliff system ──────────────────────────────────────────────────────────────
//
// Second-pass approach: base heights are computed first, then each land column
// checks whether any cardinal neighbour (COASTAL_PROBE blocks away) is ocean.
// If coastal, a 2-D world-space blob noise decides whether this stretch of
// coastline is a cliff.  The blob noise is the same one used in the export map,
// so the red sections on the map correspond exactly to where cliffs form in-game.

// How far away (blocks) to probe for ocean neighbours when classifying a column
// as coastal.  Must be > GRID_STRIDE so it always crosses at least one grid cell.
const COASTAL_PROBE: f32 = 32.0;

// Height the cliff top sits above sea level (blocks).
const CLIFF_HEIGHT: f32 = 60.0;

// These must stay in sync with export.rs so the preview map matches the game.
const CLIFF_BLOB_FREQ:  f32 = 1.0 / 3_600.0;
const CLIFF_SLOW_FREQ:  f32 = 1.0 / 15_000.0;
const CLIFF_SEED:       f32 = 31.7;
const CLIFF_THRESHOLD:  f32 = 0.55;
const CLIFF_BLEND:      f32 = 0.10;

// ── Public API ────────────────────────────────────────────────────────────────

pub struct TerrainColumn {
    pub surface_y: usize,
    pub is_ocean:  bool,
}

pub fn sample_chunk_heights(
    defs:        &[ContinentDef; 9],
    origin_x:    i32,
    origin_z:    i32,
    out_surface: &mut [u16],
    out_ocean:   &mut [bool],
) -> usize {
    let mut c_grid     = [0.0f32; GRID_DIM * GRID_DIM];
    let mut noise_grid = [0.0f32; GRID_DIM * GRID_DIM];
    let mut cliff_grid = [0.0f32; GRID_DIM * GRID_DIM];

    for gz in 0..GRID_DIM {
        for gx in 0..GRID_DIM {
            let wx = origin_x + (gx * GRID_STRIDE) as i32;
            let wz = origin_z + (gz * GRID_STRIDE) as i32;
            let (fx, fz) = (wx as f32, wz as f32);
            let gi = gx + gz * GRID_DIM;

            let c        = continentalness_defs(defs, fx, fz);
            let noise_01 = terrain_noise(fx, fz) * 0.5 + 0.5;
            c_grid[gi]     = c;
            noise_grid[gi] = noise_01 * 2.0 - 1.0; // store raw [-1,1] for bilerp
            cliff_grid[gi] = cliff_at(defs, fx, fz, c, noise_01);
        }
    }

    let mut max_y = SEA_LEVEL;

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let gx0 = x / GRID_STRIDE;
            let gz0 = z / GRID_STRIDE;
            let gx1 = (gx0 + 1).min(GRID_DIM - 1);
            let gz1 = (gz0 + 1).min(GRID_DIM - 1);
            let tx  = (x % GRID_STRIDE) as f32 / GRID_STRIDE as f32;
            let tz  = (z % GRID_STRIDE) as f32 / GRID_STRIDE as f32;

            let c        = bilerp(&c_grid,     gx0, gz0, gx1, gz1, tx, tz);
            let noise_01 = bilerp(&noise_grid, gx0, gz0, gx1, gz1, tx, tz) * 0.5 + 0.5;
            let cliff    = bilerp(&cliff_grid, gx0, gz0, gx1, gz1, tx, tz);

            let idx = x + z * CHUNK_SIZE;
            let (sy, ocean) = surface_from_cv(c, noise_01, cliff);
            out_surface[idx] = sy as u16;
            out_ocean[idx]   = ocean;
            if !ocean && sy > max_y { max_y = sy; }
        }
    }

    max_y
}

pub fn sample_column(defs: &[ContinentDef; 9], wx: i32, wz: i32) -> TerrainColumn {
    let (fx, fz) = (wx as f32, wz as f32);
    let c        = continentalness_defs(defs, fx, fz);
    let noise_01 = terrain_noise(fx, fz) * 0.5 + 0.5;
    let cliff    = cliff_at(defs, fx, fz, c, noise_01);
    let (surface_y, is_ocean) = surface_from_cv(c, noise_01, cliff);
    TerrainColumn { surface_y, is_ocean }
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Base height without any cliff boost — used for coastal neighbour probing.
fn base_surface(c: f32, noise_01: f32) -> (usize, bool) {
    let ocean_h = lerp(OCEAN_FLOOR_MIN, OCEAN_FLOOR_MAX, noise_01);
    let land_h  = lerp(LAND_HEIGHT_MIN, LAND_HEIGHT_MAX, noise_01);
    let sy      = lerp(ocean_h, land_h, cont_factor(c)).round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
}

/// Full height including cliff boost.
fn surface_from_cv(c: f32, noise_01: f32, cliff: f32) -> (usize, bool) {
    let (base_sy, is_ocean) = base_surface(c, noise_01);
    if is_ocean || cliff <= 0.0 {
        return (base_sy, is_ocean);
    }
    let cliff_top = SEA_LEVEL as f32 + CLIFF_HEIGHT;
    let sy = lerp(base_sy as f32, cliff_top.max(base_sy as f32), cliff)
        .round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
}

/// Returns the cliff weight [0,1] for a land column: 0 if non-coastal or noise
/// says no cliff, otherwise the smoothstepped blob noise value.
fn cliff_at(defs: &[ContinentDef; 9], fx: f32, fz: f32, c: f32, noise_01: f32) -> f32 {
    let (_, is_ocean) = base_surface(c, noise_01);
    if is_ocean { return 0.0; }
    if !is_coastal(defs, fx, fz) { return 0.0; }
    cliff_noise(fx, fz)
}

/// True if any of the four cardinal neighbours at COASTAL_PROBE distance is ocean.
fn is_coastal(defs: &[ContinentDef; 9], fx: f32, fz: f32) -> bool {
    let p = COASTAL_PROBE;
    for (dx, dz) in [(p, 0.0_f32), (-p, 0.0), (0.0, p), (0.0, -p)] {
        let c2 = continentalness_defs(defs, fx + dx, fz + dz);
        let n2 = terrain_noise(fx + dx, fz + dz) * 0.5 + 0.5;
        let (_, ocean) = base_surface(c2, n2);
        if ocean { return true; }
    }
    false
}

/// 2-D blob noise in world space — determines which sections of coast are cliffs.
/// Identical to the arc_noise in export.rs so the map preview matches the game.
fn cliff_noise(fx: f32, fz: f32) -> f32 {
    let n1      = simplex2d(fx * CLIFF_BLOB_FREQ, fz * CLIFF_BLOB_FREQ + CLIFF_SEED);
    let n2      = simplex2d(fx * CLIFF_SLOW_FREQ, fz * CLIFF_SLOW_FREQ + CLIFF_SEED + 7.3);
    let noise01 = (n1 * 0.65 + n2 * 0.35) * 0.5 + 0.5;
    smoothstep(CLIFF_THRESHOLD, CLIFF_THRESHOLD + CLIFF_BLEND, noise01)
}

fn terrain_noise(fx: f32, fz: f32) -> f32 {
    simplex2d(fx * TERRAIN_FREQ, fz * TERRAIN_FREQ + TERRAIN_SEED).clamp(-1.0, 1.0)
}

fn cont_factor(c: f32) -> f32 {
    if c >= 0.0 {
        0.5 * (1.0 - smoothstep(0.0, CONT_THRESHOLD, c))
    } else {
        0.5 + 0.5 * smoothstep(0.0, -LAND_DEPTH, c)
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
