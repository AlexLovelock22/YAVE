use crate::world::{
    chunk::CHUNK_SIZE,
    continents::{continentalness_defs, ContinentDef, SEA_LEVEL},
    noise::simplex2d,
};

// ── Architecture ──────────────────────────────────────────────────────────────
//
// Two independent maps, combined per column:
//
//   1. Continent map (continents.rs): c <= 0 is land, c > 0 is ocean.  This is
//      the ONLY thing that decides the coastline — changing the terrain noise
//      below never moves the shore.
//
//   2. Noise map (NOISE_FREQ fBm): pure height detail.  On land the noise
//      scales the surface up from sea level, with the amplitude growing the
//      further inland we are.  Tune NOISE_FREQ / octaves freely.
//
// Cliffs are a separate second pass: a smooth trend-level cliff surface is
// merged with the land trend via smooth_max, and the same noise rides over
// the result (see the Cliff system comment below).

// ── Height ranges ─────────────────────────────────────────────────────────────

const OCEAN_FLOOR_MIN: f32 = 30.0;   // deepest noisy ocean floor
const OCEAN_FLOOR_MAX: f32 = 105.0;  // shallowest noisy ocean floor
const OCEAN_DEPTH_C:   f32 = 0.15;   // c-units over which the floor descends from the shore
const LAND_HEIGHT_MAX: f32 = 235.0;  // max land height (deep inland, noise = 1)
const LAND_DEPTH:      f32 = 0.35;   // c-units over which land amplitude reaches full (~500 blocks)

// ── Noise map ─────────────────────────────────────────────────────────────────
//
// The single height-detail noise.  Smaller denominator = lumpier terrain.
// Safe to tune freely — the coastline comes from the continent map alone.

const NOISE_FREQ: f32 = 1.0 / 400.0;
const NOISE_SEED: f32 = 7.3;

// ── Grid interpolation ────────────────────────────────────────────────────────

const GRID_STRIDE: usize = 4;
const GRID_DIM:    usize = CHUNK_SIZE / GRID_STRIDE + 1; // 9

// ── Cliff system ──────────────────────────────────────────────────────────────
//
// The land terrain is decomposed into trend + noise: the trend is the land
// height with the noise pinned at its midpoint (0.5), the noise part is
// whatever the real noise adds on top.  The cliff is a smooth trend-level
// surface — a plateau at SEA_LEVEL + blob·CLIFF_HEIGHT extending
// PLATEAU_DEPTH_C inland, then descending at a constant gradient
// (CLIFF_RAMP_BLOCKS_PER_C).  The final height is:
//
//     smooth_max(land trend, cliff trend) + noise part
//
// Because the max merges two smooth noise-free surfaces and the SAME noise
// rides over the result everywhere, the mainland hills continue unbroken onto
// the cliff top and the meeting line carries no seam.
//
// The cliff wall forms at the shoreline because ocean columns ignore the
// cliff surface entirely (the c > 0 guard in surface_from_cv), while the
// adjacent land columns sit on the full-height plateau.

const CLIFF_HEIGHT: f32 = 60.0;
const PLATEAU_DEPTH_C: f32 = 0.06;
const RAMP_ONSET_C: f32 = 0.06;
const CLIFF_RAMP_BLOCKS_PER_C: f32 = 10.0;

const FALL_WARP_FREQ:   f32 = 1.0 / 700.0;
const FALL_WARP_AMP_C:  f32 = 0.05;
const FALL_WARP_FREQ2:  f32 = 1.0 / 180.0;
const FALL_WARP_AMP2_C: f32 = 0.012;
const FALL_WARP_RAMP_C: f32 = 0.06;

const TAKEOVER_SMOOTH: f32 = 16.0;
const CLIFF_EDGE_DROP: f32 = 30.0;

const TOP_FREQ1: f32 = 1.0 / 400.0;
const TOP_AMP1:  f32 = 8.0;
const TOP_FREQ2: f32 = 1.0 / 120.0;
const TOP_AMP2:  f32 = 3.5;

// These must stay in sync with export.rs so the preview map matches the game.
const CLIFF_BLOB_FREQ: f32 = 1.0 / 3_600.0;
const CLIFF_SLOW_FREQ: f32 = 1.0 / 15_000.0;
const CLIFF_SEED:      f32 = 31.7;
const CLIFF_THRESHOLD: f32 = 0.55;
const CLIFF_BLEND:     f32 = 0.10;

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

            let c = continentalness_defs(defs, fx, fz);
            c_grid[gi]     = c;
            noise_grid[gi] = height_noise(fx, fz);   // [-1,1] for bilerp
            cliff_grid[gi] = cliff_at(fx, fz, c);
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
    let noise_01 = height_noise(fx, fz) * 0.5 + 0.5;
    let cliff    = cliff_at(fx, fz, c);
    let (surface_y, is_ocean) = surface_from_cv(c, noise_01, cliff);
    TerrainColumn { surface_y, is_ocean }
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Land height: sea level scaled up by the noise, amplitude growing inland.
/// Exactly SEA_LEVEL at the shoreline (c = 0) so the coast meets the ocean
/// floor with no lip.
fn land_height(c: f32, noise_01: f32) -> f32 {
    let inland = smoothstep(0.0, LAND_DEPTH, -c);
    let amp    = inland * (LAND_HEIGHT_MAX - SEA_LEVEL as f32);
    SEA_LEVEL as f32 + noise_01 * amp
}

/// Land trend: land_height with the noise pinned at its midpoint (0.5).
/// The noise-free base used for the cliff smooth-max seam.
fn land_trend(c: f32) -> f32 {
    land_height(c, 0.5)
}

/// Ocean floor: starts at SEA_LEVEL right at the shoreline (the floor() in
/// surface_from_cv puts the first wet column 1 block under water), descending
/// to a noisy deep floor over OCEAN_DEPTH_C — continuous with the land side.
fn ocean_floor(c: f32, noise_01: f32) -> f32 {
    let t    = smoothstep(0.0, OCEAN_DEPTH_C, c);
    let deep = lerp(OCEAN_FLOOR_MIN, OCEAN_FLOOR_MAX, noise_01);
    lerp(SEA_LEVEL as f32, deep, t)
}

/// Final height.  Ocean (c > 0) gets the floor; land merges the land trend
/// with the cliff trend via smooth_max, then adds the noise part on top.
fn surface_from_cv(c: f32, noise_01: f32, cliff_h: f32) -> (usize, bool) {
    if c > 0.0 {
        let sy = ocean_floor(c, noise_01).floor().max(1.0) as usize;
        return (sy, true);
    }
    let trend      = land_trend(c);
    let noise_part = land_height(c, noise_01) - trend;
    let h  = smooth_max(trend, cliff_h, TAKEOVER_SMOOTH) + noise_part;
    let sy = h.round().max(1.0) as usize;
    (sy, false)
}

fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    lerp(b, a, h) + k * h * (1.0 - h)
}

/// The cliff surface height (blocks) at a column: a plateau at
/// SEA_LEVEL + blob·CLIFF_HEIGHT, descending at a constant gradient past
/// PLATEAU_DEPTH_C inland.  Not gated on ocean — water columns ignore it in
/// surface_from_cv, and leaving it ungated keeps the wall exactly at the
/// shoreline through the grid interpolation.
fn cliff_at(fx: f32, fz: f32, c: f32) -> f32 {
    let blob   = cliff_noise(fx, fz);
    let inland = -c;
    let warp = (simplex2d(fx * FALL_WARP_FREQ,  fz * FALL_WARP_FREQ  + CLIFF_SEED + 13.1) * FALL_WARP_AMP_C
              + simplex2d(fx * FALL_WARP_FREQ2, fz * FALL_WARP_FREQ2 + CLIFF_SEED + 27.4) * FALL_WARP_AMP2_C)
             * smoothstep(0.0, FALL_WARP_RAMP_C, inland);
    let ramp = smooth_max(0.0, inland + warp - PLATEAU_DEPTH_C, RAMP_ONSET_C);
    let base = SEA_LEVEL as f32 + blob * CLIFF_HEIGHT - (1.0 - blob) * CLIFF_EDGE_DROP
             - ramp * CLIFF_RAMP_BLOCKS_PER_C;
    // Detail octaves: zero at the seam (where cliff ≈ land trend), full on
    // the open plateau.
    let terrain_trend = land_trend(c);
    let lift_blend = ((base - terrain_trend) / TAKEOVER_SMOOTH).clamp(0.0, 1.0);
    let detail = (simplex2d(fx * TOP_FREQ1, fz * TOP_FREQ1 + CLIFF_SEED + 41.3) * TOP_AMP1
                + simplex2d(fx * TOP_FREQ2, fz * TOP_FREQ2 + CLIFF_SEED + 53.7) * TOP_AMP2)
               * lift_blend;
    base + detail
}

/// 2-D blob noise in world space — determines which sections of coast are cliffs.
/// Identical to the arc_noise in export.rs so the map preview matches the game.
fn cliff_noise(fx: f32, fz: f32) -> f32 {
    let n1      = simplex2d(fx * CLIFF_BLOB_FREQ, fz * CLIFF_BLOB_FREQ + CLIFF_SEED);
    let n2      = simplex2d(fx * CLIFF_SLOW_FREQ, fz * CLIFF_SLOW_FREQ + CLIFF_SEED + 7.3);
    let noise01 = (n1 * 0.65 + n2 * 0.35) * 0.5 + 0.5;
    smoothstep(CLIFF_THRESHOLD, CLIFF_THRESHOLD + CLIFF_BLEND, noise01)
}

/// The height-detail noise map: 3-octave fBm in [-1, 1].
fn height_noise(fx: f32, fz: f32) -> f32 {
    let n1 = simplex2d(fx * NOISE_FREQ,       fz * NOISE_FREQ       + NOISE_SEED);
    let n2 = simplex2d(fx * NOISE_FREQ * 2.0, fz * NOISE_FREQ * 2.0 + NOISE_SEED + 1.7);
    let n3 = simplex2d(fx * NOISE_FREQ * 4.0, fz * NOISE_FREQ * 4.0 + NOISE_SEED + 3.5);
    (n1 * 0.5 + n2 * 0.35 + n3 * 0.15).clamp(-1.0, 1.0)
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
