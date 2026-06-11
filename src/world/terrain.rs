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

const LAND_DEPTH: f32 = 0.82;

// ── Grid interpolation ────────────────────────────────────────────────────────

const GRID_STRIDE: usize = 4;
const GRID_DIM:    usize = CHUNK_SIZE / GRID_STRIDE + 1; // 9

// ── Noise frequencies ─────────────────────────────────────────────────────────

// Single-octave terrain noise: one smooth frequency, consistent character everywhere.
// Multi-octave FBM was producing two visually distinct areas (smooth vs scarified)
// because octaves cancel/add gradients differently across space.  A single octave
// avoids that entirely — the terrain looks the same style in every region.
//
//   octave 0: freq 1/1400, amplitude 1.0 (after normalisation)
const TERRAIN_BASE_FREQ:  f32 = 1.0 / 4_000.0;
const TERRAIN_LACUNARITY: f32 = 2.0;   // unused with one octave, kept for easy expansion
const TERRAIN_PERSISTENCE:f32 = 0.5;   // unused with one octave, kept for easy expansion
const TERRAIN_OCTAVES:    u32 = 1;
const TERRAIN_SEEDS: [f32; 3] = [0.0, 37.1, 71.3];

const NOISE_FREQ_HARD: f32 = 1.0 / 900.0;  // rock-hardness patches

// ── Cliff debug ───────────────────────────────────────────────────────────────
//
// When true, all three cliff conditions use hard step functions (0 or 1) instead
// of smoothstep blends.  cliff_mask is therefore binary: columns either get the
// full boost or none at all, with no gradual inland ramp.  Useful for seeing
// exactly which columns qualify and whether the coastline alignment is correct.
const CLIFF_DEBUG_MODE: bool = false;

// ── Cliff tuning ──────────────────────────────────────────────────────────────
//
// Strategy: compute normal terrain height, then ADD a boost inside the coastal
// strip where rock is hard and terrain would already be mid-to-tall.  The boost
// peaks at the waterline (c → 0⁻) and fades back to zero at CLIFF_COAST_RANGE
// inland, creating a natural hillside climbing to a cliff top that drops sheer
// into the ocean.  Nothing outside the coastal strip is affected.

// Columns this many blocks above sea level or less are in the "coastal strip"
// where the cliff boost can activate.  The boost peaks right at the waterline
// (height_above_sea ≈ 0) and fades to zero at this threshold.
//
// Keep this small (≤ 25).  Tiny ocean islands sit 20–40 blocks above sea level
// throughout — a large window boosts every column uniformly, turning them into
// flat-top pillars.  Real continental coasts have their waterline columns right
// at sea level, so a narrow window still gives them a strong boost while leaving
// the island interior (> 25 blocks above sea) unaffected.
const CLIFF_COASTAL_WINDOW: f32 = 45.0;

// How much of the (land_h - base_sy) gap the boost actually fills.
// 1.0 = lift all the way to land_h (very tall); 0.45 = moderate cliff height.
const CLIFF_BOOST_SCALE: f32 = 0.45;

// noise_01 range where cliff eligibility ramps from 0 → 1.
const CLIFF_HEIGHT_LO: f32 = 0.48;
const CLIFF_HEIGHT_HI: f32 = 0.68;

// Hardness range.
const CLIFF_HARD_LO: f32 = 0.60;
const CLIFF_HARD_HI: f32 = 0.82;

// Very tall terrain (noise_01 > CLIFF_TALL_LO) can produce cliffs even in
// moderately soft rock — the terrain height alone is enough to suggest a cliff
// where the coast is naturally steep.  Scaled down so it's weaker than a
// genuine hard-rock cliff.
const CLIFF_TALL_LO: f32   = 0.68;
const CLIFF_TALL_HI: f32   = 0.88;
const CLIFF_TALL_SCALE: f32 = 0.65; // fraction of full cliff boost for tall-only case

// ── Coastal bluff tuning ──────────────────────────────────────────────────────
//
// Where rock is soft but terrain is elevated, a gentler coastal bluff forms
// instead of a sheer cliff — rounded eroded slopes rather than vertical faces.
// The bluff uses a wider coastal window (more gradual approach) and a smaller
// boost (less dramatic elevation change).

const BLUFF_COASTAL_WINDOW: f32 = 40.0; // wider than cliff window → longer gentle ramp
const BLUFF_BOOST_SCALE:    f32 = 0.18; // subtle — a modest bump, not a wall

// ── Island filter ─────────────────────────────────────────────────────────────
//
// Before applying cliff/bluff, sample continentalness at ±CLIFF_INLAND_PROBE
// blocks in X and Z from the column.  The minimum of those four probes tells us
// whether real continental land exists nearby:
//   min > -CLIFF_MIN_INLAND_C  → all directions hit ocean quickly → tiny island
//   min < -CLIFF_INLAND_C      → at least one direction is deeply inland → real coast
//
// This eliminates pillar-shaped islands: isolated noise blips in the ocean get
// land_size_cond=0 (no boost) while continental coastlines get land_size_cond=1.

const CLIFF_INLAND_PROBE:  f32 = 200.0;
const CLIFF_MIN_INLAND_C:  f32 = 0.05;  // below this → suppress cliff entirely
const CLIFF_INLAND_C:      f32 = 0.15;  // above this (more negative) → full cliff

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
    let mut c_grid      = [0.0f32; GRID_DIM * GRID_DIM];
    let mut noise_grid  = [0.0f32; GRID_DIM * GRID_DIM];
    let mut inland_grid = [0.0f32; GRID_DIM * GRID_DIM];

    for gz in 0..GRID_DIM {
        for gx in 0..GRID_DIM {
            let wx = origin_x + (gx * GRID_STRIDE) as i32;
            let wz = origin_z + (gz * GRID_STRIDE) as i32;
            let (fx, fz) = (wx as f32, wz as f32);
            let gi = gx + gz * GRID_DIM;

            c_grid[gi]      = continentalness_defs(defs, fx, fz);
            noise_grid[gi]  = terrain_noise(fx, fz);
            inland_grid[gi] = inland_depth(defs, fx, fz);
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

            let c        = bilerp(&c_grid,      gx0, gz0, gx1, gz1, tx, tz);
            let noise_01 = bilerp(&noise_grid,  gx0, gz0, gx1, gz1, tx, tz) * 0.5 + 0.5;
            let inland   = bilerp(&inland_grid, gx0, gz0, gx1, gz1, tx, tz);

            let idx = x + z * CHUNK_SIZE;
            let (sy, ocean) = surface_from_cv(c, noise_01, inland);
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
    let inl      = inland_depth(defs, fx, fz);
    let (surface_y, is_ocean) = surface_from_cv(c, noise_01, inl);
    TerrainColumn { surface_y, is_ocean }
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// 3-octave FBM terrain noise.  Returns a value in [-1, 1].
fn terrain_noise(fx: f32, fz: f32) -> f32 {
    let mut val  = 0.0_f32;
    let mut amp  = TERRAIN_PERSISTENCE; // first octave amplitude = 0.5
    let mut freq = TERRAIN_BASE_FREQ;
    for i in 0..TERRAIN_OCTAVES as usize {
        val  += simplex2d(fx * freq, fz * freq + TERRAIN_SEEDS[i]) * amp;
        amp  *= TERRAIN_PERSISTENCE;
        freq *= TERRAIN_LACUNARITY;
    }
    // single octave: max |val| = TERRAIN_PERSISTENCE = 0.5; normalise to [-1, 1]
    (val / TERRAIN_PERSISTENCE).clamp(-1.0, 1.0)
}

fn surface_from_cv(c: f32, noise_01: f32, inland: f32) -> (usize, bool) {
    let ocean_h = lerp(OCEAN_FLOOR_MIN, OCEAN_FLOOR_MAX, noise_01);
    let land_h  = lerp(LAND_HEIGHT_MIN, LAND_HEIGHT_MAX, noise_01);

    let cf      = cont_factor(c);
    let base_sy = lerp(ocean_h, land_h, cf);

    // Coastal cliff boost: peaks at the waterline, fades to zero at CLIFF_COASTAL_WINDOW
    // blocks above sea level.  Only depends on smooth fields (noise_01, coast_strip,
    // land_size_cond) — no hardness noise, so the approach zone is uniformly smooth.
    let height_above_sea = base_sy - SEA_LEVEL as f32;
    let coast_strip = if height_above_sea > 0.0 {
        smoothstep(CLIFF_COASTAL_WINDOW, 0.0, height_above_sea)
    } else {
        0.0
    };

    let height_cond    = smoothstep(CLIFF_HEIGHT_LO, CLIFF_HEIGHT_HI, noise_01);
    let land_size_cond = smoothstep(-CLIFF_MIN_INLAND_C, -CLIFF_INLAND_C, inland);
    let cliff_mask     = coast_strip * height_cond * land_size_cond;

    let gap = land_h - base_sy;
    let sy  = (base_sy + gap * cliff_mask * CLIFF_BOOST_SCALE)
        .round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
}

/// Minimum continentalness found ±CLIFF_INLAND_PROBE blocks in the four cardinal
/// directions.  Negative = real continent exists nearby; near-zero or positive = isolated.
fn inland_depth(defs: &[ContinentDef; 9], fx: f32, fz: f32) -> f32 {
    let p = CLIFF_INLAND_PROBE;
    continentalness_defs(defs, fx + p, fz)
        .min(continentalness_defs(defs, fx - p, fz))
        .min(continentalness_defs(defs, fx, fz + p))
        .min(continentalness_defs(defs, fx, fz - p))
}

/// Continental factor: 0.0 = deep ocean, 0.5 = coastline, 1.0 = deep inland.
fn cont_factor(c: f32) -> f32 {
    if c >= 0.0 {
        let t = smoothstep(0.0, CONT_THRESHOLD, c);
        0.5 * (1.0 - t)
    } else {
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
