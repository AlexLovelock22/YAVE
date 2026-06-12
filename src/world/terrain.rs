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

/// Multiplier on TERRAIN_FREQ, set once from settings.toml (`noise_density`)
/// before any chunk generation starts.  Stored as f32 bits.
static NOISE_DENSITY: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(f32::to_bits(1.0));

pub fn set_noise_density(d: f32) {
    NOISE_DENSITY.store(d.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn noise_density() -> f32 {
    f32::from_bits(NOISE_DENSITY.load(std::sync::atomic::Ordering::Relaxed))
}

// ── Grid interpolation ────────────────────────────────────────────────────────

const GRID_STRIDE: usize = 4;
const GRID_DIM:    usize = CHUNK_SIZE / GRID_STRIDE + 1; // 9

// ── Cliff system ──────────────────────────────────────────────────────────────
//
// The terrain is decomposed into trend + noise: the trend is the heightmap
// with terrain noise pinned at its midpoint, the noise part is whatever the
// real noise adds on top.  The cliff is a smooth trend-level surface — a
// plateau at SEA_LEVEL + blob·CLIFF_HEIGHT extending PLATEAU_DEPTH_C inland,
// then descending at a constant gradient (CLIFF_RAMP_BLOCKS_PER_C).  The
// final height is
//
//     smooth_max(terrain trend, cliff trend) + noise part
//
// Because the max merges two smooth noise-free surfaces and the SAME terrain
// noise rides over the result everywhere, the mainland hills continue
// unbroken onto the cliff top and the meeting line carries no seam: at the
// crossover both sides are literally the same surface.
//
// The cliff wall forms at the shoreline because ocean columns ignore the
// cliff surface entirely (the is_ocean guard in surface_from_cv), while the
// adjacent land columns sit on the full-height plateau.
//
// A 2-D world-space blob noise scales the plateau height and decides which
// stretches of coastline are cliffs.  It is the same noise used in the export
// map, so the red sections on the map correspond to where cliffs form in-game.

// Plateau height above sea level (blocks) where the blob noise is full.
const CLIFF_HEIGHT: f32 = 60.0;

// How far inland the plateau extends before the ramp starts, in
// continentalness units (-c grows inland; near the coast c changes by roughly
// 0.0005 per block, so 0.06 ≈ 120 blocks).
const PLATEAU_DEPTH_C: f32 = 0.06;

// Constant descent gradient of the ramp beyond the plateau.  In block terms
// this is ≈ 280 × 0.0005 = 0.14 blocks of descent per block walked inland.
const CLIFF_RAMP_BLOCKS_PER_C: f32 = 280.0;

// Rounding of the plateau shoulder where the ramp begins (smooth-max width in
// continentalness units): the top eases into the descent instead of kinking.
const RAMP_ONSET_C: f32 = 0.06;

// Domain warp on the inland coordinate so the plateau edge and ramp wander
// instead of running parallel to the coast.  Two octaves: a broad sweep and a
// smaller-scale jitter.  Fades in past the shore (FALL_WARP_RAMP_C) so the
// wall and the seaward plateau keep full height.
const FALL_WARP_FREQ:    f32 = 1.0 / 700.0;
const FALL_WARP_AMP_C:   f32 = 0.05;  // ≈ ±100 blocks of edge wander
const FALL_WARP_FREQ2:   f32 = 1.0 / 180.0;
const FALL_WARP_AMP2_C:  f32 = 0.012; // ≈ ±24 blocks of fine wander
const FALL_WARP_RAMP_C:  f32 = 0.06;

// Vertical range (blocks) over which the cliff trend and the terrain trend
// merge via smooth-max — a rounded saddle instead of a sharp crease.  The
// merge happens in trend space (noise-free), so it can be generous.
const TAKEOVER_SMOOTH: f32 = 16.0;


// How far the cliff trend sinks below sea level where the blob is 0, scaled
// by (1 - blob).  This keeps the field continuous across the blob threshold
// (a 0.0 sentinel puts a visible step along the blob contour inland) and
// holds it decisively below the terrain trend on non-cliff coastline, out of
// the smooth-max's reach.
const CLIFF_EDGE_DROP: f32 = 30.0;


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
    let noise_01 = terrain_noise(fx, fz) * 0.5 + 0.5;
    let cliff    = cliff_at(fx, fz, c);
    let (surface_y, is_ocean) = surface_from_cv(c, noise_01, cliff);
    TerrainColumn { surface_y, is_ocean }
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Base height without any cliff boost, as a continuous value.
fn base_height_f(c: f32, noise_01: f32) -> f32 {
    let ocean_h = lerp(OCEAN_FLOOR_MIN, OCEAN_FLOOR_MAX, noise_01);
    let land_h  = lerp(LAND_HEIGHT_MIN, LAND_HEIGHT_MAX, noise_01);
    lerp(ocean_h, land_h, cont_factor(c))
}

/// Base height without any cliff boost — used for coastal neighbour probing.
fn base_surface(c: f32, noise_01: f32) -> (usize, bool) {
    let sy = base_height_f(c, noise_01).round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
}

/// Final height: smooth_max of the terrain trend and the cliff trend, with
/// the terrain noise riding over the result (see the Cliff system comment).
/// Ocean columns ignore the cliff surface, which is what forms the wall at
/// the shoreline.
fn surface_from_cv(c: f32, noise_01: f32, cliff_h: f32) -> (usize, bool) {
    let (base_sy, is_ocean) = base_surface(c, noise_01);
    if is_ocean {
        return (base_sy, is_ocean);
    }
    let trend      = base_height_f(c, 0.5);
    let noise_part = base_height_f(c, noise_01) - trend;
    let h  = smooth_max(trend, cliff_h, TAKEOVER_SMOOTH) + noise_part;
    let sy = h.round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
}

/// Polynomial smooth maximum: equals max(a, b) when |a - b| > k, otherwise
/// rounds the crossover into a fillet up to k/4 above the higher input.
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
    let blob = cliff_noise(fx, fz);
    let inland = -c;
    // Warp the inland coordinate so the plateau edge wanders (see FALL_WARP_*).
    let warp = (simplex2d(fx * FALL_WARP_FREQ, fz * FALL_WARP_FREQ + CLIFF_SEED + 13.1)
            * FALL_WARP_AMP_C
        + simplex2d(fx * FALL_WARP_FREQ2, fz * FALL_WARP_FREQ2 + CLIFF_SEED + 27.4)
            * FALL_WARP_AMP2_C)
        * smoothstep(0.0, FALL_WARP_RAMP_C, inland);
    // Smooth onset: the plateau shoulder eases into the descent.
    let ramp = smooth_max(0.0, inland + warp - PLATEAU_DEPTH_C, RAMP_ONSET_C);
    SEA_LEVEL as f32 + blob * CLIFF_HEIGHT - (1.0 - blob) * CLIFF_EDGE_DROP
        - ramp * CLIFF_RAMP_BLOCKS_PER_C
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
    let f = TERRAIN_FREQ * noise_density();
    simplex2d(fx * f, fz * f + TERRAIN_SEED).clamp(-1.0, 1.0)
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
