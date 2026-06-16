use crate::world::{
    chunk::{CHUNK_SIZE, CHUNK_HEIGHT},
    continents::{continentalness_defs, ContinentDef, SEA_LEVEL},
    noise::simplex2d,
};

// ── Terrain ───────────────────────────────────────────────────────────────────

const C_SCALE:    f32 = 250.0;  // height contribution per unit c
const NOISE_AMP:  f32 = 30.0;   // noise amplitude (full on ocean, reduced on land)
const LAND_NOISE: f32 = 0.35;   // noise amplitude scale on land

const NOISE_FREQ: f32 = 1.0 / 700.0;
const NOISE_SEED: f32 = 0.0;

// Higher-frequency multi-octave detail added to CLIFF TOPS so they read as varied
// terrain rather than one smooth dome. Only applied on cliffs, not base terrain.
const DETAIL_FREQ: f32 = 1.0 / 300.0;
const DETAIL_SEED: f32 = 19.3;

// ── Cliffs ────────────────────────────────────────────────────────────────────
//
// SECOND PASS over the combined map (continent + noise = `terrain_h`):
//
//   1. DESIGNATE — `cliff_n` marks which stretches of shore get a cliff. Sampled at
//      the nearest COAST point (see coast_fields) so the whole inland strip behind a
//      designated coast inherits it. Used only as a lateral (along-coast) fade.
//   2. PROFILE   — by distance-to-coast `dist`:
//        dist < CLIFF_FLAT_DIST                       → flat clifftop (SEA+CLIFF_HEIGHT)
//        FLAT .. FLAT + CLIFF_DESCENT_DIST            → blend clifftop → natural terrain
//        beyond                                       → natural terrain
//   3. BLEND     — the descent is a LERP from the clifftop to the actual terrain
//      height, so it always lands EXACTLY on natural terrain (seamless), and terrain
//      detail fades back in gradually instead of stabbing through a fixed plateau.
//      A final max(…, terrain) means the cliff only ever RAISES land, so naturally
//      high coasts are preserved rather than carved down.
//
// `dist` is a real block distance recovered from the implicit continent field via
// the first-order Eikonal estimate dist ≈ c/‖∇c‖ (a true distance field has unit
// gradient, so dividing the field by its gradient magnitude converts field-units to
// blocks). ‖∇c‖ and a low-passed c are measured over a LARGE baseline so coastline
// noise is averaged out and `dist` rises monotonically inland.
//
// Because the descent lands on terrain by construction, the profile self-terminates
// at FLAT + DESCENT — no max-distance fade or medial-axis patching needed.

const CLIFF_HEIGHT:       f32 = 50.0;   // cliff face / clifftop height above sea
const CLIFF_FLAT_DIST:    f32 = 150.0;  // flat clifftop width before the descent (blocks)
const CLIFF_DESCENT_DIST: f32 = 500.0;  // descent length blending clifftop → terrain (blocks)
const CLIFFTOP_NOISE:     f32 = 10.0;   // ± blocks of roll on the clifftop (vs. pancake flat)

// Finite-difference baseline (blocks) for estimating ‖∇c‖. MUST be large: the
// continent field carries domain-warp + coastline + ~40-block jaggedness noise.
// A small step measures that local noise (spiky gradient → warped cliff patches);
// a large step averages it out and captures only the smooth continental slope. It
// also sets the low-pass width for the smoothed `c` used in the distance estimate.
const GRAD_EPS: f32 = 320.0;

const CLIFF_FREQ:      f32 = 1.0 / 1_300.0;
const CLIFF_SLOW_FREQ: f32 = 1.0 / 12_000.0;
const CLIFF_SEED:      f32 = 31.7;
// Designation = smoothstep(THRESHOLD, THRESHOLD + BLEND_W, v). The blend band is
// the LATERAL (along-coast) fade of the cliff. A wide band keeps the sides from
// pinching off too fast. The upper edge (THRESHOLD + BLEND_W = 0.68) is held
// constant so the full-strength cliff cores stay where they were.
const CLIFF_THRESHOLD: f32 = 0.50;
const CLIFF_BLEND_W:   f32 = 0.18;

// ── Grid interpolation ────────────────────────────────────────────────────────

const GRID_STRIDE: usize = 4;
const GRID_DIM:    usize = CHUNK_SIZE / GRID_STRIDE + 1;

// ── Public API ────────────────────────────────────────────────────────────────

pub struct TerrainColumn {
    pub surface_y: usize,
    pub is_ocean:  bool,
    /// Blocks the cliff pass raised this column above natural terrain (0 = no cliff).
    pub cliff_lift: f32,
}

/// Full breakdown of the cliff pass for a single column — for debug visualisation.
pub struct CliffDebug {
    pub terrain_h:  f32,  // natural combined map (continent + noise), before cliffs
    pub final_h:    f32,  // after the cliff pass
    pub cliff_n:    f32,  // cliff designation 0..1 (the "mark")
    pub dist:       f32,  // distance-to-coast estimate (blocks) driving the spread
    pub cliff_lift: f32,  // final_h - terrain_h (the spread + descent)
    pub is_ocean:   bool, // natural terrain below sea level
}

pub fn sample_chunk_heights(
    defs:        &[ContinentDef; 9],
    origin_x:    i32,
    origin_z:    i32,
    out_surface: &mut [u16],
    out_ocean:   &mut [bool],
) -> usize {
    let mut c_grid      = [0.0f32; GRID_DIM * GRID_DIM];
    let mut dist_grid   = [0.0f32; GRID_DIM * GRID_DIM];
    let mut n_grid      = [0.0f32; GRID_DIM * GRID_DIM];
    let mut cliff_grid  = [0.0f32; GRID_DIM * GRID_DIM];
    let mut detail_grid = [0.0f32; GRID_DIM * GRID_DIM];

    for gz in 0..GRID_DIM {
        for gx in 0..GRID_DIM {
            let wx = origin_x + (gx * GRID_STRIDE) as i32;
            let wz = origin_z + (gz * GRID_STRIDE) as i32;
            let gi = gx + gz * GRID_DIM;
            let (fx, fz) = (wx as f32, wz as f32);
            let (c, dist, cliff_n) = coast_fields(defs, fx, fz);
            c_grid[gi]      = c;
            dist_grid[gi]   = dist;
            cliff_grid[gi]  = cliff_n;
            n_grid[gi]      = simplex2d(fx * NOISE_FREQ, fz * NOISE_FREQ + NOISE_SEED);
            detail_grid[gi] = detail_noise(fx, fz);
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

            let c       = bilerp(&c_grid,      gx0, gz0, gx1, gz1, tx, tz);
            let dist    = bilerp(&dist_grid,   gx0, gz0, gx1, gz1, tx, tz);
            let n       = bilerp(&n_grid,      gx0, gz0, gx1, gz1, tx, tz);
            let cliff_n = bilerp(&cliff_grid,  gx0, gz0, gx1, gz1, tx, tz);
            let detail  = bilerp(&detail_grid, gx0, gz0, gx1, gz1, tx, tz);

            let idx = x + z * CHUNK_SIZE;
            let (sy, ocean, _lift) = column(c, dist, n, cliff_n, detail);
            out_surface[idx] = sy as u16;
            out_ocean[idx]   = ocean;
            if !ocean && sy > max_y { max_y = sy; }
        }
    }

    max_y
}

pub fn sample_column(defs: &[ContinentDef; 9], wx: i32, wz: i32) -> TerrainColumn {
    let (fx, fz) = (wx as f32, wz as f32);
    let (c, dist, cliff_n) = coast_fields(defs, fx, fz);
    let n = simplex2d(fx * NOISE_FREQ, fz * NOISE_FREQ + NOISE_SEED);
    let detail = detail_noise(fx, fz);
    let (surface_y, is_ocean, cliff_lift) = column(c, dist, n, cliff_n, detail);
    TerrainColumn { surface_y, is_ocean, cliff_lift }
}

// ── Internals ─────────────────────────────────────────────────────────────────

fn column(c: f32, dist: f32, noise: f32, cliff_n: f32, detail: f32) -> (usize, bool, f32) {
    let (terrain_h, h) = terrain_and_cliff(c, dist, noise, cliff_n, detail);
    let cliff_lift = (h - terrain_h).max(0.0); // how much the cliff raised this column
    let is_ocean   = h < SEA_LEVEL as f32;
    let sy = (h.round() as usize).clamp(1, CHUNK_HEIGHT - 1);
    (sy, is_ocean, cliff_lift)
}

/// Single source of truth for the terrain + cliff math.
/// `cliff_n` is the designation sampled at the nearest COAST point (see coast_fields),
/// so the cliff spreads inland the full `dist` without being re-gated by inland noise.
/// Returns (natural terrain height, height after cliff pass).
fn terrain_and_cliff(c: f32, dist: f32, noise: f32, cliff_n: f32, detail: f32) -> (f32, f32) {
    // Pass 1 — natural terrain height (combined map).
    let noise_scale = lerp(LAND_NOISE, 1.0, (c / 0.05).clamp(0.0, 1.0));
    let terrain_h   = SEA_LEVEL as f32 - c * C_SCALE + noise * NOISE_AMP * noise_scale;

    // Pass 2 — cliffs: a uniform flat clifftop that blends DOWN to natural terrain
    // over a fixed descent distance, so the inland join is seamless by construction.
    let h = if terrain_h >= SEA_LEVEL as f32 && cliff_n > 0.0 {
        // Multi-octave detail so the clifftop reads as varied terrain, not a flat
        // dome. Plus a touch of the base noise so it ties into the surrounding land.
        let clifftop = SEA_LEVEL as f32 + CLIFF_HEIGHT
            + detail * CLIFFTOP_NOISE
            + noise * (CLIFFTOP_NOISE * 0.4);
        // 0 across the flat top, eases to 1 over the descent.
        let blend = smoothstep(CLIFF_FLAT_DIST, CLIFF_FLAT_DIST + CLIFF_DESCENT_DIST, dist);
        // Descend from the flat clifftop to the ACTUAL terrain height — lands exactly
        // on terrain (no seam) and fades terrain detail back in gradually.
        let cliff_target = lerp(clifftop, terrain_h, blend);
        // Only ever raise terrain, so naturally high coasts are preserved.
        let cliff_h = cliff_target.max(terrain_h);
        // cliff_n is the lateral (along-coast) fade.
        lerp(terrain_h, cliff_h, cliff_n)
    } else {
        terrain_h
    };

    (terrain_h, h)
}

/// Returns (continentalness, distance-to-coast in blocks, cliff designation at the
/// nearest coast).
///
/// The cliff designation is sampled NOT at this column but at the nearest point on
/// the coastline, found by stepping `dist` blocks along the (normalised) continent
/// gradient — which points perpendicular to the coast, toward the ocean (rising c).
/// This decouples the cliff's existence from the inland designation noise, so a
/// designated coast stretch produces a cliff that spreads inland by the full reach.
fn coast_fields(defs: &[ContinentDef; 9], fx: f32, fz: f32) -> (f32, f32, f32) {
    let c = continentalness_defs(defs, fx, fz); // raw — drives terrain height

    // Sample c at ±GRAD_EPS. These four taps give BOTH the gradient AND a low-passed
    // value of c. Using the smoothed value for the distance keeps `dist` monotonic
    // inland: the raw field wiggles (domain warp + coastline noise), and dividing a
    // wiggly value by a smooth gradient makes `dist` dip in spots far from any coast
    // — those dips fall under the cliff reach and surface as flat-top cliff blobs in
    // the middle of the landmass.
    let cxp = continentalness_defs(defs, fx + GRAD_EPS, fz);
    let cxm = continentalness_defs(defs, fx - GRAD_EPS, fz);
    let czp = continentalness_defs(defs, fx, fz + GRAD_EPS);
    let czm = continentalness_defs(defs, fx, fz - GRAD_EPS);
    let c_smooth = (cxp + cxm + czp + czm) * 0.25;

    // Central-difference gradient vector (raw, i.e. scaled by 2·GRAD_EPS).
    let raw_x = cxp - cxm;
    let raw_z = czp - czm;
    let raw_mag = (raw_x * raw_x + raw_z * raw_z).sqrt().max(1e-9);

    // Per-block gradient magnitude → approximate distance-to-coast (Eikonal).
    let grad = raw_mag / (2.0 * GRAD_EPS);
    let dist = (-c_smooth).max(0.0) / grad.max(1e-9);

    // Step `dist` along the unit gradient (+grad ⇒ rising c ⇒ toward the ocean) to
    // land on the nearest coast, and sample the cliff designation THERE.
    let coast_x = fx + (raw_x / raw_mag) * dist;
    let coast_z = fz + (raw_z / raw_mag) * dist;
    let cliff_n = cliff_noise_at(coast_x, coast_z);

    (c, dist, cliff_n)
}

/// Debug sampler exposing the full cliff-pass breakdown for one column.
pub fn sample_cliff_debug(defs: &[ContinentDef; 9], wx: i32, wz: i32) -> CliffDebug {
    let (fx, fz) = (wx as f32, wz as f32);
    let (c, dist, cliff_n) = coast_fields(defs, fx, fz);
    let n = simplex2d(fx * NOISE_FREQ, fz * NOISE_FREQ + NOISE_SEED);
    let detail = detail_noise(fx, fz);
    let (terrain_h, final_h) = terrain_and_cliff(c, dist, n, cliff_n, detail);
    CliffDebug {
        terrain_h,
        final_h,
        cliff_n,
        dist,
        cliff_lift: (final_h - terrain_h).max(0.0),
        is_ocean: terrain_h < SEA_LEVEL as f32,
    }
}

/// Multi-octave fBm for clifftop character (roughly [-1, 1]).
fn detail_noise(fx: f32, fz: f32) -> f32 {
    let n1 = simplex2d(fx * DETAIL_FREQ,        fz * DETAIL_FREQ        + DETAIL_SEED);
    let n2 = simplex2d(fx * DETAIL_FREQ * 2.2,  fz * DETAIL_FREQ * 2.2  + DETAIL_SEED + 5.1);
    let n3 = simplex2d(fx * DETAIL_FREQ * 4.5,  fz * DETAIL_FREQ * 4.5  + DETAIL_SEED + 9.7);
    (n1 * 0.55 + n2 * 0.30 + n3 * 0.15).clamp(-1.0, 1.0)
}

fn cliff_noise_at(fx: f32, fz: f32) -> f32 {
    let n1 = simplex2d(fx * CLIFF_FREQ,      fz * CLIFF_FREQ      + CLIFF_SEED);
    let n2 = simplex2d(fx * CLIFF_SLOW_FREQ, fz * CLIFF_SLOW_FREQ + CLIFF_SEED + 7.3);
    let v  = (n1 * 0.70 + n2 * 0.30) * 0.5 + 0.5;
    smoothstep(CLIFF_THRESHOLD, CLIFF_THRESHOLD + CLIFF_BLEND_W, v)
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
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
