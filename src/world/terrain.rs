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

const NOISE_FREQ1:    f32 = 1.0 / 1_500.0;  // broad rolling terrain
const NOISE_FREQ2:    f32 = 1.0 /   500.0;  // terrain detail
const NOISE_FREQ_HARD: f32 = 1.0 /  900.0;  // rock-hardness patches (~several hundred blocks wide)

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
// Using actual block height rather than continentalness keeps the boost aligned
// with the visible waterline regardless of the noise_01 value.
const CLIFF_COASTAL_WINDOW: f32 = 28.0;

// noise_01 range where cliff eligibility ramps from 0 → 1.
// Below CLIFF_HEIGHT_LO (low plains) → no cliff; above CLIFF_HEIGHT_HI → full cliff.
const CLIFF_HEIGHT_LO: f32 = 0.42;
const CLIFF_HEIGHT_HI: f32 = 0.68;

// Hardness range.  Below LO (soft rock / beaches) → no cliff.
const CLIFF_HARD_LO: f32 = 0.52;
const CLIFF_HARD_HI: f32 = 0.82;

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
    let mut c_grid    = [0.0f32; GRID_DIM * GRID_DIM];
    let mut n1_grid   = [0.0f32; GRID_DIM * GRID_DIM];
    let mut n2_grid   = [0.0f32; GRID_DIM * GRID_DIM];
    let mut hard_grid = [0.0f32; GRID_DIM * GRID_DIM];

    for gz in 0..GRID_DIM {
        for gx in 0..GRID_DIM {
            let wx = origin_x + (gx * GRID_STRIDE) as i32;
            let wz = origin_z + (gz * GRID_STRIDE) as i32;
            let (fx, fz) = (wx as f32, wz as f32);
            let gi = gx + gz * GRID_DIM;

            c_grid[gi]    = continentalness_defs(defs, fx, fz);
            n1_grid[gi]   = simplex2d(fx * NOISE_FREQ1,    fz * NOISE_FREQ1);
            n2_grid[gi]   = simplex2d(fx * NOISE_FREQ2,    fz * NOISE_FREQ2    + 17.3);
            hard_grid[gi] = simplex2d(fx * NOISE_FREQ_HARD + 53.1,
                                      fz * NOISE_FREQ_HARD + 71.7);
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

            let c        = bilerp(&c_grid,    gx0, gz0, gx1, gz1, tx, tz);
            let n1       = bilerp(&n1_grid,   gx0, gz0, gx1, gz1, tx, tz);
            let n2       = bilerp(&n2_grid,   gx0, gz0, gx1, gz1, tx, tz);
            let hardness = bilerp(&hard_grid, gx0, gz0, gx1, gz1, tx, tz) * 0.5 + 0.5;

            let idx = x + z * CHUNK_SIZE;
            let (sy, ocean) = surface_from_cv(c, n1, n2, hardness);
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
    let n1       = simplex2d(fx * NOISE_FREQ1, fz * NOISE_FREQ1);
    let n2       = simplex2d(fx * NOISE_FREQ2, fz * NOISE_FREQ2 + 17.3);
    let hardness = simplex2d(fx * NOISE_FREQ_HARD + 53.1,
                             fz * NOISE_FREQ_HARD + 71.7) * 0.5 + 0.5;
    let (surface_y, is_ocean) = surface_from_cv(c, n1, n2, hardness);
    TerrainColumn { surface_y, is_ocean }
}

// ── Internals ─────────────────────────────────────────────────────────────────

fn surface_from_cv(c: f32, n1: f32, n2: f32, hardness: f32) -> (usize, bool) {
    let noise_01 = (n1 * 0.7 + n2 * 0.3).clamp(-1.0, 1.0) * 0.5 + 0.5;

    let ocean_h = lerp(OCEAN_FLOOR_MIN, OCEAN_FLOOR_MAX, noise_01);
    let land_h  = lerp(LAND_HEIGHT_MIN, LAND_HEIGHT_MAX, noise_01);

    // Base height: unchanged original blend.
    let cf      = cont_factor(c);
    let base_sy = lerp(ocean_h, land_h, cf);

    // ── Coastal cliff boost ───────────────────────────────────────────────────
    //
    // coast_strip: 1 at the visible waterline, 0 beyond CLIFF_COASTAL_WINDOW blocks
    // above sea level.  Keying on actual height-above-sea keeps the boost aligned
    // with where water meets land rather than with the c=0 continentalness boundary
    // (which can be noticeably inland for tall-terrain coasts).
    let height_above_sea = base_sy - SEA_LEVEL as f32;
    let coast_strip = if height_above_sea > 0.0 {
        smoothstep(CLIFF_COASTAL_WINDOW, 0.0, height_above_sea)
    } else {
        0.0
    };

    // Both hard rock AND already-tall terrain are required.  Either alone (soft
    // lowland coast, or hard rock on a flat plain) stays as a normal beach.
    let height_cond = smoothstep(CLIFF_HEIGHT_LO, CLIFF_HEIGHT_HI, noise_01);
    let hard_cond   = smoothstep(CLIFF_HARD_LO,   CLIFF_HARD_HI,   hardness);

    let cliff_mask = coast_strip * height_cond * hard_cond;

    // Boost lifts base_sy toward the full land height for this column.
    // At cliff_mask=1 (right at the waterline, hard rock, tall terrain):
    //   sy = base_sy + (land_h - base_sy) = land_h  → full cliff-top height.
    // At cliff_mask=0 (inland, soft rock, or flat coast):
    //   sy = base_sy                                 → unmodified terrain.
    // The ramp from 0→1 as you approach the coast is the hillside climbing to
    // the cliff edge; the sheer drop is the ocean-side of that land column.
    let sy = (base_sy + (land_h - base_sy) * cliff_mask).round().max(1.0) as usize;
    (sy, sy < SEA_LEVEL)
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
