use rayon::prelude::*;

use crate::world::{
    continents::{build_defs, ContinentDef, SEA_LEVEL},
    noise::simplex2d,
    terrain::sample_column,
};

// ── Map settings ──────────────────────────────────────────────────────────────

const CENTER_X:    i32 = 2048;
const CENTER_Z:    i32 = 2048;
const HALF_EXTENT: i32 = 1500;
const IMAGE_PX:    u32 = 1024;
const SCALE:       i32 = (HALF_EXTENT * 2) / IMAGE_PX as i32;

// Land pixels within this many pixels of an ocean pixel are "coastal".
// At ~11 blocks/px, depth=2 ≈ 22 blocks from the shoreline.
const COASTAL_DEPTH: i32 = 2;

// ── Cliff arc-noise parameters ────────────────────────────────────────────────

// 2D world-space noise blobs — section length = blob_size * cliff_fraction.
// No tangent projection: avoids the curvature artifact where s oscillates on
// wiggly coasts and creates tiny blotches.
const CLIFF_BLOB_FREQ:  f32 = 1.0 / 3600.0; // blob size ≈ 3600 blocks → sections ~1600 blocks
const CLIFF_SLOW_FREQ:  f32 = 1.0 / 15000.0; // slow modulation → occasional long breaks
const CLIFF_ARC_SEED:   f32 = 31.7;
const CLIFF_THRESHOLD:  f32 = 0.55;          // ~45% cliff, ~55% break
const CLIFF_BLEND:      f32 = 0.10;

// ── Two-pass sample types ─────────────────────────────────────────────────────

struct RawSample {
    cliff_noise: f32,   // arc noise result (0..1); no coast-proximity mask yet
    surface_y:   usize,
    is_ocean:    bool,
}

struct Sample {
    cliff:     f32,     // final cliff value: arc noise if coastal, else 0
    surface_y: usize,
    is_ocean:  bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn export_cliff_maps() {
    let n = (IMAGE_PX * IMAGE_PX) as usize;
    println!(
        "[export] {IMAGE_PX}×{IMAGE_PX} px, {SCALE} blocks/px, ±{HALF_EXTENT} blocks from ({CENTER_X},{CENTER_Z})"
    );

    // Pass 1 — terrain heights + arc noise per pixel (fully parallel).
    let raw: Vec<RawSample> = (0..n)
        .into_par_iter()
        .map(|i| {
            let px = (i % IMAGE_PX as usize) as i32;
            let pz = (i / IMAGE_PX as usize) as i32;
            let wx = CENTER_X + (px - IMAGE_PX as i32 / 2) * SCALE;
            let wz = CENTER_Z + (pz - IMAGE_PX as i32 / 2) * SCALE;
            let (fx, fz) = (wx as f32, wz as f32);

            let defs        = build_defs(wx, wz);
            let col         = sample_column(&defs, wx, wz);
            let cliff_noise = arc_noise(&defs, fx, fz);

            RawSample { cliff_noise, surface_y: col.surface_y, is_ocean: col.is_ocean }
        })
        .collect();

    // Pass 2 — cliff mask: apply arc noise only where land is adjacent to ocean.
    // This aligns the cliff band with the rendered terrain edge, not a continentalness
    // proxy that can be hundreds of blocks off in bays / irregular coastlines.
    let rows: Vec<Sample> = (0..n)
        .into_par_iter()
        .map(|i| {
            let s = &raw[i];
            let cliff = if !s.is_ocean && is_coastal(&raw, i) {
                s.cliff_noise
            } else {
                0.0
            };
            Sample { cliff, surface_y: s.surface_y, is_ocean: s.is_ocean }
        })
        .collect();

    save_rgb("map_continent.png", &rows, |s| terrain_color(s.surface_y, s.is_ocean));
    save_rgb("map_cliffs.png",    &rows, |s| {
        let base = terrain_color(s.surface_y, s.is_ocean);
        if s.is_ocean { base } else { blend_red(base, s.cliff) }
    });

    println!("[export] Done — map_continent.png  map_cliffs.png");
}

// ── Second-pass coastal test ──────────────────────────────────────────────────

fn is_coastal(raw: &[RawSample], i: usize) -> bool {
    let w  = IMAGE_PX as i32;
    let px = (i % IMAGE_PX as usize) as i32;
    let pz = (i / IMAGE_PX as usize) as i32;
    for dz in -COASTAL_DEPTH..=COASTAL_DEPTH {
        for dx in -COASTAL_DEPTH..=COASTAL_DEPTH {
            let nx = (px + dx).clamp(0, w - 1);
            let nz = (pz + dz).clamp(0, w - 1);
            if raw[(nx + nz * w) as usize].is_ocean {
                return true;
            }
        }
    }
    false
}

// ── Arc noise (tangent-projected) ─────────────────────────────────────────────

fn arc_noise(_defs: &[ContinentDef; 9], fx: f32, fz: f32) -> f32 {
    // Two 2D octaves in world space. No tangent projection — blob size maps
    // directly to section length regardless of how the coast curves.
    let n1 = simplex2d(fx * CLIFF_BLOB_FREQ,  fz * CLIFF_BLOB_FREQ  + CLIFF_ARC_SEED);
    let n2 = simplex2d(fx * CLIFF_SLOW_FREQ,  fz * CLIFF_SLOW_FREQ  + CLIFF_ARC_SEED + 7.3);
    let noise01 = (n1 * 0.65 + n2 * 0.35) * 0.5 + 0.5;
    smoothstep(CLIFF_THRESHOLD, CLIFF_THRESHOLD + CLIFF_BLEND, noise01)
}

// ── Colour helpers ────────────────────────────────────────────────────────────

fn terrain_color(sy: usize, is_ocean: bool) -> [u8; 3] {
    if is_ocean {
        let depth = SEA_LEVEL.saturating_sub(sy);
        return match depth {
            0..=4   => [120, 180, 230],
            5..=14  => [ 55, 120, 195],
            15..=29 => [ 30,  80, 160],
            _       => [ 12,  40, 110],
        };
    }
    let above = sy.saturating_sub(SEA_LEVEL);
    match above {
        0..=2   => [215, 205, 135],
        3..=20  => [ 95, 155,  65],
        21..=55 => [ 85, 120,  55],
        56..=90 => [105, 100,  80],
        _       => [225, 225, 225],
    }
}

fn blend_red(base: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        lerp_u8(base[0], 220, t),
        lerp_u8(base[1],  35, t),
        lerp_u8(base[2],  35, t),
    ]
}

#[inline] fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

#[inline] fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── PNG writer ────────────────────────────────────────────────────────────────

fn save_rgb<T, F>(path: &str, data: &[T], f: F)
where
    F: Fn(&T) -> [u8; 3] + Send + Sync,
    T: Sync,
{
    let buf: Vec<u8> = data.iter().flat_map(|d| f(d)).collect();
    match image::RgbImage::from_raw(IMAGE_PX, IMAGE_PX, buf) {
        Some(img) => match img.save(path) {
            Ok(_)  => println!("[export]   wrote {path}"),
            Err(e) => eprintln!("[export]   {path} FAILED: {e}"),
        },
        None => eprintln!("[export]   {path} FAILED: buffer size mismatch"),
    }
}
