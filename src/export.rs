use rayon::prelude::*;

use crate::world::{
    continents::{build_defs, continentalness_defs, SEA_LEVEL},
    terrain::{
        base_elevation_raw, humidity_raw, rock_hardness_raw, sample_column, temperature_raw,
    },
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// World-space centre of every exported map.
pub const CENTER_X: i32 = 2048;
pub const CENTER_Z: i32 = 2048;

const HALF_EXTENT: i32 = 4096;     // ±4096 blocks from centre
const IMAGE_PX:    u32  = 1024;    // pixels per side
const SCALE: i32 = (HALF_EXTENT * 2) / IMAGE_PX as i32; // blocks per pixel = 8

// ── Per-pixel sample ──────────────────────────────────────────────────────────

struct Sample {
    c:         f32,
    h:         f32,
    e:         f32,
    t:         f32,
    u:         f32,
    surface_y: usize,
    is_ocean:  bool,
    biome:     u8,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn export_noise_maps() {
    let n = IMAGE_PX as usize * IMAGE_PX as usize;
    println!(
        "[export] Sampling {n} points ({IMAGE_PX}×{IMAGE_PX}, {SCALE} blocks/px, ±{HALF_EXTENT} blocks from ({CENTER_X},{CENTER_Z}))…"
    );

    let rows: Vec<Sample> = (0..n)
        .into_par_iter()
        .map(|i| {
            let px = (i % IMAGE_PX as usize) as i32;
            let pz = (i / IMAGE_PX as usize) as i32;
            let wx = CENTER_X + (px - IMAGE_PX as i32 / 2) * SCALE;
            let wz = CENTER_Z + (pz - IMAGE_PX as i32 / 2) * SCALE;
            let (fx, fz) = (wx as f32, wz as f32);

            let defs = build_defs(wx, wz);
            let col  = sample_column(&defs, wx, wz);
            Sample {
                c:         continentalness_defs(&defs, fx, fz),
                h:         rock_hardness_raw(fx, fz).clamp(0.0, 1.0),
                e:         base_elevation_raw(fx, fz),
                t:         temperature_raw(fx, fz),
                u:         humidity_raw(fx, fz),
                surface_y: col.surface_y,
                is_ocean:  col.is_ocean,
                biome:     col.biome,
            }
        })
        .collect();

    save_gray("noise_continentalness.png", &rows, |s| {
        ((-s.c * 0.6 + 0.5).clamp(0.0, 1.0) * 255.0) as u8
    });
    save_gray("noise_hardness.png", &rows, |s| {
        (s.h * 255.0) as u8
    });
    save_gray("noise_elevation.png", &rows, |s| {
        ((s.e * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8
    });
    save_gray("noise_temperature.png", &rows, |s| {
        ((s.t * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8
    });
    save_gray("noise_humidity.png", &rows, |s| {
        ((s.u * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8
    });
    save_gray("noise_surface_height.png", &rows, |s| {
        ((s.surface_y as f32 / 256.0) * 255.0) as u8
    });
    save_rgb("world_surface.png", &rows, |s| {
        terrain_color(s.surface_y, s.is_ocean, s.biome)
    });

    println!("[export] Done.");
}

// ── Colour mapping ────────────────────────────────────────────────────────────

fn terrain_color(sy: usize, is_ocean: bool, biome: u8) -> [u8; 3] {
    use crate::world::terrain::{BIOME_ALPINE, BIOME_DESERT, BIOME_TUNDRA};

    if is_ocean {
        let depth = SEA_LEVEL.saturating_sub(sy);
        return match depth {
            0..=4   => [120, 180, 230],
            5..=14  => [ 55, 120, 195],
            15..=29 => [ 30,  80, 160],
            _       => [ 12,  40, 110],
        };
    }

    // Biome-tinted land colouring
    let above = sy.saturating_sub(SEA_LEVEL);
    match biome {
        BIOME_DESERT => match above {
            0..=2   => [225, 210, 140],  // beach / low desert
            3..=30  => [210, 185,  95],  // sandy plains
            _       => [180, 155,  75],  // desert upland
        },
        BIOME_ALPINE => match above {
            0..=60  => [130, 120, 110],  // rocky mountain base
            61..=90 => [170, 165, 155],  // high rock
            _       => [230, 230, 230],  // snow peak
        },
        BIOME_TUNDRA => match above {
            0..=5   => [170, 185, 165],  // low tundra
            _       => [195, 200, 185],  // tundra upland
        },
        _ => match above {
            0..=2   => [215, 205, 135],  // beach / sand
            3..=20  => [ 95, 155,  65],  // coastal plain
            21..=55 => [ 85, 120,  55],  // hills
            56..=90 => [105, 100,  80],  // mountain base
            91..=110=> [155, 145, 130],  // high mountain
            _       => [225, 225, 225],  // snow peak
        },
    }
}

// ── PNG writers ───────────────────────────────────────────────────────────────

fn save_gray<T, F>(path: &str, data: &[T], f: F)
where
    F: Fn(&T) -> u8,
{
    let buf: Vec<u8> = data.iter().map(f).collect();
    match image::GrayImage::from_raw(IMAGE_PX, IMAGE_PX, buf) {
        Some(img) => match img.save(path) {
            Ok(_)  => println!("[export]   {path}"),
            Err(e) => eprintln!("[export]   {path} FAILED: {e}"),
        },
        None => eprintln!("[export]   {path} FAILED: buffer size mismatch"),
    }
}

fn save_rgb<T, F>(path: &str, data: &[T], f: F)
where
    F: Fn(&T) -> [u8; 3],
{
    let buf: Vec<u8> = data.iter().flat_map(|d| f(d)).collect();
    match image::RgbImage::from_raw(IMAGE_PX, IMAGE_PX, buf) {
        Some(img) => match img.save(path) {
            Ok(_)  => println!("[export]   {path}"),
            Err(e) => eprintln!("[export]   {path} FAILED: {e}"),
        },
        None => eprintln!("[export]   {path} FAILED: buffer size mismatch"),
    }
}
