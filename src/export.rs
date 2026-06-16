use rayon::prelude::*;

use crate::world::{
    continents::{build_defs, SEA_LEVEL},
    terrain::{sample_cliff_debug, sample_column, CliffDebug},
};

// ── Map settings ──────────────────────────────────────────────────────────────

const CENTER_X: i32 = 0;
const CENTER_Z: i32 = 0;

// Continent overview: wide area, low resolution.
const OVERVIEW_PX:     u32 = 1024;
const OVERVIEW_EXTENT: i32 = 12_000; // world-space radius shown
const OVERVIEW_SCALE:  i32 = (OVERVIEW_EXTENT * 2) / OVERVIEW_PX as i32; // blocks per pixel

// Grayscale heightmap: 2048×2048 blocks at 1 block/px (high resolution).
const HEIGHTMAP_PX:    u32 = 2048;
const HEIGHTMAP_SCALE: i32 = 1; // 1 block per pixel

// Cliff-pass debug: a region big enough to show several coastlines but detailed
// enough to read the inland descent (~12k blocks across).
const CLIFF_DBG_PX:    u32 = 2048;
const CLIFF_DBG_SCALE: i32 = 6; // blocks per pixel
const CLIFF_DIST_BAND: f32 = 100.0; // contour spacing (blocks) on the distance map

// Cliff red highlight: lift (blocks the cliff pass raised terrain) is mapped to
// red intensity over this range, so faint slope tails stay subtle and cliff faces
// read as solid red.
const CLIFF_RED_LO: f32 = 1.5;
const CLIFF_RED_HI: f32 = 6.0;

// ── Types ─────────────────────────────────────────────────────────────────────

struct Sample {
    cliff:     f32, // actual cliff lift in blocks (from the generated terrain)
    surface_y: usize,
    is_ocean:  bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn export_cliff_maps() {
    // Continent overview + cliff overlay.
    println!(
        "[export] overview {OVERVIEW_PX}×{OVERVIEW_PX} px @ {OVERVIEW_SCALE} blocks/px"
    );
    let overview = render_region(OVERVIEW_PX, OVERVIEW_SCALE, CENTER_X, CENTER_Z);
    save_rgb("map_world.png",  OVERVIEW_PX, &overview, |s| terrain_color(s.surface_y, s.is_ocean));
    save_rgb("map_cliffs.png", OVERVIEW_PX, &overview, |s| {
        let base = terrain_color(s.surface_y, s.is_ocean);
        blend_red(base, cliff_red_t(s.cliff))
    });

    // High-res grayscale heightmap with red cliff highlight.
    println!(
        "[export] heightmap {HEIGHTMAP_PX}×{HEIGHTMAP_PX} px @ {HEIGHTMAP_SCALE} block/px"
    );
    let hm = render_region(HEIGHTMAP_PX, HEIGHTMAP_SCALE, CENTER_X, CENTER_Z);
    let (lo, hi) = hm.iter().fold((usize::MAX, 0usize), |(lo, hi), s| {
        (lo.min(s.surface_y), hi.max(s.surface_y))
    });
    println!("[export]   height range {lo}..{hi}");
    save_rgb("map_heightmap.png", HEIGHTMAP_PX, &hm, |s| {
        let g = grayscale(s.surface_y, lo, hi);
        blend_red(g, cliff_red_t(s.cliff))
    });

    export_cliff_debug();

    println!(
        "[export] done → map_world.png  map_cliffs.png  map_heightmap.png  \
         cliff_lift.png  cliff_dist.png  cliff_profile.png"
    );
}

// ── Cliff-pass debug ───────────────────────────────────────────────────────────
//
// Isolates the cliff pass so the "mark → spread → slow descent" can be inspected:
//   cliff_lift.png    — how much the cliff raised each column (the actual result)
//   cliff_dist.png    — the distance-to-coast field that drives the spread, drawn
//                       with contour bands so you can see if it's smooth or warped
//   cliff_profile.png — a vertical cross-section through the middle row showing the
//                       natural terrain (gray) vs the final surface (red) so the
//                       face + descent shape is directly visible in side view

fn export_cliff_debug() {
    let px = CLIFF_DBG_PX;
    let n  = (px * px) as usize;
    let w  = px as i32;
    println!("[export] cliff debug {px}×{px} px @ {CLIFF_DBG_SCALE} blocks/px");

    let dbg: Vec<CliffDebug> = (0..n)
        .into_par_iter()
        .map(|i| {
            let cx = (i % px as usize) as i32;
            let cz = (i / px as usize) as i32;
            let wx = CENTER_X + (cx - w / 2) * CLIFF_DBG_SCALE;
            let wz = CENTER_Z + (cz - w / 2) * CLIFF_DBG_SCALE;
            sample_cliff_debug(&build_defs(wx, wz), wx, wz)
        })
        .collect();

    // 1. Cliff lift heatmap — the spread + descent of the cliff pass.
    save_rgb("cliff_lift.png", px, &dbg, |d| {
        if d.is_ocean { return [18, 28, 55]; }       // ocean
        if d.cliff_lift < 0.25 { return [38, 40, 44]; } // plain land, no cliff
        heat(d.cliff_lift / 50.0)                      // lift relative to face height
    });

    // 2. Distance-to-coast field with contour bands — verifies the SDF is clean.
    save_rgb("cliff_dist.png", px, &dbg, |d| {
        if d.is_ocean { return [18, 28, 55]; }
        let band = ((d.dist / CLIFF_DIST_BAND) as i32) % 2 == 0;
        let t    = (d.dist / 1500.0).clamp(0.0, 1.0);
        let v    = (255.0 * (1.0 - t)) as u8;          // bright at coast → dark inland
        if band { [v, v, v] } else { [v / 2, v / 2, (v as u16 * 3 / 4) as u8] }
    });

    // 3. Side-view profile of the middle row: natural terrain vs final surface.
    save_cliff_profile("cliff_profile.png", &dbg, px);
}

/// Render the centre row as a side-on height profile so the cliff shape is visible.
fn save_cliff_profile(path: &str, dbg: &[CliffDebug], px: u32) {
    let w = px as usize;
    let h = px as usize;
    let row = h / 2;
    let mut buf = vec![0u8; w * h * 3];

    // Vertical scale: map a height window around sea level onto the image.
    let y_lo = SEA_LEVEL as f32 - 80.0;
    let y_hi = SEA_LEVEL as f32 + 120.0;
    let to_py = |height: f32| -> usize {
        let t = ((height - y_lo) / (y_hi - y_lo)).clamp(0.0, 1.0);
        ((1.0 - t) * (h - 1) as f32) as usize // higher terrain → higher on image
    };

    let put = |buf: &mut [u8], x: usize, y: usize, c: [u8; 3]| {
        let i = (y * w + x) * 3;
        buf[i] = c[0]; buf[i + 1] = c[1]; buf[i + 2] = c[2];
    };

    // Sea level reference line.
    let sea_py = to_py(SEA_LEVEL as f32);
    for x in 0..w { put(&mut buf, x, sea_py, [40, 60, 90]); }

    for x in 0..w {
        let d = &dbg[row * w + x];
        // natural terrain in gray, final (with cliff) in red over the top.
        let ty = to_py(d.terrain_h);
        let fy = to_py(d.final_h);
        // fill columns down to give a solid silhouette
        for y in fy..h { put(&mut buf, x, y, [70, 30, 30]); }
        for y in ty..h { put(&mut buf, x, y, [55, 55, 60]); }
        put(&mut buf, x, ty.min(h - 1), [150, 150, 160]); // natural surface line
        put(&mut buf, x, fy.min(h - 1), [230, 70, 70]);    // final surface line
    }

    match image::RgbImage::from_raw(px, px, buf) {
        Some(img) => match img.save(path) {
            Ok(_)  => println!("[export]   wrote {path}"),
            Err(e) => eprintln!("[export]   {path} FAILED: {e}"),
        },
        None => eprintln!("[export]   {path} FAILED: buffer size mismatch"),
    }
}

/// Simple dark→red→yellow→white heat ramp for t in 0..1.
fn heat(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        // dark → red
        let u = t / 0.5;
        [lerp_u8(40, 220, u), lerp_u8(40, 40, u), lerp_u8(48, 40, u)]
    } else {
        // red → yellow → white
        let u = (t - 0.5) / 0.5;
        [lerp_u8(220, 255, u), lerp_u8(40, 245, u), lerp_u8(40, 210, u)]
    }
}

// ── Region sampling (two-pass) ─────────────────────────────────────────────────

fn render_region(image_px: u32, scale: i32, cx: i32, cz: i32) -> Vec<Sample> {
    let n = (image_px * image_px) as usize;
    let w = image_px as i32;

    (0..n)
        .into_par_iter()
        .map(|i| {
            let px = (i % image_px as usize) as i32;
            let pz = (i / image_px as usize) as i32;
            let wx = cx + (px - w / 2) * scale;
            let wz = cz + (pz - w / 2) * scale;

            // sample_column returns the full combined surface (continent + noise +
            // cliffs) plus the actual cliff lift used for the red highlight.
            let col = sample_column(&build_defs(wx, wz), wx, wz);
            Sample {
                cliff:     col.cliff_lift,
                surface_y: col.surface_y,
                is_ocean:  col.is_ocean,
            }
        })
        .collect()
}

// ── Colour helpers ────────────────────────────────────────────────────────────

/// Map actual cliff lift (blocks) to red intensity 0..1.
fn cliff_red_t(lift: f32) -> f32 {
    smoothstep(CLIFF_RED_LO, CLIFF_RED_HI, lift)
}

fn grayscale(sy: usize, lo: usize, hi: usize) -> [u8; 3] {
    let t = (sy.saturating_sub(lo)) as f32 / (hi - lo).max(1) as f32;
    let v = (t.clamp(0.0, 1.0) * 255.0).round() as u8;
    [v, v, v]
}

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
        0..=5     => [215, 205, 135],
        6..=40    => [ 95, 155,  65],
        41..=100  => [ 85, 120,  55],
        101..=180 => [105, 100,  80],
        _         => [225, 225, 225],
    }
}

fn blend_red(base: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        lerp_u8(base[0], 220, t),
        lerp_u8(base[1],  30, t),
        lerp_u8(base[2],  30, t),
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

fn save_rgb<T, F>(path: &str, width: u32, data: &[T], f: F)
where
    F: Fn(&T) -> [u8; 3] + Send + Sync,
    T: Sync,
{
    let buf: Vec<u8> = data.iter().flat_map(|d| f(d)).collect();
    match image::RgbImage::from_raw(width, width, buf) {
        Some(img) => match img.save(path) {
            Ok(_)  => println!("[export]   wrote {path}"),
            Err(e) => eprintln!("[export]   {path} FAILED: {e}"),
        },
        None => eprintln!("[export]   {path} FAILED: buffer size mismatch"),
    }
}
