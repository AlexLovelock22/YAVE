pub struct Settings {
    /// Width of the full-resolution band (Chebyshev distance 0..lod0).
    pub lod0: i32,
    /// Width of the half-resolution band (lod0..lod0+lod1).
    pub lod1: i32,
    /// Width of the quarter-resolution band (lod0+lod1..total).
    pub lod2: i32,
    /// Multiplier on the base terrain-noise frequency: > 1.0 = denser, busier
    /// terrain; < 1.0 = smoother, broader. Temporary global control until
    /// biome-specific noise exists.
    pub noise_density: f32,
}

impl Settings {
    pub fn render_distance(&self) -> i32 { self.lod0 + self.lod1 + self.lod2 }
    pub fn lod1_dist(&self)       -> i32 { self.lod0 }
    pub fn lod2_dist(&self)       -> i32 { self.lod0 + self.lod1 }
}

impl Default for Settings {
    fn default() -> Self {
        Self { lod0: 20, lod1: 60, lod2: 0, noise_density: 1.0 }
    }
}

/// Load `settings.toml` from the working directory.
/// If the file doesn't exist, create it with defaults and return them.
pub fn load() -> Settings {
    let defaults = Settings::default();

    let text = match std::fs::read_to_string("settings.toml") {
        Ok(t) => t,
        Err(_) => {
            let content = format!(
                "# YAVE render distance settings\n\
                 lod0 = {}\nlod1 = {}\nlod2 = {}\n\
                 \n\
                 # Terrain noise density: > 1.0 = denser/busier, < 1.0 = smoother.\n\
                 noise_density = {}\n",
                defaults.lod0, defaults.lod1, defaults.lod2, defaults.noise_density,
            );
            let _ = std::fs::write("settings.toml", content);
            println!("[settings] created settings.toml with defaults");
            crate::world::terrain::set_noise_density(defaults.noise_density);
            return defaults;
        }
    };

    let mut s = defaults;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() { continue; }
        if let Some((k, v)) = line.split_once('=') {
            let v = v.trim();
            match k.trim() {
                "lod0" => if let Ok(n) = v.parse() { s.lod0 = n },
                "lod1" => if let Ok(n) = v.parse() { s.lod1 = n },
                "lod2" => if let Ok(n) = v.parse() { s.lod2 = n },
                "noise_density" => if let Ok(n) = v.parse() { s.noise_density = n },
                other  => eprintln!("[settings] unknown key: {other}"),
            }
        }
    }

    println!(
        "[settings] lod0={} lod1={} lod2={}  render_distance={}  noise_density={}",
        s.lod0, s.lod1, s.lod2, s.render_distance(), s.noise_density
    );
    crate::world::terrain::set_noise_density(s.noise_density);
    s
}
