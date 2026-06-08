use std::path::PathBuf;
use std::process::Command;

fn main() {
    let shaders = [
        ("shaders/voxel.vert", "voxel.vert.spv"),
        ("shaders/voxel.frag", "voxel.frag.spv"),
    ];

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    for (src, out) in &shaders {
        println!("cargo:rerun-if-changed={src}");
        let out_path = out_dir.join(out);
        let status = Command::new("glslc")
            .args([src, "-o", out_path.to_str().unwrap()])
            .status()
            .expect("glslc not found — install the Vulkan SDK and ensure glslc is on PATH");
        assert!(status.success(), "glslc failed to compile {src}");
    }
}
