# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

YAVE is a Vulkan-based voxel engine written in Rust. It renders procedurally generated terrain using greedy mesh generation and a raw Vulkan rendering pipeline — conceptually similar to Minecraft's renderer.

## Build & Run

Requires the Vulkan SDK with `glslc` on PATH (for shader compilation). The `build.rs` script compiles `shaders/voxel.vert` and `shaders/voxel.frag` to SPIR-V automatically.

```bash
cargo run --release   # debug builds stutter due to chunk gen overhead
```

There are no tests in the codebase.

## Architecture

### Chunk Pipeline (CPU, parallel via rayon)

World chunk loading is two-stage:

1. **Stage 1 — Generation**: Each chunk's blocks are generated procedurally, then boundary solidity masks (`ChunkFaceData`) are extracted and sent back to the main thread via channel.
2. **Stage 2 — Meshing**: Runs once a chunk *and all 4 XZ neighbors* have completed Stage 1. Sends vertex/index data back to the main thread.

The combined GPU mesh is rebuilt every 30 frames during the loading phase, then continuously once all chunks are settled. The rayon pool is sized to `cores - 2` (min 1) to leave headroom for the render thread.

Key files: [src/world/world.rs](src/world/world.rs), [src/world/chunk.rs](src/world/chunk.rs), [src/world/neighbor.rs](src/world/neighbor.rs)

### Greedy Meshing

[src/meshing/greedy.rs](src/meshing/greedy.rs) runs per-direction greedy rectangle merging over each chunk. Full-height faces are merged into rectangles; non-full faces (e.g., slab sides) are emitted directly from geometry. Neighbor boundary masks are used to skip faces between chunks where both sides are solid.

### Vulkan Rendering

- **VulkanContext** ([src/render/context.rs](src/render/context.rs)): Instance, physical device selection, logical device, queues, surface.
- **Renderer** ([src/render/renderer.rs](src/render/renderer.rs)): Double-buffered frame submission (`MAX_FRAMES_IN_FLIGHT = 2`) with fence/semaphore sync.
- **Pipeline** ([src/render/pipeline.rs](src/render/pipeline.rs)): Single graphics pipeline with color + D32_SFLOAT depth attachment. MVP pushed via push constants — no descriptor sets.
- **Mesh** ([src/render/mesh.rs](src/render/mesh.rs)): `Vertex` is 16-byte `#[repr(C)] + Pod` (pos + normal + UV). GPU buffers are deferred-destroyed with a 3-frame delay.

### Camera & Input

[src/app.rs](src/app.rs) handles the event loop: mouse grab on click, ESC to release, WASD + Space/Shift for movement, scroll to adjust speed (1–200 units/sec). [src/camera.rs](src/camera.rs) builds the MVP matrix with Y-flipped projection for Vulkan.

## Key Constants

- `RD` in [src/app.rs](src/app.rs): render distance in chunks (default 4 → 9×9 chunk grid)
- Chunk size: 32×32×256 blocks (XZ × Y), flat-indexed as `x + z·32 + y·32²`
- Chunk generation: Dirt at Y < 48, Stone at Y ≥ 48

## Shaders

`shaders/voxel.vert` — MVP transform via push constant 4×4 matrix.  
`shaders/voxel.frag` — Directional light at (0.6, 1.0, 0.4) + 0.25 ambient, light blue base color.

Shader SPIR-V is compiled into the build output directory by `build.rs` and loaded at runtime.
