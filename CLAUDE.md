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

### Chunk Pipeline (CPU, parallel via rayon global pool)

World chunk loading is two-stage:

1. **Stage 1 — Generation**: Each chunk's blocks are generated procedurally using noise-based terrain, then boundary solidity masks (`ChunkFaceData`) are extracted and sent back to the main thread via channel.
2. **Stage 2 — Meshing**: Runs once a chunk *and all 4 XZ neighbors* have completed Stage 1. Sends vertex/index data back to the main thread.

Both stages use `rayon::spawn` on the global thread pool (no custom pool). Chunk gen is throttled via a `spawn_queue` to avoid flooding rayon with mesh work before gen completes.

Finished meshes are queued in `upload_queue` and uploaded to the GPU arena in batches of up to 64 chunks per frame via a single `vkQueueSubmit`. The arena is a DEVICE_LOCAL buffer with a first-fit free-list sub-allocator; staging is persistent HOST_VISIBLE mapped memory.

Key files: [src/world/world.rs](src/world/world.rs), [src/world/chunk.rs](src/world/chunk.rs), [src/world/neighbor.rs](src/world/neighbor.rs)

### LOD System

Render distance and detail are controlled by three bands in `settings.toml`:

- **lod0** — full mesh (`mesh_chunk`), with per-vertex AO. Chunks within Chebyshev distance `lod0` from the camera.
- **lod1** — surface-only mesh (`mesh_chunk_surface`), AO=1.0. Chunks in the band `lod0..lod0+lod1`.
- **lod2** — water surface only (`mesh_chunk_water`). Chunks in `lod0+lod1..total`.

LOD0 is only uploaded to the GPU arena for chunks within the lod0 band. Distant chunks only get LOD1 uploaded; LOD0 is queued on demand when the camera moves close. This keeps arena usage manageable (see GPU Memory Budget section).

### Greedy Meshing

[src/meshing/greedy.rs](src/meshing/greedy.rs) runs per-direction greedy rectangle merging over each chunk. Faces with uniform per-corner AO are merged freely; faces with non-uniform AO stay as 1×1 quads to preserve the AO gradient. Neighbor boundary masks are used to cull faces between chunks where both sides are solid.

### Vulkan Rendering

- **VulkanContext** ([src/render/context.rs](src/render/context.rs)): Instance, physical device selection, logical device, queues, surface.
- **Renderer** ([src/render/renderer.rs](src/render/renderer.rs)): Double-buffered frame submission (`MAX_FRAMES_IN_FLIGHT = 2`) with fence/semaphore sync. Single render pass renders directly to swapchain images.
- **Pipeline** ([src/render/pipeline.rs](src/render/pipeline.rs)): Single graphics pipeline with swapchain-format color + D32_SFLOAT_S8_UINT depth. MVP pushed via push constants. Texture array bound via descriptor set.
- **Mesh** ([src/render/mesh.rs](src/render/mesh.rs)): `Vertex` is 36-byte `#[repr(C)] + Pod` (pos + normal + UV + ao). GPU buffers are deferred-destroyed with a 3-frame delay.

### Camera & Input

[src/app.rs](src/app.rs) handles the event loop: mouse grab on click, ESC to release, WASD + Space/Shift for movement, scroll to adjust speed (1–200 units/sec). [src/camera.rs](src/camera.rs) builds the MVP matrix with Y-flipped projection for Vulkan.

## Key Constants

- Render distance: configured via `settings.toml` (`lod0`, `lod1`, `lod2`). Defaults: `lod0=20, lod1=60, lod2=0` → 80-chunk render distance, ~26,000 chunks total.
- Chunk size: 32×32×256 blocks (XZ × Y), flat-indexed as `x + z·32 + y·32²`
- Terrain: noise-based heightmap, `SEA_LEVEL = 120`. Stone below surface, dirt at surface, water fills to sea level.

## Shaders

`shaders/voxel.vert` — MVP transform via push constant 4×4 matrix.  
`shaders/voxel.frag` — Directional light at (0.6, 1.0, 0.4), brightness = `(0.35 + diffuse * 0.65) * pow(ao, 1.8)`. Texture layer encoded in UV.y.

Shader SPIR-V is compiled into the build output directory by `build.rs` and loaded at runtime.

## When Adding Features — GPU Memory Budget

Any change that increases per-chunk vertex data (new vertex attributes, reduced greedy merging, new geometry passes) must update the arena sizing in `World::new` in [src/world/world.rs](src/world/world.rs):

```rust
let vb_cap = (chunks * 60_000).next_power_of_two().clamp(512 << 20, 2048 << 20);
let ib_cap = (chunks * 10_000).next_power_of_two().clamp(128 << 20,  512 << 20);
```

The `60_000` and `10_000` per-chunk byte estimates must cover the actual mesh output. **If these are too small, the arena silently grows at runtime by calling `queue_wait_idle`, which freezes all rendering and chunk uploading for several seconds** — visible as the world abruptly stopping mid-load. This has happened: when AO was added, vertex size grew from 16 → 36 bytes and greedy merging decreased, so the old 5 KB/chunk estimate was ~32× too small, causing 3 arena grows on startup.

Rules of thumb:
- `sizeof(Vertex)` is currently 36 bytes. Multiply by expected average verts/chunk for your worst-case LOD.
- LOD0 is only uploaded for chunks within `lod1_dist` (default 20 chunks). LOD1 is uploaded for all chunks. Size accordingly.
- The staging buffers (`STAGING_VB_CAP` / `STAGING_IB_CAP`) must fit one full batch of 64 chunks. If they overflow, uploads are re-queued — check the console for `[upload] staging overflow` messages.
- Arena grows are logged as `[arena_vb] grow:` or `[arena] vb alloc failed`. If you see these, the per-chunk estimate needs to increase.
