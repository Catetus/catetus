# Streaming-tile viewer adapter

Consumes a Cesium 3D Tiles 1.1 `tileset.json` and renders only the tiles
that lie inside the camera frustum at the appropriate LOD, fetching the
rest on demand. This is the runtime counterpart to `--preset geospatial`
(see `crates/splatforge-optimize/src/tileset.rs`).

## Architecture

```
                      tileset.json
                           │
                           ▼
                  TilesetLoader (tileset_loader.ts)
                  • fetch + parse
                  • build TileNode tree (root, parent, depth)
                           │
                           ▼
   ┌───────────────────────────────────────────────────┐
   │ Per-frame loop (Viewer.runStreamingCameraPath)    │
   │                                                   │
   │  CameraPose                                       │
   │      │                                            │
   │      ▼                                            │
   │  extractFrustum (frustum.ts)                      │
   │      │                                            │
   │      ▼                                            │
   │  selectVisibleTiles (lod_selector.ts)             │
   │  • culls outside-frustum                          │
   │  • SSE formula vs. maximumScreenSpaceError        │
   │  • returns {render, fetch}                        │
   │      │                                            │
   │      ▼                                            │
   │  TileStreamer.fetchTile (tile_streamer.ts)        │
   │  • LRU cache (512 MB default)                     │
   │  • dedup + priority queue + concurrency cap       │
   │  • GLB → manifest                                 │
   │      │                                            │
   │      ▼                                            │
   │  Viewer.uploadTileToRenderer                      │
   │      │                                            │
   │      ▼                                            │
   │  WebGPURenderer.uploadChunk + renderFrame         │
   │  (composes with ComputeDecodePipeline + radix     │
   │   sort already in webgpu/index.ts)                │
   └───────────────────────────────────────────────────┘
```

## SSE thresholds

The selector uses the standard Cesium formula:

```
sse = geometricError * height / (distance * 2 * tan(fovY / 2))
```

A tile is refined into its children when `sse > maximumScreenSpaceError`.
`maximumScreenSpaceError` defaults to 16 (Cesium's default). The v2 plan
calls for:

| Budget preset      | maximumScreenSpaceError |
|---|---|
| `web-mobile`       | 8 (higher quality, more tile fetches) |
| `web-desktop`      | 16 (default) |
| `quest-browser`    | 12 |
| `visionos-preview` | 8 |

Pass it via `viewer.loadTileset(url, { maximumScreenSpaceError: N })`.

## Eviction policy

`TileStreamer` runs an LRU keyed by render-frame index:

1. Each rendered tile's `lastUsed` is bumped on every frame it appears in
   the `render` set.
2. When `residentBytes > maxBytes` (default 512 MB), the streamer drops
   the lowest `lastUsed` tile, repeating until under budget.
3. Tiles touched on the *current* frame are pinned — they can't be
   evicted mid-frame.

This is "render-frame LRU" rather than "wall-clock LRU" so the policy is
deterministic for a fixed camera path: identical runs evict in identical
order.

## Determinism

* The selector returns tiles in a stable preorder traversal.
* Plane extraction uses pure float math with no IEEE-edge branches.
* In deterministic mode the viewer awaits pending fetches before
  emitting `frameRendered`, so the resident set on every frame is
  exactly the tiles the SSE/frustum walk wanted (no race with the
  network).

The visual-regression spec in `tests/visual/tests/streaming-tileset.spec.ts`
asserts a ≤ 2 % per-pixel diff against a per-pose golden — that bound is
the float-precision noise floor of the rasterizer, not the streaming
adapter.

## Expected performance

Measured on the committed `geospatial-sample` fixture (4 LODs, ~450
splats total, ~30 KB on disk):

| Metric | Value |
|---|---|
| Cold-start time to first frame | ~2 ms (Node bench) |
| Sustained FPS (JS-only path, 60-frame orbit) | 10 000+ |
| Peak resident bytes | ~30 KB (well under 512 MB budget) |
| Cache evictions | 0 |

These numbers are **JS-pipeline only** — the GPU rasterizer is the
dominant cost for production-sized scenes. The streaming adapter's job is
to keep the resident set small enough that the rasterizer's per-tile
draw cost stays under frame budget; the bench validates that the
selector + streamer overhead is negligible (sub-millisecond).

Per the v2 plan, 30 GB scenes at 60 fps on mobile is achievable when:

* `maximumScreenSpaceError = 8` keeps the on-screen tile count bounded.
* `maxBytes = 256 MB` (mobile) or 512 MB (desktop) caps GPU memory.
* `useComputeDecode = true` routes splat decode through the WGSL shader,
  saving the CPU→GPU vertex-buffer upload at >1M splats.

## Mobile target tuning

For Quest Browser / iOS Safari:

```ts
await viewer.loadTileset(url, {
  maximumScreenSpaceError: 8,
  maxBytes: 256 * 1024 * 1024,
  prefetchLookahead: 0.3,    // extrude camera 30% of last-frame motion
  maxConcurrency: 4,         // tighter on mobile networks
});
```

`prefetchLookahead` enables research-queue #63: each frame the selector
runs a second time at `eye + lastMotion * lookahead`, and any
out-of-frustum-but-soon-to-be-visible tiles are fetched at low priority.

## Authoring a tileset

The producer side ships in `splatforge-optimize`:

```bash
splatforge optimize --preset geospatial input.ply -o /tmp/asset
```

Outputs a directory with `tileset.json` + `lod{0,1,2,3}.glb`. Each GLB is
a self-contained binary glTF with the `KHR_gaussian_splatting` extension
carrying that LOD's splats. The 12-float OBB bounding volume Cesium
expects is axis-aligned in the asset's local frame; Cesium ion picks up
the ECEF transform from upload-time geolocation.

See `crates/splatforge-optimize/src/tileset.rs` for the full producer
specification.

## Fallback path (WebGL2)

The streaming adapter is WebGPU-first. WebGL2 falls back to a single-LOD
root render — the per-frame compute upload churn is too expensive on
WebGL2 to maintain the v2 frame-time budget. Tilesets authored for
streaming still load on WebGL2; they just don't refine.
