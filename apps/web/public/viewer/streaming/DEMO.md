# Streaming tileset viewer — demo + verification

STREAM-3 wired the Catetus WebGPU viewer to load a **tileset** (octree of
tiles + manifest) and progressively fetch+render root-then-visible tiles,
SuperSplat-style. This is the viewer side; STREAM-1 (`catetus-tileset`) is the
encoder.

## What landed

- `streaming/sftile.js` — decoder for STREAM-1's `.sftile` v1 tile payload
  (magic `SFT1`), converting linear-scale + baked-RGB IR into the log-scale +
  raw-`f_dc` shape `splatSceneToSoaChunk` expects.
- `streaming/tile_streamer.js` — `actuallyFetch` now dispatches on tile magic:
  `.sftile` → `decodeSftile`, `glTF` → existing `decodeGlb`. So the same
  streamer serves STREAM-1 tiles now and STREAM-2 GLB tiles when they land.
- `streaming/runtime.js` — `TilesetRuntime`: owns a `StreamingTileset`, drives
  `.frame(camera)` each RAF, diffs the visible render-set, rebuilds the WebGPU
  renderer from resident tiles (root first), and records TTFP / tiles-over-time
  / bytes-over-time into `window.__tilesetMetrics`.
- `index.html` — new `?tileset=<url>` bootstrap (Mode C) alongside `?src=`,
  plus an optional `?sse=<px>` to tune the screen-space-error threshold. The
  HUD shows `drawn / resident / in-flight` and first-paint time.
- `streaming/smoke_sftile.mjs` — headless (no-GPU) test of the load-bearing
  data path: parse `tileset.json` → decode every `.sftile` → pack SoA, asserting
  manifest/count/byte-size invariants and root-first decodability.

## Generate a tileset (STREAM-1)

```sh
cargo run -p catetus-tileset --example build_tileset -- /tmp/ts-out
# writes /tmp/ts-out/{tileset.json, lod-meta.json, tiles/*.sftile}
cp -R /tmp/ts-out/. apps/web/public/fixtures/tileset-demo/
```

A pre-generated copy lives at `apps/web/public/fixtures/tileset-demo/`.

## Headless verification (no browser/GPU needed)

```sh
node apps/web/public/viewer/streaming/smoke_sftile.mjs
# -> parses the manifest, decodes the root first, then every tile, and asserts
#    splat counts match the manifest + SoA byte sizes are exact. Prints PASS.
```

## Browser demo (progressive load)

```sh
cd apps/web/public
python3 -m http.server 8080
# open:
#   http://localhost:8080/viewer/index.html?tileset=/fixtures/tileset-demo/tileset.json
```

Expected: the coarse root tile paints almost immediately (HUD shows a first-
paint time in ms), then finer tiles fill in as you orbit/zoom — nearby/large
nodes refine to higher LOD, off-frustum tiles are skipped, and the LRU evicts
tiles past the byte budget. Open the console and inspect:

```js
window.__tilesetMetrics
// { t0, firstPaintMs, timeline: [{t, tiles, bytes, inFlight, drawTiles}, …] }
```

`firstPaintMs` is the time-to-first-paint that feeds the TTF-1 benchmark;
`timeline` is the tiles-over-time / bytes-over-time series.

## Notes / coordination

- Tile format is owned by STREAM-2; the viewer dispatches on magic so it does
  not need to change when GLB tiles land — only `manifestFromGlb`/`decodeGlb`
  (already present) handle that branch.
- The renderer is rebuilt only when the visible tile set changes (camera
  crosses an SSE/frustum boundary or a tile finishes loading), not every frame.
