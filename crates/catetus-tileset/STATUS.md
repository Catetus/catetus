# catetus-tileset — STATUS

**Created:** 2026-05-29
**Owner:** streaming workstream (STREAM1)
**One-liner:** Octree LOD tileset encoder — turns one `SplatCloud` into a
streaming-first multi-tile LOD octree with two interoperable manifests
(SuperSplat `lod-meta.json` + 3D Tiles `tileset.json`).

## Why this crate exists (strategic)

SuperSplat's competitive moat is **streaming, not compression ratio**. The
Koriyama head-to-head ([[koriyama-strategic-learnings]],
`splatforge-private/research/competitive/koriyama_head_to_head_2026-05-29.md`)
showed they ship a **7-level LOD octree of ~87 SOG tiles** from CloudFront; the
viewer fetches root + visible tiles first so a 492 MB scene first-paints in
~1 s. Catetus shipped a single all-or-nothing GLB blob. `catetus-tileset`
closes that gap.

## SuperSplat `lod-meta.json` format (studied, verbatim)

Fetched `d28zzqy0iyovbz.cloudfront.net/b11e45d1/v1/lod-meta.json` and parsed it
(saved analysis; 84.8 KB):

```json
{
  "lodLevels": 7,
  "environment": "env/meta.json",
  "filenames": ["2_0/meta.json", "0_0/meta.json", ... 88 entries ...],
  "tree": {
    "bound": { "min": [x,y,z], "max": [x,y,z] },
    "children": [
      { "bound": {...},
        "lods": [ {"file":6,"count":7456}, {"file":33,"count":31960}, {"file":62,"count":124648} ],
        "children": [...] }
    ]
  }
}
```

Measured facts about the real tree:
- `lodLevels: 7`; the tree is a **sparse octree of depth 6 (7 levels), 73 nodes**.
- **Every node carries exactly 3 `{file, count}` LOD entries**, coarse→fine.
- `filenames` is a flat array of **88** SOG tile `meta.json` paths; `lods[].file`
  indexes into it. Tile addressing is `<col>_<row>` (e.g. `6_0`, `0_40`).
- Bounds are AABBs (`min`/`max`), not spheres. Plus a separate `environment` SOG.
- LOD-switch criterion is **viewer-side** (not in the manifest): the viewer
  picks a node's LOD by projected size / distance and refines into children.

`catetus-tileset` reproduces this shape exactly (see `tests/supersplat_compat.rs`,
which deserializes a verbatim Koriyama fragment with our own types).

## Inventory of prior art (reused / superseded)

- `catetus-optimize/src/tileset.rs` — a **stub**: `plan_tileset` is `todo!()`,
  `TileSplitConfig::default` is malformed (`pub fn` in a trait impl), contains a
  `morton_dummy`. It defined good 3D-Tiles manifest *types* and a `decimate`
  helper but never produced a tileset. `catetus-tileset` supersedes it with a
  working implementation (richer: real octree, importance-weighted LODs, the
  SuperSplat manifest, a payload-codec trait). The optimize stub + its
  `tests/geospatial_tileset.rs` + `fixtures/geospatial-sample/` can later be
  deleted or repointed at this crate.
- `catetus-lodge` (893 lines) — single-file LODGE hierarchical builder.
  Complementary: lodge does in-place LOD within one file; this crate does
  multi-tile spatial partitioning for streaming. Worth a follow-up to share
  decimation/importance code.
- `catetus-ply/src/progressive.rs` — progressive PLY reader; relevant to the
  large-scene streaming-ingest roadmap item, not the encoder.

## What the MVP does (this version)

- **Real octree** (`octree.rs`): depth-limited spatial subdivision at box
  centers, empty octants pruned (sparse, like SuperSplat). Configurable
  `max_depth` (default 6) and `max_splats_per_leaf` (default 50k). Handles
  coincident/degenerate points (forces a leaf — no stack blow-up).
  Conservation invariant: every splat lands in exactly one leaf (tested).
- **Per-node LODs** (`plan.rs`): each node gets `lods_per_node` (default 3)
  representative sub-clouds, coarse→fine, via **importance-weighted decimation**
  (importance = opacity × cube-root scale volume): keep the top-half by
  importance, then stride the remainder for spatial coverage. Finest LOD = the
  node's full splat set. Counts are monotonic coarse→fine (tested).
- **Two manifests** (`manifest.rs`): emits both `lod-meta.json` (SuperSplat
  shape, byte-compatible) and `tileset.json` (3D Tiles 1.1, `box` bounding
  volumes, `REPLACE` refine, per-node `content.uri` → finest LOD). Both describe
  the same octree.
- **Pluggable tile payload codec** (`codec.rs`): `TilePayloadCodec` trait so GLB
  / SOG plug in later. MVP ships `SfTileCodec` — a minimal, dependency-free,
  losslessly round-tripping `.sftile` binary — so the crate compiles and is
  fully testable **without** the GLB/SOG writers.
- **Disk writer** (`plan.rs::write_tileset`): writes `lod-meta.json`,
  `tileset.json`, and `tiles/<i>.<ext>`; patches manifests to the codec's real
  extension; returns total tile bytes.
- **Example** (`examples/build_tileset.rs`): builds a 64k-splat synthetic scene
  end-to-end with zero inputs.
- **Tests**: octree conservation/termination/subdivision; codec round-trip
  (DC-only, SH-rest, empty); plan multi-level + monotonic counts; full
  write→reparse→per-tile-decode→count-match; SuperSplat-fragment parse +
  our-output-reparses-as-LodMeta.

## What's left (roadmap)

### Near-term wiring (days)
1. **Real tile payloads.** Implement `TilePayloadCodec` for SF GLB
   (`catetus-gltf`) and SuperSplat-compatible SOG (`catetus-spz` / private
   `catetus-sog`). The trait boundary already exists; this is the highest-value
   next step (turns `.sftile` placeholders into viewer-loadable tiles).
2. **CLI integration.** `catetus optimize --target tileset scene.ply -o out/`
   in `catetus-cli` + PLY decode via `catetus-ply` into a `SplatCloud`. (PLY/GLB
   reader signatures were not re-read this session — confirm them first.)
3. **Replace the optimize stub.** Delete `catetus-optimize/src/tileset.rs` (or
   re-export from here) and repoint `tests/geospatial_tileset.rs`.

### Quality of LOD (1–2 wks)
4. **Screen-space-error selection in the viewer.** Wire the WebGPU viewer
   (splatforge.dev/viewer) to fetch root-first, then refine when
   `sse = (geometricError * viewportHeight) / (distance * 2 tan(fov/2))` exceeds
   a threshold (~16 px). MVP geometric error = node bbox diagonal; refine into
   children. Add a distance-ring fallback.
5. **Perceptual decimation.** Replace opacity×volume importance with a render-
   aware metric (camera-coverage from CAMCOV-1 [[camcov-validated]], or a small
   render-loss probe). Coarse LODs should preserve silhouettes/contrast.
6. **Tile size balancing / merging.** Cap tile byte size; merge tiny sibling
   leaves; split oversized leaves. Match SuperSplat's ~3 LODs/node, balanced
   tile counts.

### Streaming/CDN (2–3 wks)
7. **Byte-range packing.** Optionally pack all tiles into one file with a
   range-request index (CloudFront-style) to cut request count.
8. **First-paint benchmark.** Honest time-to-first-paint vs SuperSplat on a
   400+ MB scene (Koriyama). We do not know our own number yet.
9. **Large-scene ingest.** 48M-splat scenes (2.5 GB PLY) need streaming/chunked
   decode — pairs with `catetus-ply/src/progressive.rs`. Octree build is
   currently in-memory; make it external-memory or tile-streaming for >50M.

### Interop
10. **`lod-meta.json` transcoder.** Emit exactly SuperSplat's `<col>_<row>`
    filename scheme + `env/meta.json` environment tile so a SuperSplat viewer
    can load Catetus output unmodified, and vice-versa.

## Verification (VERIFIED GREEN 2026-05-29)

- `cargo test -p catetus-tileset` → **16 tests pass, 0 fail** (octree 4, codec
  4, plan 5, supersplat_compat 2, doctest 1).
- `cargo check -p catetus-tileset --all-targets` → clean, no warnings.
- `cargo check --workspace` → **clean, 0 errors** (this crate integrates into
  the workspace; only `crates/catetus-tileset/*` and the workspace `Cargo.toml`
  member/dep lists were added).
- `cargo run -p catetus-tileset --example build_tileset -- /tmp/ts-out` →
  64,000-splat synthetic scene → **3-level octree, 219 tiles**, 18.7 MB of tile
  payloads + `lod-meta.json` (46.6 KB, `lodLevels:3`) + `tileset.json`, all on
  disk and re-parsed/decoded by the tests. Emitted `tileset.json` root carries a
  12-float `box` bounding volume, `geometricError` (67.5), `REPLACE` refine, and
  `content.uri` → its finest tile (`tiles/2.sftile`).

### Important finding: SuperSplat's real `lods` shape

The full Koriyama `lod-meta.json` (84.8 KB, parsed) stores each node's `lods`
as a JSON **object keyed `"0".."6"`** with **`{file, offset, count}`** entries —
NOT an array, and with a per-LOD `offset`. The `offset` means multiple nodes'
LODs are packed into a **shared tile file** at byte ranges (range-request
streaming). Our `LodMeta` currently uses `lods: Vec<{file,count}>` (one tile
file per LOD, no offset) — same information, flatter, one-file-per-tile. A
byte-exact SuperSplat transcoder (object-keyed `lods` + `offset` + shared tile
files) is roadmap item #10 (§Interop). The tree shape, `bound{min,max}`,
`lodLevels`, flat `filenames`, and `environment` are already matched exactly.

## File map

```
crates/catetus-tileset/
  Cargo.toml
  STATUS.md                      (this file)
  README.md
  src/
    lib.rs                       crate docs + re-exports
    octree.rs                    sparse octree over splat positions
    manifest.rs                  LodMeta (SuperSplat) + TilesetManifest (3D Tiles)
    codec.rs                     TilePayloadCodec trait + SfTileCodec (.sftile v1)
    plan.rs                      plan_tileset + write_tileset + LOD assignment
  examples/build_tileset.rs      synthetic end-to-end demo
  tests/supersplat_compat.rs     real SuperSplat manifest parse + our-output reparse
```
