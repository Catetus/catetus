# catetus-tileset

Octree LOD **tileset encoder** for Catetus — the streaming-first multi-tile
output that closes the gap with SuperSplat's CDN-streamed LOD octree.

Takes one `SplatCloud` and emits a spatial octree where each node carries
progressively-finer tile payloads, described by two interoperable manifests:

- **`lod-meta.json`** — byte-shape-compatible with SuperSplat
  (`{lodLevels, environment, filenames, tree:{bound, lods:[{file,count}], children}}`).
- **`tileset.json`** — 3D Tiles 1.1 (`{asset, geometricError, root:{boundingVolume,
  geometricError, refine, content, children}}`) for Cesium / the Catetus viewer.

```rust
use catetus_tileset::{plan_tileset, write_tileset, SfTileCodec, TilesetConfig};

let plan = plan_tileset(&cloud, &TilesetConfig::default())?;
let bytes = write_tileset(&plan, &SfTileCodec, std::path::Path::new("out/"))?;
// out/lod-meta.json, out/tileset.json, out/tiles/*.sftile
```

Tile payload encoding is pluggable via the `TilePayloadCodec` trait; the MVP
ships `SfTileCodec` (a minimal dependency-free `.sftile` binary). SF GLB / SOG
codecs plug in behind the same trait — see **STATUS.md** for what's done and the
roadmap.
