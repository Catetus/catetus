# Cesium ion quickstart for Catetus `--preset geospatial`

Catetus's `geospatial` preset emits a [Cesium 3D Tiles 1.1](https://github.com/CesiumGS/3d-tiles) tileset
with [`KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/pull/2454) per-tile content. Cesium ion
accepts that format directly as of the April 2026 "3D Gaussian Splats with LOD" release.

This guide gives a partner engineer at Bentley / Cesium a five-minute path from a `.ply` capture to a tiled
splat asset rendered in CesiumJS or Cesium for Unreal.

## 1. Produce the tileset locally

```bash
catetus optimize \
  --preset geospatial \
  --output-dir my-asset/ \
  capture.ply
```

This writes:

```
my-asset/
├── tileset.json          # 3D Tiles 1.1 hierarchy with KHR_gaussian_splatting tiles
├── lod0.glb              # full-resolution leaf
├── lod1.glb              # ~½ count
├── lod2.glb              # ~¼ count
├── lod3.glb              # ~⅛ count
└── optimize-report.json  # per-pass stats (size deltas, counts, notes)
```

Each tile is a self-contained binary glTF (`.glb`) carrying its own splats — Cesium ion only needs the
`tileset.json` and the GLBs Cesium ion will fetch lazily.

## 2. Validate locally (optional but recommended)

```bash
# Inspect any tile as a sanity check:
catetus inspect my-asset/lod0.glb

# Or run the 3D Tiles validator from the CesiumGS tools repo:
npx @cesium/3d-tiles-validator --tilesetFile my-asset/tileset.json
```

The validator should report zero errors and zero warnings. The `extensionsRequired` array contains
`KHR_gaussian_splatting`; the per-tile GLBs each declare the same extension with `extensionsRequired` and
`extensionsUsed`.

## 3. Upload to Cesium ion

Cesium ion accepts 3D Tiles tilesets as a single `.zip` upload, or via the REST API for automated pipelines.

### Web UI

1. Sign in at <https://cesium.com/ion/>.
2. **My Assets → Add data → 3D Tiles** (the menu changed in April 2026; pick the entry that mentions
   "3D Tiles 1.1 with Gaussian Splatting").
3. Drag-and-drop the entire `my-asset/` directory (or its `.zip`). Cesium ion auto-detects the
   tileset root from `tileset.json`.
4. Choose a geolocation: a fixed lat/lon/altitude origin, or "use existing georeference" if the splat
   capture's coordinate system was already ECEF.
5. Click **Create**. The asset moves to `COMPLETE` once ingestion finishes (typically <2 min for a
   ≤100 MB capture).

### REST API

```bash
curl -X POST https://api.cesium.com/v1/assets \
  -H "Authorization: Bearer $ION_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "Catetus geospatial sample",
        "description": "KHR_gaussian_splatting + 3D Tiles 1.1",
        "type": "3DTILES",
        "options": { "sourceType": "3D_TILES" }
      }'
```

Upload the directory contents to the returned S3 prefix, then `POST .../onComplete` per the
standard ion ingestion flow.

## 4. View in CesiumJS

```js
import { Viewer, Cesium3DTileset, Cesium3DTileStyle } from "cesium";

const viewer = new Viewer("cesiumContainer");
const tileset = await Cesium3DTileset.fromIonAssetId(<YOUR_ASSET_ID>);
viewer.scene.primitives.add(tileset);
await viewer.zoomTo(tileset);
```

Cesium handles the screen-space-error refinement automatically. The four LOD levels Catetus emits give
~4 stops of detail; tune the viewer's `maximumScreenSpaceError` (default `16`) if your scene needs more
aggressive culling.

## 5. View in Cesium for Unreal / Cesium for Unity

Identical workflow: drop the **Cesium 3D Tileset** actor into the level, point its `IonAssetID` at the new
asset, and the engine consumes the same `tileset.json`. The native runtime calls into Cesium's
[`gltfReader`](https://github.com/CesiumGS/cesium-native) which gained `KHR_gaussian_splatting` support in
the April 2026 release of `cesium-native 0.42`.

## 6. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Asset upload fails with "unknown extension" | ion account predates the April 2026 release. File a ticket; the extension is gated by feature flag for some pre-existing tenants. |
| Tiles render but appear flat/grey | Viewer is older than `CesiumJS 1.124`. Upgrade. |
| Refinement never picks finer LODs | Reduce `maximumScreenSpaceError` (more aggressive refinement) or check that the bounding boxes are correctly oriented. Catetus emits axis-aligned boxes in the local frame — if the capture's local frame is offset from ECEF, set the per-tileset `transform` in CesiumJS. |
| Coarsest LOD is too low quality at intro fly-in | Increase the LOD chain depth in `catetus-optimize/src/presets.rs` (`BuildLOD` levels). |

## 7. Sample tileset

A tiny committed-to-git reference tileset lives at
`crates/catetus-optimize/tests/fixtures/geospatial-sample/`. It is built deterministically by the
`geospatial_preset_emits_cesium_tileset` test and is safe to upload to a sandbox ion account for a
visual smoke test.
