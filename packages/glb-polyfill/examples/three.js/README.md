# three.js example

Decode an Catetus GLB with `@catetus/glb-polyfill` and ingest it
as a `THREE.BufferGeometry`. Renders it as `THREE.Points` so the example
runs against unmodified `three` from npm — Three.js core does not ship a
Gaussian splat shader, so a "real" splat preview means dropping the
`BufferGeometry` into a community splat renderer
([`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D),
[`gsplat.js`](https://github.com/huggingface/gsplat.js), etc.). The data
prep (positions, rotations, scales, opacity, DC color) is what this
example demonstrates.

## Run

```sh
cp /path/to/your/scene.glb scene.glb
cp /path/to/your/scene.glb.shpal scene.glb.shpal   # optional
cp /path/to/your/scene.glb.v5tail scene.glb.v5tail # optional

npx esbuild main.ts --bundle --format=esm --external:three --outfile=main.js
python -m http.server 8080
# open http://localhost:8080
```

`three` is loaded from an importmap in `index.html` so the bundle stays
small; swap the importmap pin for your project's preferred channel.

## Wiring into a real splat shader

The geometry returned by `load()` exposes the conventional attribute
names that most splat-shader forks consume:

| attribute       | length     | semantics                       |
| --------------- | ---------- | ------------------------------- |
| `position`      | count × 3  | XYZ                             |
| `color`         | count × 3  | baked sRGB DC (preview only)    |
| `splatScale`    | count × 3  | linear per-axis scale           |
| `splatRotation` | count × 4  | XYZW quaternion                 |
| `splatOpacity`  | count × 1  | linear opacity in [0, 1]        |

Hand that geometry to your renderer's splat material instead of
`PointsMaterial` to get a full Gaussian splat draw.
