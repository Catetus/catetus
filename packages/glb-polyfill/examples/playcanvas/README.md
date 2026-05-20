# PlayCanvas example

Decode an Catetus GLB with `@catetus/glb-polyfill` and render it
through PlayCanvas as a point-cloud `Mesh` + `MeshInstance`. Stays on
stable PlayCanvas APIs (`pc.Mesh`, `pc.StandardMaterial`,
`pc.PRIMITIVE_POINTS`) so the example works against the published
`playcanvas` package without touching engine-private splat internals. Once
your PlayCanvas version's `GSplatComponent` API stabilizes, the
positions/rotations/scales/opacity/DC arrays decoded here drop in
directly.

## Run

```sh
cp /path/to/your/scene.glb scene.glb
cp /path/to/your/scene.glb.shpal scene.glb.shpal   # optional
cp /path/to/your/scene.glb.v5tail scene.glb.v5tail # optional

npx esbuild main.ts --bundle --format=esm --external:playcanvas --outfile=main.js
python -m http.server 8080
# open http://localhost:8080
```

The `playcanvas` runtime is pulled from an importmap in `index.html`.

## Going beyond points

The decoded data carries everything PlayCanvas's splat pipeline needs:
`positions`, `rotations` (XYZW), `scales` (linear, per-axis),
`opacities` (linear, [0, 1]), `dcRaw` (raw SH DC), and optionally
`sh_rest`. Hand those buffers to whatever splat component matches your
engine version.
