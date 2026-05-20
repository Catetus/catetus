# raw-canvas example

A bare WebGL2 viewer that draws each splat as a 2-pixel point. No Gaussian
rasterization, no SH evaluation, no third-party rendering library — the
file exists to prove that `decodeSFExtensions` + `decodeV5TailBytes` +
`applyV5TailToScene` is everything you need to go from `.glb` bytes to
GPU-uploadable typed arrays.

## Run

```sh
# from this directory
cp /path/to/your/scene.glb scene.glb           # required
cp /path/to/your/scene.glb.shpal scene.glb.shpal   # optional (SH palette)
cp /path/to/your/scene.glb.v5tail scene.glb.v5tail # optional (V5.2 residual)

# bundle once (or use any other bundler you like)
npx esbuild main.ts --bundle --format=esm --outfile=main.js
# rewrite the script tag in index.html to load main.js instead of main.ts,
# or use a TS-aware dev server like vite.

python -m http.server 8080
# open http://localhost:8080
```

## What the code does

1. `fetch()` the GLB bytes and split into JSON + BIN chunks (30-line
   helper, inlined).
2. `decodeSFExtensions(json, bin, { 'scene.glb.shpal': palBytes })` →
   normalized splat arrays.
3. If the scene carries `SF_log_quant_attrs`, exp/sigmoid the scales and
   opacities once into linear space.
4. If a `.glb.v5tail` is present, `decodeV5TailBytes` + `applyV5TailToScene`
   for the +6.54 dB residual layer.
5. Bake `SH_C0 * dc + 0.5` into per-splat RGB, upload positions + colors
   to two `ARRAY_BUFFER`s, draw as `GL_POINTS`.

Total: ~150 lines of TS, no runtime deps beyond `@catetus/glb-polyfill`
and `fzstd`.
