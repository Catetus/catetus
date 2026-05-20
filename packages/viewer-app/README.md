# @catetus/viewer-app

Web-deployable interactive 3D Gaussian Splatting viewer. Loads:

| Format                  | Extension          | Loader file                                | SH-rest |
| ----------------------- | ------------------ | ------------------------------------------ | ------- |
| Inria 3DGS PLY          | `.ply`             | `src/loaders/ply.ts`                       | yes (`f_rest_0..44`) |
| antimatter15 splat      | `.splat`           | `src/loaders/splat.ts`                     | no (format doesn't carry it) |
| PlayCanvas SOG V2       | `.sog`             | `src/loaders/sog.ts`                       | yes (`shN_centroids` + `shN_labels`) |
| Catetus GLB + sidecar | `.glb` + `.glb.shpal` | `src/loaders/sf-glb.ts` (via `@catetus/glb-polyfill`) | yes (palette codebook) |

All loaders normalize SH-rest to a single splat-major / coef-major /
channel-major layout (`shRest[i*coefCount*3 + k*3 + c]`); the renderer
evaluates degree-3 SH bands in the vertex shader for view-dependent color.

## Local dev

```sh
pnpm -C packages/viewer-app install     # one-time
pnpm -C packages/viewer-app dev         # serves on http://localhost:5173
```

Then either:

- Drop a `.ply` / `.splat` / `.sog` / `.glb` (+ its `.glb.shpal`) on the canvas, or
- Append `?src=https://your-cdn/scene.ply` to the URL — sibling `.shpal` is
  auto-fetched for SF GLBs.

## Controls

| Input          | Action                                               |
| -------------- | ---------------------------------------------------- |
| Left-drag      | Orbit around the scene origin                        |
| Right-drag (or shift+drag) | Pan                                      |
| Wheel          | Zoom (dolly)                                         |
| 1-finger touch | Orbit                                                |
| 2-finger touch | Pan + pinch-zoom                                     |
| `W` `A` `S` `D` | Pan / dolly                                         |
| `Q` `E`        | Roll left / right                                    |
| `R`            | Reset to default view                                |
| `F`            | Frame all (fit current scene)                        |

All motion is inertia-damped (~90 ms time constant). No snap-jumps.

## Build

```sh
pnpm -C packages/viewer-app build
```

The deployable build lands in `packages/viewer-app/dist/`.

## Deploy to Cloudflare Pages

The simplest path — drag-and-drop the `dist/` directory:

```sh
# One-time wrangler install (or use the dashboard):
npm install -g wrangler
wrangler pages deploy packages/viewer-app/dist --project-name catetus-viewer
```

Or via the dashboard:
1. Sign in to https://dash.cloudflare.com → Workers & Pages → Create → Pages
   → Upload assets.
2. Drop `packages/viewer-app/dist/` and click Deploy. Done.

Cloudflare Pages enforces a 25 MiB per-file limit on free tier, which is fine
for the bundle itself; large scenes should be hosted on R2 / Vercel Blob and
loaded via `?src=`.

### Vercel (alternative)

```sh
cd packages/viewer-app
vercel deploy --prebuilt --prod
```

(after `pnpm build` produces `dist/`; configure `dist` as the output directory
in the Vercel project settings).

### GitHub Pages (alternative)

Push `dist/` to a `gh-pages` branch or use any static-host action — there is
no backend.

## Adding V5.2 sidecar support (task #109 handoff)

Once `SF_v5_tail_residual` lands in `@catetus/glb-polyfill`, plumb the
`.v5tail` bytes through `loaders/sf-glb.ts → opts.sidecars` and call the
residual apply pass after `decodeSFExtensions(...)`. The renderer needs no
other changes — residuals write through the same {DC, scale, opacity} channels.

## Architecture notes

- **Renderer (`src/renderer.ts`)** — a single instanced-quad pipeline. Per
  splat we:
    1. project the 3D world-space covariance to a 2D screen-space ellipse
       via the projection Jacobian;
    2. size an instanced quad to ±3σ along the major axis;
    3. evaluate degree-3 SH for view-dependent color in the vertex shader
       (DC bake + SH-rest contribution clamped to [0,1] per channel);
    4. evaluate the Gaussian and premultiply alpha in the fragment shader.
  Sorting is back-to-front by view-space depth, recomputed when the camera
  moves more than ~4° / ~12 % distance from its last sort pose. Works to
  ~1.5 M splats on a discrete GPU.

- **SH-rest (view-dependent color)** — when a scene carries SH-rest
  coefficients (PLY `f_rest_*`, SOG `shN_*`, SF GLB `.shpal`), they are
  uploaded to a single `RGBA16F` 2D texture (12 texels per splat = 48 fp16
  capacity, holds full SH3 = 45 coefs with 3-float padding). The vertex
  shader fetches 12 texels per splat per quad-vertex (only 6 invocations
  per splat), evaluates the standard Inria 3DGS SH bands against
  `viewDir = normalize(splatCenter - cameraPos)`, and adds the result on
  top of `0.5 + SH_C0 * f_dc` before clamping to display range. .splat
  assets (which don't carry SH-rest) skip this path. The canonical layout
  in `SplatScene.shRest` is splat-major then coef-major then channel-major:
  `shRest[i * coefCount * 3 + k * 3 + c]` — see `src/splat-scene.ts`.

- **Controls (`src/controls.ts`)** — pure Pointer-Events + KeyboardEvent.
  No third-party orbit lib so the bundle stays small and you can read the
  whole interaction surface in one file.

- **Loaders (`src/loaders/*`)** — every loader normalizes to the
  `SplatScene` shape in `src/splat-scene.ts`. New formats only need a new
  `loaders/<fmt>.ts` + a branch in `dispatch.ts`.

- **Deps** — `fflate` (ZIP/Deflate for SOG), `fzstd` (Zstd for SF GLB), the
  in-workspace `@catetus/glb-polyfill`. No GPU-renderer libs (Three /
  Babylon / PlayCanvas) so the bundle is small and the GL state is fully
  inspectable.
