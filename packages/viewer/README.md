# @splatforge/viewer

Browser viewer SDK for SplatForge optimized glTF / SPZ gaussian-splat assets.
Implements [SPEC-0008](../../specs/0008-viewer-sdk.md).

## Install

```bash
pnpm add @splatforge/viewer
```

No runtime dependencies. WebGPU types are dev-only.

## Quick start

```ts
import { SplatForgeViewer } from '@splatforge/viewer';

const viewer = new SplatForgeViewer({
  canvas: document.getElementById('canvas') as HTMLCanvasElement,
  src: '/assets/scene/scene.gltf',
  budget: 'web-mobile',
  renderer: 'auto',
  cameraPath: 'orbit-8',
});

viewer.on('firstRender', () => console.log('ready'));
viewer.on('chunkLoaded', ({ chunkIndex, byteLength }) => {
  console.log(`chunk ${chunkIndex}: ${byteLength}B`);
});
viewer.on('error', ({ code, message }) => console.error(code, message));

await viewer.load();
```

When you're done:

```ts
viewer.dispose();
```

## Events

| Event             | Payload                                  | Meaning                                          |
| ----------------- | ---------------------------------------- | ------------------------------------------------ |
| `loadStart`       | `{}`                                     | `load()` was called.                             |
| `manifestLoaded`  | `{ chunkCount }`                         | glTF JSON parsed.                                |
| `chunkLoaded`     | `{ chunkIndex, byteLength }`             | A streaming tile finished downloading.           |
| `firstRender`     | `{}`                                     | First frame submitted to the renderer.           |
| `qualityChanged`  | `{ level }`                              | LOD bias changed (driven by budget heuristics).  |
| `complete`        | `{}`                                     | All chunks loaded.                               |
| `warning`         | `{ code, message }`                      | Non-fatal (e.g. checksum unsupported).           |
| `error`           | `{ code, message }`                      | Fatal. `load()` rejects with `Error(code)`.      |

### Error codes

`manifest_not_found`, `manifest_invalid`, `chunk_not_found`,
`checksum_failed`, `renderer_unavailable`, `renderer_init_failed`, `unknown`.

## Renderer strategy

- **WebGPU** primary. Instanced single-quad per splat, alpha-blended back-to-
  front. Detected via `navigator.gpu.requestAdapter()`.
- **WebGL2** fallback. Same algorithm; GLSL ES 3.00 shaders.
- **Headless deterministic** mode (`deterministic: true`) uses a seeded RNG
  and the `orbit-8` camera path so visual-regression frames are reproducible.

## Browser support

| Browser          | Versions     | Backend |
| ---------------- | ------------ | ------- |
| Chrome / Edge    | 113+         | WebGPU  |
| Chrome / Edge    | 80 – 112     | WebGL2  |
| Firefox          | All current  | WebGL2  |
| Safari (macOS)   | 17+          | WebGL2  |
| Safari (iOS)     | 16+          | WebGL2  |
| Quest Browser    | Recent       | WebGL2  |
| visionOS Safari  | 1.0+         | WebGL2  |

WebGPU on Firefox / Safari is still behind a flag — the SDK falls back
automatically. Force a backend with `renderer: 'webgl2'` or `'webgpu'`.

## Preview shell

The package ships an HTML shell consumed by the CLI's `splatforge preview`
subcommand. It reads `?src=<url>` from the query string, mounts a
full-window canvas, and displays a small HUD with renderer / chunk state.

Resolved path: `@splatforge/viewer/preview-shell.html`.

## Scripts

```bash
pnpm --filter @splatforge/viewer build   # tsc -> dist/
pnpm --filter @splatforge/viewer lint    # tsc --noEmit
pnpm --filter @splatforge/viewer test    # vitest run
```

## License

Apache-2.0.
