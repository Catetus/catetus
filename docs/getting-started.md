# Getting started with Catetus

This guide assumes you have a working Rust toolchain (`rustup` + stable) and Node.js 20+ with `pnpm`.

## 1. Build the CLI

```bash
cargo build --release -p catetus-cli
export PATH="$PWD/target/release:$PATH"
```

## 2. Analyze a splat

```bash
catetus analyze fixtures/tiny/basic_binary.ply
```

This prints a deterministic JSON report. To save it:

```bash
catetus analyze fixtures/tiny/basic_binary.ply --pretty > analyze.json
```

## 3. Optimize for the web

```bash
catetus optimize fixtures/tiny/basic_binary.ply \
  --preset web-mobile \
  --out out/scene.gltf
```

Output layout:

```
out/
  scene.gltf
  buffers/
    root.bin
    lod0_0001.bin
    ...
  reports/
    optimize.json
```

## 4. Preview locally

```bash
catetus preview out/scene.gltf
# opens http://localhost:5170 with the WebGPU viewer
```

## 5. Generate a visual diff

```bash
catetus diff fixtures/tiny/basic_binary.ply out/scene.gltf \
  --threshold 0.03 \
  --out reports/diff/
open reports/diff/diff.html
```

The CLI delegates to a small Node helper (`tests/visual/scripts/diff-cli.mjs`)
which drives a headless Chromium via [`playwright-core`](https://playwright.dev)
to render 8 deterministic orbit frames per asset and pixel-compare them. To
enable the full rendering path, install the workspace dependencies once:

```bash
pnpm -F catetus-visual install
pnpm -F catetus-visual exec playwright install chromium
```

If `playwright-core` is missing, `catetus diff` still produces a valid
`diff.json` (with `status: "degraded"`) and a `diff.html` placeholder so
downstream tooling never has to special-case a missing report.

You can override the helper location with `CATETUS_DIFF_HELPER=/path/to/diff-cli.mjs`
(useful when running an installed binary outside the source tree). Set
`CATETUS_INCLUDE_TIMESTAMPS=1` to embed a `generatedAt` field — the default
output is deterministic so snapshot tests stay stable.

## 6. Run a parity matrix

```bash
pnpm -F catetus-visual run test
open tests/visual/report/index.html
```

## 7. Use the viewer SDK

```ts
import { CatetusViewer } from '@catetus/viewer';

const viewer = new CatetusViewer({
  canvas: document.getElementById('canvas') as HTMLCanvasElement,
  src: '/assets/scene.gltf',
  budget: 'web-mobile',
});

viewer.on('firstRender', () => console.log('ready'));
await viewer.load();
```
