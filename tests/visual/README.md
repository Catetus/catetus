# splatforge-visual

Visual-diff (SPEC-0009) and viewer-parity (SPEC-0010) harness.

## Prerequisites

```bash
pnpm install
pnpm --filter @splatforge/viewer build       # produce packages/viewer/dist/
pnpm --filter splatforge-visual exec playwright install
```

WebGPU requires a recent Chromium and a swiftshader/vulkan capable host —
the `chrome-webgpu` Playwright project launches Chromium with
`--enable-unsafe-webgpu --use-vulkan=swiftshader`.

## Run the matrix

```bash
pnpm -F splatforge-visual test         # runs all 4 renderer projects
pnpm -F splatforge-visual run report   # builds report/index.html
```

Per-renderer outputs land in:

```
report/raw/<asset>/<project>/
  0001.png ... 0008.png
  metrics.json
  parity-cell.json
```

The aggregated matrix is at `report/parity.json` (schema:
`splatforge.parity/1`). The browseable HTML is `report/index.html`.

## Run a single project

```bash
pnpm -F splatforge-visual exec playwright test --project=chrome-webgpu
```

## Fixtures and goldens

Fixtures live under `fixtures/`:

```
fixtures/tiny/cube.gltf            # tiny known asset used by the smoke test
fixtures/golden/frames/<asset>/<project>/0001.png ... 0008.png
```

If no goldens exist for a given asset+project, the diff test does **not**
fail — it records a `golden-missing` annotation and saves the run under
`report/raw/<asset>/<project>/candidate/` for review. Promote the
candidate by copying it into `fixtures/golden/frames/`.

## Visual diff CLI

```bash
node scripts/diff-cli.mjs before.ply after.gltf \
  --out reports/diff/ \
  --threshold 0.03 \
  --camera-path orbit-8 \
  --frame-size 512x512 \
  --renderer webgpu
```

The CLI:

1. Converts any `.ply` input via `${SPLATFORGE_BIN:-./target/release/splatforge} convert`.
2. Spins up a local static server and launches headless Chromium via
   `playwright-core` against `harness/page.html`.
3. Captures 8 frames per side, pixel-diffs them, writes `diff.json` + a
   self-contained `diff.html` (rendered via `@splatforge/report-ui`).

Exit code is `0` for pass (mean pixel diff &le; threshold), `1` for fail.

## CI hint

Run the matrix and upload `report/` as an artifact:

```yaml
- run: pnpm -F splatforge-visual test
  env:
    SPLATFORGE_DIFF_THRESHOLD: '0.03'
- run: pnpm -F splatforge-visual run report
- uses: actions/upload-artifact@v4
  with:
    name: splatforge-visual-report
    path: tests/visual/report
```

`webkit-webgl2` and `firefox-webgl2` are marked optional in SPEC-0010 —
allow them to fail in CI via `continue-on-error` or a per-project
`expectedFailure` annotation.

## Hero visual-regression (pink-blob guard)

A separate, lightweight Playwright config catches the failure mode that
produced a "pink blob" hero in production: an asset that loads without
errors but renders as a single saturated color or all-black.

```bash
# Requires: cargo build --release -p splatforge-cli
#           pnpm -F @splatforge/viewer build
pnpm -F splatforge-visual run test:hero
```

The suite optimizes every (scene, preset) ∈
`{bonsai, bicycle, garden, stump} × {web-mobile, web-desktop, quest-browser, hero, quality-max}`
through `splatforge optimize --chunked`, loads each result in
`preview-hero.html`, and asserts mean luminance and RGB variance over the
canvas pixels. Screenshots land in `tests/visual/__screenshots__/` for
human review.

PLY sources for each scene are discovered via:

1. `SPLATFORGE_PLY_<SCENE>` env var (preferred for CI),
2. Common agent-machine paths (`/private/tmp/<scene>-hero.ply`,
   `/private/tmp/4090-scenes-sync/<scene>.ply`, …),
3. `tests/visual/fixtures/scenes/<scene>.ply` (CI-staged location).

A missing PLY skips the case with a clear message rather than failing.

Optional: set `OPENAI_API_KEY` to enable the "antimatter15-eye"
vision-LLM check — each screenshot is classified by GPT-4o-mini into
"recognizable / blank / sparse" and the latter two fail the case.
