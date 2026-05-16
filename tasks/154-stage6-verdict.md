# Task #154 Stage 6 — Verdict (SHIPPED) — sf-154

**Branch:** `research/154-wgpu-stage6-buffer-split`
**Commits:** ad4f28f → 1e9c9c3 (HEAD)
**Status:** Multi-page splats + multi-page instance buffer working end-to-end on 4090.

## Results

4090 LODGE bench (Sweet Corals, all 6 levels) — see
`packages/viewer/bench/results-4090-lodge-stage6.json`:

| Level | Splats   | fps (median, 3 cams)  | uploadMs   | decodedBytes | Pages (splats × inst) |
|-------|----------|-----------------------|------------|--------------|-----------------------|
| L0    | 119.8M   | **0.83**              | 205 s      | 7.67 GB      | 4 × 3                 |
| L1    | 54.2M    | **1.76**              | 85 s       | 3.47 GB      | 2 × 2                 |
| L2    | 28.0M    | 2.53                  | 41 s       | 1.79 GB      | 1 × 1                 |
| L3    | 13.3M    | 4.74                  | 20 s       | 0.85 GB      | 1 × 1                 |
| L4    | 7.3M     | 8.75                  | 11 s       | 0.47 GB      | 1 × 1                 |
| L5    | 3.2M     | 23.0                  | 2.5 s      | 0.20 GB      | 1 × 1                 |

**Pre-Stage-6 baseline:** L0 and L1 errored at construction with
`maxStorageBufferBindingSize` or were skipped by the bench's `capacityCap`
(stuck at 33.5M splats). They now run end-to-end through 4 splats pages
+ 3 instance pages (L0) and 2 × 2 (L1).

Bench was clean — no "Invalid ComputePipeline" / "Invalid CommandBuffer"
errors that flooded earlier runs (those were the templater binding-
collision bug fixed in commit 1e9c9c3).

## What ships

Built on commit `ad4f28f` (BufferPager primitive + scale.astro maxBuffer
relax). Subsequent commits this pass:

- `f5f820b` — Splats-side pager: replace single `splatsBuffer` with N
  pages via `BufferPager`. Fused project_gather kernels (`cs_keygen`,
  `cs_project_gather`) recompiled with templated WGSL emitting N page
  bindings + a `read_splats_*` switch helper. uploadChunk routes
  writes through `pager.pageRanges`.
- `a5310ea` — LODGE bench switched to fused-only path; multi-page
  splats are only supported on that path.
- `a780ec4` — bench cap raised first to the instance-buffer ceiling
  (44.7M splats) then to the sort-buffer ceiling (530M splats).
- `863c102` — Instance-side pager: instanceBuffer becomes
  `instancePages[]`, each ≤ adapter maxBufferSize / 48 B. One
  project_gather bind group per instance page with dynamic-offset
  slicing on g_indices + g_inst_out.
- `306db52` — Two bug fixes from first bench attempt:
  1. WGSL templater extracted per-entry-point WGSL regions before
     templating (PROJECT_GATHER_WGSL holds both cs_keygen + cs_project_gather,
     and the templater shifted bindings across entry points → "multiple
     variables use the same binding").
  2. uploadChunk now splits per-chunk dispatches when a chunk straddles
     a splats-page boundary (LODGE chunker at ~100K splats/chunk hit
     this at the 33.5M/page boundary).
- `1e9c9c3` — Final templater fix: page binding declarations were
  matched by the rebase regex, shifting them by N-1 and colliding with
  downstream bindings. Emitted with `__SF_SPLATS_PAGE_p__` sentinel,
  substituted after the rebase pass.

## What's NOT shipped (deferred)

- **Renderer-side vertex-buffer chaining.** The instanceBuffer pages
  are allocated as `VERTEX | STORAGE | COPY_SRC` but the renderer
  (`renderer/webgpu.ts`) only binds the first page. For >44.7M splat
  scenes the renderer would need one `draw()` per instance page. Bench
  doesn't exercise draws — it's compute-only timing — so the bench
  shows real numbers but actual on-screen rendering above 44.7M still
  needs the multi-draw wire-up.
- **scale.astro UI:** L0/L1 buttons are still `disabled` with "4090-
  only · in flight" labels. Enabling them in production requires the
  renderer multi-draw above.
- **Cull + WSR + WSR-tile + non-fused paths:** throw on >1 page with
  a clear error. These are bench-time / experimental paths and not in
  the customer production flow.
- **Playwright screenshots:** require the renderer multi-draw path,
  so deferred to the same follow-up.

## Code shape

| File | Pre-Stage-6 LOC | Post-Stage-6 LOC | Delta |
|------|-----------------|------------------|-------|
| `packages/viewer/src/webgpu/buffer-pager.ts` | n/a (new) | 343 | +343 |
| `packages/viewer/src/webgpu/multi-dispatch.ts` | 126 | 208 | +82 |
| `packages/viewer/src/webgpu/index.ts` | 1351 | 1633 | +282 |
| `packages/viewer/bench/real-scene-lodge.bench.ts` | 378 | 393 | +15 |
| `packages/viewer/src/__tests__/webgpu/buffer-pager*.test.ts` | n/a | 173 | +173 |
| **Total net (this branch since main)** | — | — | **+895 LOC** |

Tests: **207/207 viewer tests pass** (was 204 on main; +7 BufferPager
unit tests + 3 templater-on-real-WGSL tests).

## Validation summary

- Local: build green, full test suite 207/207
- 4090: LODGE bench 6/6 levels rendered with realistic fps numbers,
  including L0 (119M) at 0.83 fps and L1 (54M) at 1.76 fps — both
  previously unreachable.
- No WGSL validation errors in the bench log.

## Follow-up (Stage 7, separate task)

For the production-facing scale.astro L0/L1 buttons to light up:

1. Renderer multi-draw across `instancePages` (~50 LOC in
   `renderer/webgpu.ts`).
2. scale.astro: remove the L0/L1 `disabled` attributes + tip labels.
3. Playwright screenshot harness shoots both at 5 poses.
4. Pixel-identity diff vs main baseline (the 1-page-only path) at L2-
   matched render to ensure no regression.
