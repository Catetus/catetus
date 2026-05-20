# Stage 6 (sf-154): wgpu maxBufferSize ceiling — progress note

Branch: `research/154-wgpu-stage6-buffer-split` (worktree at
`catetus-private/worktrees/sf-154-buffer-split/`).

## What landed in this slice

1. **`apps/web/src/pages/scale.astro` clamp relaxed.** The artificial
   1 GiB cap on `maxStorageBufferBindingSize` / `maxBufferSize` was
   replaced with the adapter's actual ceiling (typically 2 GiB on desktop
   adapters per WebGPU spec; ~256 MB on Mac default). With this single
   change, scenes up to ~33 M splats (2 GiB / 64 B per splat) now fit
   in a single decoded-splats binding without paging.

2. **`packages/viewer/src/webgpu/buffer-pager.ts`** (336 LOC, +7 unit
   tests). New `BufferPager` class + `templateSplatsAccess()` WGSL
   helper. The pager owns N storage buffers (each ≤
   `maxStorageBufferBindingSize`) and provides:
     - `splatToPage(idx)` — global index → `(page, localSplat)`.
     - `pickPageForRange(start, count)` — assert range lies in one page.
     - `pageRanges(start, count)` — generator of per-page sub-ranges.
     - `writeSplats(start, count, bytes)` — `device.queue.writeBuffer` to
       the right page(s).
     - `pageBuffers[]` — direct array of `GPUBuffer` for binding.

   The `templateSplatsAccess()` helper rewrites a kernel's WGSL to:
     - replace its single `splats : array<DecodedSplat>` binding with
       N read-only page bindings,
     - emit a `read_splats_<name>(i: u32)` selector function with a
       `switch(i / SPLATS_PER_PAGE)` over the N page bindings,
     - rebase all later binding numbers (so the kernel's other resources
       sit at `originalBinding + N - 1`),
     - rewrite all `splats[i]` accesses in the body to
       `read_splats_splats(i)`.

   Build green; all 204 viewer tests pass (7 new + 197 existing).

## What's left for L1/L0 end-to-end

The remaining work is the mechanical wiring of `BufferPager` through the
viewer pipeline. **Kernels split into two patterns:**

### Pattern 1 — per-dispatch contiguous slice (8 kernels)

These kernels all use `i = gid.x + chunk_offset` where `i` is monotonic.
They get the existing multi-dispatch wrapper extended to a "per-page,
then per-chunk-within-page" 2-level loop. Each dispatch binds a single
page's `pageBuffers[k]` and uses the per-page local `chunk_offset`. No
WGSL templating needed — the kernel still sees a single
`splats : array<DecodedSplat>` binding.

  - `cs_decode`            (`decode.wgsl`)
  - `cs_project`           (`decode.wgsl`)
  - `cs_keygen`            (`cs_project_gather.wgsl`)
  - `cs_cull`              (`cs_cull.wgsl`)
  - `cs_lod_blend`         (`cs_lod_blend.wgsl`, read_write)
  - `cs_lod_alpha_reset`   (`cs_lod_blend.wgsl`, read_write)
  - `cs_tile_bin`          (`cs_tile_bin.wgsl`)
  - `cs_wsr_accumulate`    (`cs_wsr_accumulate.wgsl`, scatters atomics
    into per-pixel buffers — atomics are fine across page-level binding
    swaps because the atomics target the *output* buffer, not splats)

### Pattern 2 — random read access (3 kernels)

These kernels index `splats[idx]` with `idx` coming from sorted-indices
or per-tile splat-list buffers. They need the WGSL templating helper.

  - `cs_project_gather`    (`cs_project_gather.wgsl`) — `splats[indices[i]]`
  - `cs_project_cmpct`     (`cs_cull.wgsl`)           — `splats[compact[i]]`
  - `cs_wsr_tile_accumulate` (`cs_wsr_tile_accumulate.wgsl`) —
    `splats[shared_idx[k]]` inside the per-pixel inner loop

The host-side bind-group-layout for these kernels also expands from
1 splats binding to N splats bindings (rebased helper bindings sit at
`originalBinding + N - 1`).

### Numerator/denominator merge (WSR-tile slot relief)

`cs_wsr_tile_accumulate` uses 5 of its 8 storage-binding slots. With
N=4 splats pages it would need 8 slots — over the spec limit. To free a
slot, **merge `numerator` (4 u32/px) + `denominator` (1 u32/px) into a
single `5 u32/px` interleaved buffer.** Touches:
  - `cs_wsr_tile_accumulate.wgsl` — write to `combined[pidx*5 + 0..4]`
  - `cs_wsr_resolve.wgsl`         — read `combined[pidx*5 + 0..4]`
  - `wsr_tile.ts`                 — single buffer allocation, single
    binding entry in both bind groups, resolve sees only one input
    binding instead of two.

After the merge, accumulate uses 4/8 slots (splats × N + tile_count +
tile_lists + combined + uniforms = 4 + N), giving N=4 budget headroom.

### Bench / visual gates

  - L1 (Sweet Corals, ~54 M splats): 2 pages of ~1.73 GiB each. Visual
    gate is pixel-identity vs. main (the Stage 5 chunked sort already
    handles dispatch chunking; only the per-buffer cap blocks rendering).
  - L0 (Sweet Corals, ~119 M splats): 4 pages of ~1.91 GiB each. Visual
    gate is "renders coherently via WSR-tile" (no baseline — first time
    119 M renders end-to-end in a browser).
  - 4090 bench: existing harness in `packages/viewer/bench/run-bench.mjs`
    + `bench/real-scene-lodge.bench.ts`. Set `SF_BENCH_PLY_DIR` to the
    Sweet Corals SoA directory.
  - Playwright screenshots at 5 poses: see
    `tasks/scripts/screenshot-pages.mjs` for the existing harness model.

## Stop reason

The BufferPager primitive + the scale.astro relax are a clean, tested
foundation. The kernel migration is ~2 000 LOC across 7 WGSL files +
4 TS files (index.ts, cull.ts, wsr.ts, wsr_tile.ts) + a multi-dispatch
wrapper extension + bind-group-layout updates + WGSL templating
integration in the embed-wgsl pipeline (since the templated WGSL is
pager-N-aware, the templating must run at runtime against
`shaders.generated.ts` rather than at embed time). That's a multi-hour
debug-cycle-heavy pass — the next slice should pick up from this
foundation, with full 4090 bench + screenshot validation budgeted as
the same atomic unit of work as the kernel migration.

## Files in this slice

  - `apps/web/src/pages/scale.astro` — clamp relax (1 line of behavior
    change, +6 lines of comment).
  - `packages/viewer/src/webgpu/buffer-pager.ts` — new, 336 LOC.
  - `packages/viewer/src/__tests__/webgpu/buffer-pager.test.ts` — new,
    102 LOC, 7 passing tests.
  - `tasks/scripts/wgpu-stage6-progress.md` — this file.

