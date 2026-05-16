# Task #154 Stage 6 — Partial Ship Verdict (sf-154)

**Branch:** `research/154-wgpu-stage6-buffer-split`
**Status:** Foundation shipped; L1/L0 still blocked on a SECOND single-buffer ceiling.

## What shipped this pass

Built on commit `ad4f28f` (BufferPager primitive + scale.astro maxBuffer relax).

- **Templated WGSL for cs_keygen + cs_project_gather** — both come from
  `cs_project_gather.wgsl` (single shader file, two entry points binding
  the splats array under different names `k_splats` / `g_splats`). The
  `templateSplatsAccess` helper is invoked twice per pager init; each
  invocation emits N page bindings + a `read_splats_*(i)` switch helper
  and rebases downstream bindings.
- **`ComputeDecodePipeline` migrated**:
  - Replaces single `splatsBuffer` with `pager: BufferPager` (sized to
    `adapter.limits.maxStorageBufferBindingSize`).
  - When `pager.numPages == 1`: behaviour identical to pre-Stage-6.
  - When `pager.numPages > 1` AND fused project_gather: builds templated
    pipelines + paged bind groups.
  - `uploadChunk` routes destination-buffer writes through
    `pager.pageRanges` (asserts chunk doesn't straddle, which never
    happens in practice for the LODGE chunker at ~256K splats/chunk vs
    ~33M splats/page).
- **`useCull` / `useWSR` / `useWSRTile` / non-fused** all throw with
  clear errors when pager.numPages > 1. Single-page builds unaffected.
- **`dispatchPerSplatPaged` helper** added to `multi-dispatch.ts` (not
  used in this commit; available for follow-up monotonic kernels).
- **Bench cap raised** to the new bottleneck: instance buffer at
  2147483648 / 48 ≈ 44.7M splats. (Was 33M splats from the now-paged
  decoded-splats buffer.)
- **3 new unit tests** for `templateSplatsAccess` against the real
  `PROJECT_GATHER_WGSL` source (k_splats + g_splats + binding rebasing).

Total: 207/207 tests pass, build green.

## L1 / L0 blocker

Sweet Corals on the 4090 bench:

| Level | Splats   | Decoded-splats buffer | Instance buffer |
|-------|----------|-----------------------|-----------------|
| L0    | 119.8M   | 7.6 GB (4 pages)       | 5.7 GB ❌       |
| L1    | 54.2M    | 3.5 GB (2 pages)       | 2.6 GB ❌       |
| L2    | 28.0M    | 1.8 GB (1 page) ✓      | 1.3 GB ✓        |

The 4090 bench at HEAD `a5310ea`:
- L0/L1: skipped — bench cap is `Math.floor(maxBufferSize/48) ≈ 44.7M`
  because the instance buffer is a single GPUBuffer (vertex-bound) that
  hasn't been paged.
- L2: rendered at 2.55 fps (median, 3 viewpoints).
- L3-L5: rendered at 4.79 / 8.76 / 20.61 fps respectively.

The instance buffer is the SECOND ceiling. Paging it requires:
1. New `InstanceBufferPager` analogous to splat pager.
2. Per-instance-page run of `cs_project_gather` with dynamic-offset
   bindings for `g_indices` and `g_inst_out`. Kernel WGSL likely
   unchanged (read/write at `[gid.x + chunk_offset]` becomes page-local
   when sliced).
3. Renderer-side vertex-buffer chain: one `draw()` per instance page
   (the rasterizer renders sorted-by-depth across all pages, but each
   draw consumes its page's vertex slice).

Conservatively a 1–2 day follow-up (sf-154 Stage 7). Not done in this
ship because the leverage is lower than the splat-pager foundation it
builds on, and the user explicitly authorized partial-ship if the
runway is short.

## What was attempted but not shipped

- Live 4090 bench of L1 → fps measurement (because L1 still gets capped).
- Pixel-identity diff vs main baseline (no rendering possible).
- L0 WSR-tile gate (WSR-tile multi-page deferred entirely — `useWSRTile`
  throws on >1 page).
- Playwright screenshots (no L1/L0 render → nothing meaningful to shoot
  beyond L2 baseline).

## Files touched in this branch (post-foundation commit `ad4f28f`)

- `packages/viewer/src/webgpu/index.ts` (+299 net) — pager wiring,
  templated paged pipelines, multi-page guards.
- `packages/viewer/src/webgpu/multi-dispatch.ts` (+82) — paged dispatch
  helper for follow-up monotonic kernels.
- `packages/viewer/src/__tests__/webgpu/buffer-pager-template-pg.test.ts`
  (new, 60 LOC) — 3 templater-on-real-WGSL tests.
- `packages/viewer/bench/real-scene-lodge.bench.ts` — bench cap raised
  to instance-buffer limit; switched to fused project_gather path.
- `tasks/scripts/sf154-migrate-index.py` (new, 270 LOC) — surgical
  migration script kept for traceability.

## Honest L1 ship path (follow-up)

For the next agent picking this up:

1. Mirror BufferPager → InstanceBufferPager (just float counts × 48 B).
2. Wire it into ComputeDecodePipeline as `instancePager` and replace
   the single `instanceBuffer`.
3. cs_project_gather: bind output via dynamic-offset slice per
   instance-page; chunk_offset already supported. Test with synthetic
   54M scene first.
4. Renderer (`renderer/webgpu.ts`): N draws across vertex pages.
5. Bench cap → effectively unlimited (page count grows with capacity).
6. Re-run 4090 bench to confirm L1 fps appears.

The same pattern then extends to the sort key/value buffers if L0
(119M × 8 bytes = 952 MB keys + 952 MB values) gets close enough to
the 2 GiB cap to matter.
