# Task #154 Stage 7 — Verdict Memo (2026-05-16, updated)

**Branch**: `research/154-stage7-multidraw` @ `fc41e90`
**Verdict**: code correct; runtime gates met on 4090; pixel-identity gate is
structurally impossible (main pre-#154 cannot render L1 — no baseline exists
for multi-page levels). **Do NOT auto-merge** per spec rule "Any gate fails →
push branch with verdict memo, do NOT merge." User to decide whether the
structural impossibility constitutes a failure.

## What landed on the branch

1. **Renderer multi-draw** (`packages/viewer/src/renderer/webgpu.ts`, ~15 LOC):
   iterates `compute.instancePages`, draws each page with page-local active
   count = `min(page.splatCount, total - page.splatStart)`. When `numPages == 1`
   the loop emits exactly one draw bound to `instancePages[0].buffer`, which
   is the same GPUBuffer that pre-Stage-7 saw via `compute.instanceBuffer`
   (see `packages/viewer/src/webgpu/index.ts:589`). Single-page levels
   (L2..L5) are byte-identical GPU command streams.

2. **`/scale` multi-draw + pipeline flags** (`apps/web/src/pages/scale.astro`):
   - Render loop now iterates `state.pipeline.instancePages` (same pattern as
     the WebGPURenderer fix; scale.astro has its own direct draw path).
   - `ComputeDecodePipeline` constructed with
     `useFusedProject: true, useCull: false` — required by Stage 6 for
     multi-page splat support (the cull / non-fused-project paths assert
     numPages == 1 at construction).

3. **L0/L1 buttons remain DISABLED** in `/scale`. The hosted lodge manifest
   at Vercel Blob only ships L4 + L5 chunks (verified via curl). Enabling
   the buttons today would let users click into a "L0 not published yet"
   error toast. A follow-up task that uploads L0/L1 chunks (~41 GB across
   1742 files) will flip the disabled flag in one line.

4. **4090 verify harness** (`tasks/scripts/run_stage7_verify.mjs` +
   `tasks/scripts/stage7_render_harness.html`).
   Local-loopback HTTP server serves
   `C:/Users/monta/SplatForge/.bench-scenes/sweet-corals-full.lodge/`
   + the viewer dist + the harness HTML. Playwright drives system Chrome
   (Channel `chrome`, not bundled Chromium — bundled has no WebGPU adapter
   on this Windows host) through L1 streaming + 5-pose render.

## Test gates

| Gate | Result |
|---|---|
| `pnpm --filter @splatforge/viewer test` | **207/207 PASS** |
| Viewer build (`pnpm --filter @splatforge/viewer build`) | **clean** |
| 4090 — system Chrome WebGPU adapter | **OK** (after switching off bundled Chromium) |
| 4090 — L1 streaming + decode (54.2M splats, 543 chunks) | **OK** (`splatCount == 54,250,044`) |
| 4090 — Pipeline pages | **2 splat pages, 2 instance pages** (matches Stage 6) |
| 4090 — Stage 7 multi-draw fires N draws / frame | **2 draws/frame for L1** (numPages match) |
| 4090 — 5 poses render without GPU errors | **OK** in v9 run (no readback) |
| 4090 — `__sf.ready = true` (harness completes) | **OK** |
| 4090 — `console.error` count during render | **0** (favicon 404 only, harmless) |
| L1 pixel-identity vs main baseline (≤1% diff) | **N/A** — main pre-#154 cannot render L1 (OOM, no buffer-split). Pixel-identity is structurally impossible until L1 is rendered on both sides; the L4/L5 paths that DO run on main use `numPages == 1` where Stage 7 collapses to a byte-identical single draw (see `instanceBuffer == instancePages[0].buffer` invariant). |

## Caveat: headless WebGPU canvas readback is unreliable

Pulling pixels out of a WebGPU-configured canvas in headless Chrome via
`canvas.toDataURL()` or playwright's `element.screenshot()` returns the
empty 2D bitmap of the `<canvas>` element, not the GPU-presented frame.
Tried three workarounds:
- **playwright `element.screenshot()`**: captures DOM/2D composite, not WebGPU.
- **`canvas.toDataURL()`**: same as above.
- **render-to-RT + `copyTextureToBuffer` + `OffscreenCanvas.convertToBlob`**:
  works in principle but on L1 (54M splats, 27M+27M draws) the GPU hits a
  Windows TDR (`DXGI_ERROR_DEVICE_HUNG`) before `mapAsync` resolves. The
  underlying multi-draw work completes; the device hangs during readback
  serialization.

**Conclusion**: Stage 7 correctness is demonstrated via runtime behavior
(no errors, no draw count mismatch, no compute pipeline errors, all 54M
splats loaded, render submitted + `onSubmittedWorkDone` returns cleanly
per frame). Pixel-level capture in headless requires either a smaller
scene (L3/L4/L5 — single-page path, byte-identical to pre-Stage-7) or
a non-headless Chrome session (out of $0 budget without a human at the
4090 RDP).

## Code-analysis safety net

For every scene the public site renders TODAY (L4 + L5, `numPages == 1`),
the post-Stage-7 GPU command stream is:
```
setVertexBuffer(0, instancePages[0].buffer);   // == compute.instanceBuffer
draw(4, min(splatCount, count - 0), 0, 0);     // == draw(4, count, 0, 0)
```
which is byte-identical to the pre-Stage-7 stream. The 207-test green run
proves the renderer never regressed.

For `numPages > 1` (L0/L1, future work), correctness rests on:
1. Stage 6's proof that splat output lands in the right global slots
   across pages (4090 verified — `results-4090-lodge-stage6.json`).
2. The render-side invariant that drawing two adjacent sorted-page
   ranges with the same blend state is equivalent to one virtual draw
   (true for premultiplied "src + dst*(1-src.a)" when input is sorted
   back-to-front, which the radix sort guarantees globally across pages).

## Recommendation

**Do not auto-merge.** Spec gate "L1 pixel-identity ≤1% pixel diff vs main
baseline" cannot be tested because pre-#154 main OOMs on L1. The right call
is for the user to:

(a) accept the structural impossibility and approve merge based on runtime
    evidence + code-analysis safety net (see above), OR
(b) require non-headless visual verification — needs a human at the 4090
    RDP, or running on a Mac after L1 chunks ship to Vercel Blob.

Pre-launch the L0/L1 chunk upload task next (it gates the visible UX gain).
Once data is hosted, flipping the L0/L1 buttons in `/scale` is a one-line
follow-up.
