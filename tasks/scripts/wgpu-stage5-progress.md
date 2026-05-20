# Stage 5 (radix sort multi-dispatch) — landed; what's next

## Status: LANDED at 293663e on main (push: 359e1e0..293663e)

Stage 5 of the wgpu-multidispatch series chunks the radix sort + merge path so
the full per-frame compute pipeline works above the WebGPU 1.0 dispatch cap
(16 776 960 splats per single dispatch).

## What landed

**Approach chosen: (a) per-chunk sort + binary merge tree.**

Approach (b) — chunked global histogram + scan — was rejected after
inspection of `cs_scan_block_sums` in `scan_multiblock.wgsl`. That kernel
has an algorithmic correctness cap at `numScanWgs <= WG_SIZE * WG_SIZE =
65536`, which is hit at exactly the same 16.7M-splat threshold as the
dispatch cap. Going past that requires recursively chained block-sums,
which is a much bigger change than the merge-tree approach. The followup
spec already listed (a) as a sound fallback; we took it.

### Kernel changes

- `packages/viewer/src/webgpu/radix_sort.wgsl` — `Uniforms` gained
  `chunk_offset_splats: u32`. `cs_histogram` reads `keys_in[i +
  chunk_offset_splats]`; `cs_scatter` reads from offset and writes to
  `wg_offsets[bin] + local_rank + chunk_offset_splats`.
- `packages/viewer/src/webgpu/histogram_subgroup.wgsl` — same uniform
  addition, same read-side offset.
- `packages/viewer/src/webgpu/radix_merge.wgsl` (new) — pairwise stable
  merge over (u32 key, u32 value) using Merge-Path binary search per output
  slot. Stable on ties (favors A side, which preserves global stability
  because A appears first in the input). Per-chunk dispatch via
  `chunk_offset_splats` so each merge can itself exceed 16.7M outputs.

### Orchestration (`radix_sort.ts`)

- `encode()` splits at `SPLAT_DISPATCH_CAP = 65535 * 256`.
- `count <= cap` goes through `encodeChunkInPlace(_, 0, count)` —
  bit-equivalent to the prior 4-pass radix path with
  `chunk_offset_splats = 0`.
- `count > cap`: per-chunk sort then `log2(K)` merge rounds with ping-pong
  between keysA/B; final copyBufferToBuffer if rounds are odd so the
  public contract ("sorted result lives in keysA/valuesA") holds.

### Bench gate removed

`bench/real-scene-lodge.bench.ts:336–358` previously enforced
`dispatchCap = 65535 * 256` via `capacityCap = min(dispatchCap, bufferCap)`.
The dispatchCap variable was deleted; the only remaining ceiling is
`bufferCap = floor(maxBufferSize / BYTES_PER_DECODED_SPLAT) - 1`.

### Tests

18 new tests in `packages/viewer/src/__tests__/multi-dispatch.test.ts`
(part of the existing test file rather than a new file because
`multi-dispatch.test.ts` is the natural home — it's where the planner and
chunked-dispatch helper already live, and Stage 5 is a direct continuation
of that work):

1. `radix_merge.wgsl` shader sanity (entry point + uniform contract)
2. `radix_sort.wgsl` + `histogram_subgroup.wgsl` `chunk_offset_splats`
   contract pinning
3. `chunkedSortMirror` cross-checks against `cpuStableSort` on:
   - 1 chunk (single-chunk fast path)
   - 2 chunks, 3 chunks (odd), 4 chunks, 10 chunks (stable-ties stress)
   - 8 chunks (LODGE L0 shape)
   - LODGE L1 shape (4 partial chunks)
   - n=0 and n=1 edge cases
4. Dispatch-shape sanity for L1 (4 chunks) and L0 (8 chunks)

Full viewer test result: **197/197 pass** (179 baseline + 18 new). No
existing test changed.

## What's still operator-pending

### 4090 LODGE bench at L1 / L0

The Tailscale SSH path to MontesPC was not available in this agent
invocation; the 4090 skill is gated by Anthropic permission and was
denied. To collect the bench:

```
# On MontesPC (Windows):
#   schtasks /Run /TN sf_bench_real
# or run packages/viewer/bench/real-scene-lodge.bench.ts directly via the
# bench harness against the 4090.
```

Update `packages/viewer/bench/results-4090-lodge.json` with the L1 (54M)
result. Expected: single-digit fps given DRAM bandwidth ceiling (decoded
splat buffer ~3.4 GB at L1, 7.6 GB at L0) — that's compute-bound not sort-
bound, so the chunked sort itself should be a small contributor.

### Visual proof at L1 (Sweet Corals)

Once L1 renders end-to-end on the 4090, capture
`packages/viewer/bench/screenshots-multi-dispatch/sweet-corals-l1.png` via
the standard playwright harness.

## Followup: Stage 6 — maxBufferSize ceiling at 119M

Per-buffer max on most adapters is 256–512 MB. The decoded splats buffer
wants `119M * 64B = 7.6 GB` at L0. To render L0 end-to-end we need to
split decoded splats into N storage buffers and bind the active range
per-chunk for each per-splat kernel. The per-splat kernels already chunk
on dispatch (Stage 1-4); Stage 6 makes them chunk on buffer binding too.

The sort itself is fine at L0 in this regard: the histogram buffer is
`numWgs * RADIX * 4 = 465_625 * 256 * 4 = 476 MB`, which exceeds many
adapters' per-buffer max. So Stage 6 also needs to split the histogram
buffer or use per-chunk smaller histograms (which is already true given
the per-chunk sort approach we chose — each chunk's histogram is
`65535 * 256 * 4 = 67 MB`, well under any adapter's per-buffer max).

The histogram buffer in radix_sort.ts is allocated for the worst case
(`maxWgs * RADIX * 4`); to be safe with maxBufferSize, that allocation
should also clamp to `dispatchCap * RADIX * 4 = 67 MB` since the per-chunk
sort never uses more than that — the rest of the allocation is wasted.
That's a 1-line change worth bundling with Stage 6.
