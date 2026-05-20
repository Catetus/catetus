# Progressive 3DGS bitstream — design memo (`.mgs2`)

**Status:** design only. No code shipped. Intended to be implemented on top of
the existing `apps/mesongs/` codec (the "MesonGS++" lineage) after the v0.5
render-fidelity check lands. See "Integration plan" at the end for the
estimated diff.

**Goal.** Replace the all-or-nothing `.mgs` download with a layered
bitstream that lets the WebGPU viewer render a recognizable scene from the
first ~5–10 % of bytes and refine to full quality as the remainder arrives.
On `catetus.com/explore` this is the difference between a 2–5 s blank
viewport and a ~200 ms first paint that progressively sharpens.

---

## 1. Algorithm summary

The codec is a layered range-coded format. Each scene is encoded once into
**one base layer L0 plus N − 1 enhancement layers L1..L_{N-1}**, contiguous
in a single file. The viewer can stop decoding at any layer boundary and
still render a complete, well-formed splat scene.

Two axes of progressivity, taken from PCGS (AAAI 2026, Oral, arXiv
[2503.08511](https://arxiv.org/abs/2503.08511)) and adapted to our
flat (non-anchor-based, non-Scaffold) 3DGS pipeline:

1. **Quantity progressivity — splat-count masking.**
   Splats are Morton-sorted (this is already what `apps/mesongs/encoder.py`
   does) and then re-ordered by a per-splat **importance score** so the most
   important splats land first. Each enhancement layer adds a fixed-size
   tranche of additional splats. L0 contains roughly 8–12 % of total splats;
   each subsequent layer roughly doubles the count until the budget is
   exhausted. A render after layer k is "all splats decoded so far" — every
   splat that exists is fully decoded (no half-quantized splats in the
   render set), so attribute correlations stay tight.

2. **Quality progressivity — bit-plane refinement of attribute quantization.**
   For attributes that benefit from extra precision (means, scales, opacity,
   rotations), we encode the **top K_0 most-significant bits of the quantized
   code in L0** and stream subsequent bit-planes in later layers. This is
   the JPEG-2000 / EBCOT idea, recast for 3DGS. PCGS calls this trit-plane
   quantization; we use ordinary bit-planes because (a) range-coded bits are
   already near the entropy limit for our data and (b) bit-planes interact
   trivially with the existing `quantize.allocate_bits` machinery.

The base layer is a **standalone valid `.mgs`-style scene** (its own coarse
quantization, its own range-coded payload). Enhancement layers are
**residual** streams: they decode either (a) new splats appended to the
splat list or (b) extra bit-plane bits to be ORed into the previously
decoded codes of existing splats. Layer boundaries are byte-aligned so the
viewer can stream-truncate.

Importance score (used to order splats for layer-tranche assignment):
the same heuristic PRoGS uses, projected onto our flat representation —
roughly `opacity * det(scale)^{2/3}` (a proxy for the splat's contribution
to a render). We pre-compute and sort once at encode time; this score is
**not** stored in the bitstream, only the resulting permutation.

---

## 2. Bitstream format (`.mgs2`)

All multi-byte fields are little-endian. Numeric types are explicit.

```
[4]   magic                        = b"MGS2"
[4]   version (uint32)             = 1
[4]   global_header_len (uint32)
[H]   global header (JSON, UTF-8)
[8]   layer_index_offset (uint64)  — pointer to the layer index table
                                     (placed at end of file; lets the encoder
                                     stream-write layers and patch a single
                                     trailing block)
[L0]  layer 0  bytes
[L1]  layer 1  bytes
 ...
[L_{N-1}] layer N-1 bytes
[I]   layer index table (one entry per layer; see below)
```

### 2.1 Global header (JSON)

```jsonc
{
  "n_splats_total":      4_812_103,         // splats in the fully-decoded scene
  "sh_degree":           3,
  "sh_k_rest":           15,
  "n_layers":            6,                 // typical: 6 layers
  "morton_axes":         "xyz",             // for forward-compat with non-xyz Mortons
  "scene_bbox":          [[lo_x,lo_y,lo_z],[hi_x,hi_y,hi_z]],
  "importance_metric":   "opacity_x_det_scale",  // tag, not the values
  "cols": [
    {
      "name":         "mean_x",
      "bits_total":   14,                  // bits-after-full-decode
      "bits_per_layer": [8, 2, 2, 1, 1, 0],// sums to bits_total
      "lo": -0.42, "hi": 11.8,
      "sym_count":    256                  // L0 alphabet (= 2^(bits_layer0 + 1) for zig-zag)
    },
    { "name": "opacity",  "bits_total": 10, "bits_per_layer": [6,1,1,1,1,0], "lo": -10, "hi": 8, "sym_count": 128 },
    ...
  ]
}
```

`bits_per_layer[k]` is the number of **additional** quantization-code bits
this column contributes at layer k. Layer 0 sets the coarse code; later
layers OR additional LSBs into each splat's code (within the splat subset
that has been "admitted" by the splat-count mask up to that layer).

If `bits_per_layer[k] == 0` for layer k, the column produces no per-splat
data in that layer — the splat-count addition still happens for it, but no
quality bits.

### 2.2 Layer index table (`[I]`)

`n_layers` rows, each 32 bytes:

```
uint64  layer_offset     // file-byte offset to start of this layer block
uint64  layer_size       // bytes in this layer block (header + payload)
uint32  splats_added     // number of NEW splats this layer admits
uint32  splats_cum       // running total after this layer
uint32  reserved         // 0
uint32  crc32            // CRC of the layer bytes; lets the viewer detect
                         // partial-fetch corruption and stop early
```

Splat ordering invariant: splats `[0 .. splats_cum[k])` are the renderable
splats after layer k. The encoder writes them in importance-then-Morton
order (importance buckets coarser than Morton — within a layer's tranche we
keep Morton order so first-difference predictor still works).

### 2.3 Layer block format

```
[4]   layer_magic = b"MGSL"
[4]   layer_idx (uint32)
[4]   layer_header_len (uint32)
[H_k] layer header (JSON, UTF-8)
[C_k] CDF blob (per-column counts arrays, uint32) for any column with new
      bits in this layer
[P_k] range-coded payload
```

Layer header:

```jsonc
{
  "splats_added":  421_000,
  "splats_cum":    421_000,                 // for L0; later layers cum > added
  "new_splat_columns":  ["mean_x", "mean_y", "mean_z", "sh_dc_r", ..., "rot_3"],
  "refine_columns":     [
    { "name": "mean_x", "bits": 2 },        // 2 additional LSBs for ALL admitted splats so far
    { "name": "opacity", "bits": 1 }
  ],
  "cdf_offsets":      [...],                // matches MGS1 layout
  "payload_len":      9_215_104
}
```

The decoder reads the layer header, decodes new-splat streams (these are
just MGS1-style per-column range-coded diffs, one column at a time), then
decodes refine streams (the bit-plane LSBs, in row-major order over the
**cumulative splat set after admission**). The refine streams use a
**separate CDF per (column, layer)** because the entropy distribution of
LSBs after coarse quantization is very different from the L0 distribution
of MSBs.

### 2.4 Why JSON in headers (when we care about size)

Global + layer headers together are ~1–4 KB total even with verbose JSON,
which is < 0.1 % of payload for a typical 5–50 MB scene. Easier to debug
and forward-compat than a packed binary header. The bulk of the file is
the range-coded payloads.

---

## 3. Decoder state machine

The viewer maintains four mutable buffers on the GPU:

- `codes[N_total, C]` — per-splat quantized integer codes, one row per
  *admitted* splat. Allocated for `n_splats_total` up front (sparse rows
  filled in as layers arrive).
- `splats[N_total]` — the dequantized splat struct used by the existing
  `ComputeDecodePipeline`. Updated on every layer commit.
- `n_visible` — number of currently renderable splats. The render pipeline
  draws splats `[0 .. n_visible)` and ignores the rest. **No re-sort
  required** between layers — splat ordering is fixed at encode time and
  every layer only appends to the visible range.
- `bit_planes_decoded[C]` — per-column, how many LSBs have been decoded
  for the currently-visible splats.

### 3.1 States

```
┌───────────┐  L0 header bytes received   ┌──────────┐
│  IDLE     │ ──────────────────────────▶ │ L0_PEND  │
└───────────┘                              └─────┬────┘
                                                 │ L0 payload received
                                                 ▼
                              ┌──────────────────────────────┐
                              │ RENDERABLE_L0                │
                              │ (n_visible = splats_cum[0])  │
                              └─────────┬────────────────────┘
                                        │ Lk payload received, for k=1..N-1
                                        ▼
                              ┌──────────────────────────────┐
                              │ RENDERABLE_Lk                │
                              │ (n_visible = splats_cum[k])  │
                              │ codes ORed with new LSBs     │
                              └─────────┬────────────────────┘
                                        │ k == N-1
                                        ▼
                              ┌──────────────────────────────┐
                              │ COMPLETE                     │
                              └──────────────────────────────┘
```

Transition into each `RENDERABLE_Lk` requires three things to be done in
order:

1. **Range-decode new splats and refine bits (CPU side, JS worker).**
   This is `constriction`-style decode; we already do this for MGS1. Wasm
   port: 3–5 ms per MB of payload on a modern laptop. A 5 MB scene split
   into 6 layers means each layer's decode is < 5 ms — well under one
   frame.
2. **Upload new code rows + write back refine bits (CPU → GPU).**
   `device.queue.writeBuffer(codesBuffer, byteOffset, ...)`. For refines we
   stage a small "delta" buffer and run a fused compute shader
   `cs_apply_refine` that for each `(splat_idx, col)` does
   `codes[splat_idx, col] = (codes[splat_idx, col] << bits) | delta`.
3. **Dequantize updated codes into splats (GPU side, new compute pass).**
   New compute pass `cs_dequantize` — one workgroup-per-column, one
   thread-per-splat-in-admitted-set. Reads `(lo, hi, bits)` from a small
   per-column constants buffer. Output is the same `splats[]` struct the
   existing pipeline already expects.

After step 3 we flip `n_visible` and on the next animation frame the
existing render pipeline picks up the new visible range. Because splat
order is fixed and only appends, **no re-sort is required** when crossing
layer boundaries; the existing radix sort runs once per frame anyway and
will incorporate the new splats automatically.

### 3.2 JS / WebGPU responsibility split

| Step | Where | Notes |
|------|-------|-------|
| HTTP `fetch` (range requests by layer offset) | JS main thread | Use `ReadableStream`; chunk on layer boundaries from the index table. |
| Range decode | JS Web Worker + Wasm (`constriction-wasm`) | Off main thread to keep frame time clean. |
| Upload codes & deltas | JS Worker → `device.queue.writeBuffer` | Worker holds a `GPUDevice` (transferable in Chromium 121+); falls back to `postMessage(Uint8Array)` + main-thread upload if not available. |
| `cs_apply_refine` (bit-plane OR) | WebGPU compute | New shader, ~10 lines. |
| `cs_dequantize` (codes → splats) | WebGPU compute | New shader, mostly an FMA per column. Could be fused into `project_gather` later. |
| Render | WebGPU graphics | **Unchanged.** Existing `cs_keygen` → radix sort → `cs_project_gather`. Just reads a different `n_visible`. |

### 3.3 Failure modes

- **Layer fetch aborts mid-payload** → CRC mismatch on layer-table entry →
  viewer drops back to last committed layer. Render quality stays at L_{k-1};
  no flicker.
- **Slow connection** → after `RENDERABLE_L0` the user sees a low-quality
  but topologically complete scene. We do not wait for higher layers to
  enable interaction (orbit/pan).
- **CPU starvation** → layer decode falls behind network. JS Worker
  coalesces multiple pending layers into one batch decode; the OR-merge is
  associative so this is safe.

---

## 4. Quality vs bandwidth curve (predicted)

These are **predicted** numbers based on (a) PCGS's reported behavior on
Mip-NeRF360, (b) the SizeGS-style allocator already in our `quantize.py`,
and (c) the importance ordering being a near-monotone PSNR predictor.
**To be validated by D2 experiments** — this memo does not measure.

For a fully-converged Inria 3DGS scene at our v0.5 outdoor preset
(~5 M splats, sh_degree=3, MGS1 baseline ≈ 28 MB → predicted MGS2 ≈ 28 MB
since the total bit budget is preserved; the layer split is overhead-only):

| Layer | Cum bytes | Cum % | Cum splats (% of total) | Predicted PSNR vs full | What the user sees |
|-------|-----------|-------|--------------------------|------------------------|--------------------|
| L0    | ~1.4 MB   | 5 %   | ~10 %                    | ~ −3 to −4 dB          | Scene shape, hero subject readable, distant detail blurry. |
| L1    | ~2.8 MB   | 10 %  | ~25 %                    | ~ −2 dB                | Subject sharp; mid-distance objects readable. |
| L2    | ~5.6 MB   | 20 %  | ~50 %                    | ~ −1 dB                | Indistinguishable from full on phone-sized viewport. |
| L3    | ~11.2 MB  | 40 %  | ~75 %                    | ~ −0.4 dB              | Indistinguishable on laptop viewport. |
| L4    | ~19.6 MB  | 70 %  | ~95 %                    | ~ −0.1 dB              | Reference quality. |
| L5    | ~28.0 MB  | 100 % | 100 %                    | 0 dB                   | Full quality. |

These columns derive from a Lagrangian split of the SizeGS allocator's
rate-distortion curve: the first 10 % of splats (top-importance) carry
roughly 60 % of the rendered-pixel energy, so PSNR climbs steeply at the
left of the curve. The numerical gap to the published PCGS curve on
Mip-NeRF360 is small (PCGS reports ~ −2.5 dB at 8 % of bits, ~ −0.3 dB at
50 % of bits) and we expect to land in the same envelope after one tuning
pass.

**First-paint budget:** on a 10 Mbps connection, 1.4 MB downloads in
~1.1 s; on 100 Mbps, ~110 ms. The L0 decode + upload + dequantize is
≤ 50 ms on a laptop. So **200 ms first-paint is achievable above ~50 Mbps**;
below that the network dominates and we should fall back to a smaller L0
(e.g. 3 % of bytes) configurable per scene.

### Layer count sensitivity

- `n_layers = 4`: cleaner UX (fewer perceptible "ratchet" steps); more
  bytes in L0 (~12–15 %) before first paint.
- `n_layers = 6`: balanced, the recommended default.
- `n_layers = 10`: smoother visual ramp but more per-layer header overhead
  (still < 1 %) and more JS Worker round-trips.

Header overhead at `n_layers = 6` for a 28 MB scene is ~3 KB of JSON +
192 B of index table ≈ 0.011 %. Trivial.

---

## 5. Integration plan with MesonGS++

Codebase under audit: `catetus-private/apps/mesongs/` on branch
`research/mesongs-plusplus`. Files: `bitstream.py` (114 LoC),
`encoder.py` (216 LoC), `decoder.py` (103 LoC), `quantize.py` (164 LoC).

### 5.1 What changes in each file

**`bitstream.py` (~+250 LoC).**
- Add `MAGIC2 = b"MGS2"`, version field, `GlobalHeader` and `LayerHeader`
  dataclasses (mostly a rename + extension of `FileHeader` / `ColHeader`).
- Add `LayerIndexEntry` dataclass and `write_mgs2(path, global_header,
  layers, index)` / `read_mgs2(path)` (and `read_mgs2_layer(path, k)` for
  range-read of a single layer).
- Keep MGS1 reader functions intact (we ship a back-compat decoder for
  existing assets).

**`quantize.py` (~+80 LoC).**
- New `allocate_bits_progressive(columns, target_bits, n_layers,
  l0_fraction=0.30)` that returns `List[ColumnSpec]` extended with
  `bits_per_layer: List[int]`.
  Algorithm: run the existing `allocate_bits` to get `bits_total[c]`, then
  split each `bits_total[c]` across layers with the rule "most sensitive
  columns get the most bits in L0, least sensitive get all their bits in
  late layers". Concretely: sort columns by per-bit marginal distortion
  drop, then assign `floor(bits_total[c] * w_k)` bits to layer k, where
  `w_k` is a precomputed split that satisfies `sum_k w_k = 1` and concentrates
  weight in L0 for high-priority columns. SENSITIVE_PREFIXES (already in
  the file) get all their bits in L0.
- No changes to `quantize_column` / `dequantize_column`.

**`encoder.py` (~+220 LoC).**
- New `encode_ply_progressive(src, dst, *, target_mb, n_layers=6, ...)`.
  Reuses `read_inria_ply`, `morton_argsort_from_floats`, and the per-column
  flatten.
- After Morton sort, compute **importance score** (one pass over `opacity`
  and `scales` — both already columns we read), then a stable argsort that
  buckets by importance with Morton as the tie-breaker.
- Apply `allocate_bits_progressive` to get `bits_per_layer` per column.
- For each layer k = 0..N-1:
  - Determine `splat_range_k = [splats_cum[k-1], splats_cum[k])`.
  - New-splats stream: for each column with `bits_per_layer[k] > 0`,
    compute the quantized code at the **cumulative** precision after layer
    k (i.e. `sum(bits_per_layer[:k+1])` bits), take the first-difference
    over the new splat range, fold and range-code (same as MGS1).
  - Refine stream: for each column with `bits_per_layer[k] > 0` and k > 0,
    extract the new LSB bits for the **existing** splat range
    `[0, splats_cum[k-1])` (re-quantize at the higher precision, subtract
    `code << bits_new`, take the residual, range-code with a fresh CDF).
  - Write layer header + CDF blob + payload.
- Write the global header (front) and layer index table (tail).
- The old `encode_ply` stays as a thin wrapper that calls
  `encode_ply_progressive(n_layers=1)`. Down the road we deprecate it.

**`decoder.py` (~+180 LoC).**
- New `decode_mgs2(src, dst)` that decodes **all** layers and writes the
  final PLY. Identical observable output to `decode_ply` for the
  `n_layers=1` case.
- New `decode_mgs2_upto(src, k, dst)` for offline testing of partial
  decodes (used by the D2 quality-vs-bandwidth benchmark).
- Helper `apply_refine(codes, new_lsbs, bits_new)` =
  `codes = (codes << bits_new) | new_lsbs` (numpy one-liner).

**`bench.py` (~+50 LoC).**
- Add a sweep mode: for each layer k = 0..N-1, decode-up-to-k, dequantize,
  re-render via the existing eval rig, log PSNR / SSIM / LPIPS vs ground
  truth. Produces the data backing § 4.

**`__main__.py` (~+30 LoC).**
- New CLI flag `--progressive --n-layers 6` on the `encode` subcommand.
- New `decode-progressive --upto K` subcommand.

### 5.2 Viewer side (not in `apps/mesongs/`; lives in `apps/web/`)

Two new files (~400 LoC total):

- `apps/web/src/lib/mgs2-stream.ts` — the layer-fetcher + decoder host.
  Wraps `fetch` with byte-range requests, runs a Web Worker that holds a
  `constriction-wasm` decoder, emits `onLayerReady(k, codes, deltas)`
  events.
- `apps/web/src/lib/mgs2-gpu.ts` — the WebGPU side. Allocates the
  `codes` and `splats` buffers up front, defines `cs_apply_refine.wgsl`
  and `cs_dequantize.wgsl`, runs them on `onLayerReady`.

The existing `TryIt.astro` dropzone state machine (`feedback` log:
"dropzone state machine — idle/uploading/processing/done/error") gets two
new substates inside `done`: `done:l0` (renderable, refining) and
`done:complete`.

### 5.3 Effort estimate

| Component | LoC | Effort |
|-----------|-----|--------|
| `bitstream.py` v2 | +250 | 1 day |
| `quantize.py` allocator | +80 | 0.5 day |
| `encoder.py` progressive | +220 | 1.5 days |
| `decoder.py` progressive | +180 | 1 day |
| `bench.py` sweep | +50 | 0.5 day |
| `apps/web/.../mgs2-stream.ts` | ~200 | 1 day |
| `apps/web/.../mgs2-gpu.ts` | ~200 | 1 day |
| Wasm port of constriction range decoder | — | 1 day (existing wasm crate; just wire it) |
| Visual QA + bench | — | 1 day |

**Total: ~8 engineer-days for a v1 that ships.** Round to **two
calendar weeks** with normal review cycles. Cleanly parallelizable into
"codec side" (D2a, 4 days) and "viewer side" (D2b, 3 days) once the
bitstream format in `bitstream.py` is frozen.

### 5.4 Back-compat

- MGS1 files keep decoding via the existing `decode_ply`. We never
  rewrite shipped assets.
- The web viewer detects `MGS2` magic and falls back to the old
  monolithic loader on `MGS1`.
- The encoder gains a `--progressive` flag but defaults stay on the
  non-progressive path until v0.6.

---

## 6. Open questions

1. **Importance metric calibration.** `opacity * det(scale)^{2/3}` is a
   reasonable proxy but PRoGS uses per-pixel contribution accumulated
   over the training views, which is strictly better. For our pipeline we
   don't always have training views available at encode time (we encode
   shipped `.ply`s, not during training). Question: is the cheap proxy
   within ~0.5 dB of the training-view-aware metric? D2 experiment.

2. **Bit-plane allocation per column.** § 5.1 sketches a heuristic split
   of `bits_total[c]` across layers. The optimal split is the Lagrangian
   solution of "min sum-distortion s.t. per-layer-bytes constraint" — i.e.
   the same allocator we already have, but run N times with the per-layer
   byte caps. Probably worth implementing the full per-layer ILP rather
   than the heuristic in v1, even at +1 day of effort.

3. **Refine-stream CDF cost.** Each refine layer ships its own CDF for
   each column it touches. For 6 layers × ~80 columns × ~256 symbols ×
   4 B = ~500 KB of CDF blob in the worst case. That's 1.8 % of a 28 MB
   scene. Might want to share CDFs across layers when the LSB distribution
   is uniform-enough (it usually is). D2 experiment: measure entropy
   loss from a single shared CDF per column.

4. **What goes in L0 — splats vs bits?** Our default is "10 % of splats
   at full coarse-bit precision". PCGS does roughly "all anchors at very
   coarse precision". The PCGS choice gives smoother visual ramps; the
   splat-tranche choice gives a faster first paint at the cost of a more
   visible refinement step. Question: should L0 be 10 % full-precision or
   30 % half-precision? D2 user-perception test on three scenes.

5. **Interaction with v0.5 hosted-neural-outdoor preset.** The hosted
   neural decoder runs **after** the splats are on the GPU; in principle
   it could also run on partial splat sets. Need to confirm the neural
   refiner's training distribution covers low-splat-count inputs. If not,
   we may need to gate the neural pass on `state == COMPLETE`.

6. **HTTP range request granularity.** CDN-level: do Vercel and Cloudflare
   support 206 partial-content for `.mgs2` files out of the box? Almost
   certainly yes (it's just a static asset) but worth a 30-min smoke test
   before depending on it. Alternative: split into N actual files
   (`scene.mgs2.l0`, `scene.mgs2.l1`, ...) and lose the single-file
   property in exchange for cache-friendliness.

7. **WebGPU buffer-zeroing cost on `n_splats_total` allocation.** On
   browsers that zero-init storage buffers (most do), allocating an
   N-row codes buffer up front costs O(N) bytes of upload-side memset.
   For 5 M splats × 80 cols × 2 B = 800 MB upper bound on the codes
   buffer. We should pack codes more tightly (bit-packed per-column
   uint8/uint16 tracks) or stream the buffer in tile-blocks. This is the
   one place where the design might need to shift before v1 ships.

---

## 7. Why this beats the obvious alternatives

- **"Just gzip the file"** — already done by the CDN. Doesn't help
  first-paint because the viewer can't render gzip-stream bytes
  mid-decompression.
- **"Just stream the raw splats in importance order, uncompressed"** —
  works (PRoGS does this) but throws away the 18× compression of
  MesonGS++. Our 5 MB compressed scene becomes 90 MB on the wire.
- **"Run a separate low-res scene"** — doubles encode time and storage,
  and the L0 / full-quality switch is visible as a pop. Layered codec
  refines smoothly.
- **"Use Spark 2.0's .RAD format directly"** — closed-format dependency,
  doesn't compose with our existing `.mgs` pipeline. Our `.mgs2` is a
  natural superset of MGS1 and stays under our control.

---

## 8. Sources

- PCGS: Progressive Compression of 3D Gaussian Splatting (AAAI 2026, Oral).
  arXiv [2503.08511](https://arxiv.org/abs/2503.08511).
- ProGS: Towards Progressive Coding for 3D Gaussian Splatting.
  arXiv [2603.09703](https://arxiv.org/abs/2603.09703).
- PRoGS: Progressive Rendering of Gaussian Splats (WACV 2025).
  arXiv [2409.01761](https://arxiv.org/abs/2409.01761).
- LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive
  Streaming (3DV 2025, Best Paper MMSys'25).
  arXiv [2408.14823](https://arxiv.org/abs/2408.14823).
- Context-Based Trit-Plane Coding for Progressive Image Compression
  (CVPR 2023) — methodological ancestor of PCGS's trit-plane.
- JPEG 2000 EBCOT — methodological ancestor of bit-plane progressive
  coding in general. ITU-T T.800.
