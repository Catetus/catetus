# Catetus interop one-pager

> Third-party integrators do not need the Catetus viewer to read
> Catetus-optimized assets. The `@catetus/glb-polyfill` package
> decodes the proprietary extensions; the `.glb.v5tail` / `.sog.v5tail`
> sidecar layers add the V5.2 residual quality bump on top of any
> base container.

## The shape of the interop story

Catetus ships gaussian-splat scenes in two container families that the
rest of the ecosystem already understands:

1. **GLB**, with the Catetus `SF_*` glTF extensions on top of the
   pending [`KHR_gaussian_splatting`](./standards/KHR_gaussian_splatting_compression_spz.md)
   base. Decode via
   [`@catetus/glb-polyfill`](../packages/glb-polyfill/README.md):
   `decodeSFExtensions(json, bin, sidecars)`.
2. **SOG** (PlayCanvas / SuperSplat-compatible). Decode SOG with whatever
   SOG loader your stack already has; no Catetus extension is needed.

On top of either container, a *companion sidecar* file
(`<base>.v5tail`) carries the V5.2 joint-tail residual. Add the sidecar:
**+6.54 dB** PSNR on the canonical-11 leaderboard for **+3.95 %** bytes,
with the base file unchanged and renderers that don't understand the
sidecar getting graceful fallback to the original quality.

```
scene.glb        ← base container (any KHR_gaussian_splatting reader)
scene.glb.shpal  ← SF palette sidecar (decoded by glb-polyfill)
scene.glb.v5tail ← V5.2 residual sidecar (decoded by glb-polyfill)

scene.sog        ← SOG base (any SOG-aware reader)
scene.sog.v5tail ← V5.2 residual sidecar (same wire format, same decoder)
```

## Why a sidecar (and not "yet another container")

- **Graceful fallback** — every existing SOG / GLB renderer keeps working.
  Older clients render the base file at the base quality; clients that
  know about `.v5tail` add the residual and get the +6.54 dB.
- **Container-agnostic** — the sidecar wire format
  (magic `SFV51TAL`, V5.2 per-cell affine variant) is the same on a GLB
  base and on a SOG base. One decoder, two ecosystems.
- **Cheap to adopt** — `decodeV5TailBytes` + `applyV5TailToScene` are
  ~50 lines of caller code; the sidecar inflates total bytes by ≈ 4 %.

The methodology and per-scene PSNR results are pinned in
[`experiments/gaussian-rasterizer-bench/CANONICAL_11_LEADERBOARD.md`](../experiments/gaussian-rasterizer-bench/CANONICAL_11_LEADERBOARD.md);
the defensive-publication writeup of the sidecar pattern lives in
[`experiments/defensive-publication/V5_2_PUBLIC.md`](../experiments/defensive-publication/V5_2_PUBLIC.md).

## How to actually wire it into a third-party viewer

1. `npm install @catetus/glb-polyfill` (publishing is gated on the
   `v5.2-public` release — for now consume via `npm link` from a
   Catetus checkout).
2. Read `packages/glb-polyfill/README.md` for the API surface.
3. Pick an example to start from in
   `packages/glb-polyfill/examples/`: `raw-canvas/` (WebGL2),
   `three.js/`, or `playcanvas/`.
4. Once your renderer is reading base GLBs, the `.v5tail` integration is
   two function calls — see the "applyV5TailToScene" snippet in the
   polyfill README.

## Standards engagement

The `.v5tail` sidecar pattern is the substantive contribution we're
proposing to standards bodies and adjacent splat formats. See
[`docs/standards-outreach/README.md`](./standards-outreach/README.md) for
the canonical-11 leaderboard methodology pitch, the Khronos
re-engagement plan, and the SPZ / USD / AEC outreach drafts.
