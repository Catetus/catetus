# `v5.2-public` — Release Notes (Defensive Publication)

**Tag:** `v5.2-public`
**Anchored commit:** `451995b` (`v5-2-patentability: prior-art landscape +
patentability verdict for V5.2 sidecar`)
**Publication date:** 2026-05-18
**Document:** `experiments/defensive-publication/V5_2_PUBLIC.md`

This tag exists to establish a public, dated, citable artifact for the
Catetus V5.2 sidecar codec and T2.1.R Jacobian-weighted Lloyd recipe.
It is a **prior-art anchor**, not a product release. The accompanying
arXiv-style markdown is the authoritative technical description; this
file enumerates exactly what the tag covers.

## In scope (publication-grade)

- **Defensive publication document** — `experiments/defensive-publication/
  V5_2_PUBLIC.md` (~6 pages, 8 sections plus 2 appendices). Specifies
  T2.1.R, the V5.2 sidecar architecture, the SOG-container integration,
  the bonsai + canonical-11 bench results, and the wire format with enough
  detail for a third-party decoder.

- **T2.1.R Jacobian-weighted Lloyd centroid update** for VQ palette
  compression of SH-rest. Reference impl:
  - `crates/catetus-optimize/src/vq_palette.rs` (weighted-mean centroid
    update + Gumbel-top-K weighted training-pool sampler).
  - `--jacobian-sidecar <NPZ_PATH>` CLI flag on `catetus optimize`.
  - `PassContext.sh_rest_weights` plumbing through `RemoveInvalidSplats`
    and `MortonSort`.
  - Experiment: `experiments/render-space-lloyd-max-rust-port/RESULT.md`
    (+6.16 dB on bonsai at strictly negative byte cost).

- **V5.2 joint per-splat tail-protection sidecar** —
  `SFV51TAL` variant=2 wire format, 64 Morton cells, per-cell affine
  predictor, 8/10/12/12/8/8 mixed bit-depth profile. Reference impls:
  - **Rust encoder + decoder:** `crates/catetus-gltf/src/v5_tail.rs`
    (11 unit tests + 4 end-to-end tests; golden-fixture parity with the
    Python prototype's 802,152-byte sidecar).
  - **Rust SOG-side emitter:** `crates/catetus-sog/src/v5tail_emit.rs`
    (writes `<sog>.v5tail` alongside any SOG; supports identity, Morton,
    or arbitrary splat orderings via NN GT alignment).
  - **TypeScript decoder + apply:** `packages/sf-glb-polyfill/src/v5tail.ts`
    (golden-fixture tested; mirrors Rust decoder).
  - **Python prototype encoder:** `experiments/v5-2-composed/code/
    compose_v5_2.py`.
  - **CLI:** `catetus optimize --emit-v5-tail <gt-ply> ...`.
  - Experiments: `experiments/v5-2-composed/RESULT.md`,
    `experiments/v5-2-rust-port/RESULT.md`,
    `experiments/v5-1-sidecar-refinement/RESULT.md` (V5.1-F lineage).

- **Rust SOG writer with T2.1.R support** —
  `crates/catetus-sog/src/{writer,quantize1d,webp_enc}.rs`. Emits
  `.sog` byte-compatible with `@playcanvas/splat-transform v2.1.1
  writeSog`; SuperSplat round-trip confirmed. The `--jacobian-sidecar`
  flag flows into the SH-rest palette and produces a render-weighted
  SOG that loads unmodified in any SOG decoder.
  - Experiment: `experiments/sog-render-weighted/RESULT.md` (+2.18 dB on
    bonsai at identical byte budget vs uniform Lloyd, controlled bench).

- **glTF extension JSON contract** — `CT_v5_tail_residual` root extension
  carrying `{ "uri": "<scene>.glb.v5tail" }`. Hard-fail / graceful-ignore
  semantics keyed off `extensionsRequired` plus
  `CATETUS_ALLOW_MISSING_TAIL=1` opt-in override.

- **Golden conformance fixture** —
  `experiments/v5-2-composed/data/sidecar_v5_2.bin` (802,152 bytes on
  bonsai). Any conformant V5.2 decoder must round-trip this fixture to
  byte-exact per-group residual tensor equality.

- **Bench provenance** —
  - `experiments/v5-2-rust-port/RESULT.md` (Rust 16.78 MB / 58.679 dB).
  - `experiments/v5-2-composed/RESULT.md` (Python 16.71 MB / 59.006 dB).
  - `experiments/sog-render-weighted/RESULT.md` (SOG controlled bench).
  - `experiments/gaussian-rasterizer-bench/CANONICAL_11_LEADERBOARD.md`
    (11/11 strict-wins, +1.99 dB avg, −30.6% bytes vs SOG).
  - `experiments/harness-reconciliation-bonsai/RESULT.md` (audit-trail
    receipt: 8-frame vs 72-view, float vs uint8 PSNR, scene-file md5
    discipline).
  - `experiments/v5-2-patentability/PRIOR_ART.md` (prior-art landscape
    feeding the publication's §1 and §7).

## Out of scope (NOT in this tag)

- **No business-side material.** Nothing from `docs/partnerships/` or any
  commercial-licensing track is referenced. This is a technical-only
  archive.

- **No cross-scene T2.1.R + V5.2 numbers.** The canonical-11 leaderboard
  in §5.2 of the publication uses the SF baseline (uniform Lloyd) for
  cross-scene SF-vs-SOG comparison. T2.1.R and V5.2 composed are
  demonstrated only on bonsai. Cross-scene composition is a deferred
  follow-up (`experiments/v5-2-composed/RESULT.md` §"Open follow-ups").

- **No V5.3 sidecar shrink.** The 71%-of-bytes `shr` block (which after
  T2.1.R is ~10⁻³ everywhere) is an obvious cut, but is not implemented
  in this tag. Deferred.

- **No GLB-embedded V5.2 packaging.** The sidecar travels as a sibling
  file (`.glb.v5tail` / `.sog.v5tail`); embedding inside the GLB as a
  glTF buffer-view is a future packaging convenience, not a wire-format
  change.

- **No preset hook for V5.2.** `wmv-v52-tight` is not in the preset
  registry yet — phase D of the Rust port is gated on closing the last
  0.33 dB drift to the Python prototype. Encoding is driven via the
  `--emit-v5-tail` flag on the existing `wmv-vq45-no-prune-tight` preset
  in this tag.

- **No new patent filing.** Per
  `experiments/v5-2-patentability/PRIOR_ART.md`, the V5.2 combination is
  at high § 103 obviousness risk over GoDe + CompGS + G-PCC + SizeGS.
  This release is the defensive-publication path explicitly chosen as
  the primary IP move. No provisional or non-provisional accompanies
  this tag.

- **No partner names, no financial figures, no licensing terms.** Per
  the task brief, this is a pure technical document.

## How to cite

Until an arXiv preprint number is assigned, cite as:

> Catetus Authors. "Catetus V5.2: A Render-Jacobian-Selected
> Residual Sidecar and Jacobian-Weighted Lloyd Codebook for 3D Gaussian
> Splatting Compression." Defensive technical publication, 2026-05-18.
> Repository tag `v5.2-public`, commit `451995b`.

## What changes if a future revision ships

Any future change to the `SFV51TAL` wire format (new variant, new group,
revised bit-depth profile, header field re-purposing) MUST bump the
header `version` field and MUST NOT recycle `variant=2` for incompatible
semantics. The decoder reference impls (`v5_tail.rs`,
`v5tail_emit.rs`, `v5tail.ts`) hard-fail on unrecognized
version/variant combinations — that is the contract conformant third-
party implementations should mirror.

## Verifying the tag

```sh
git fetch --tags
git show v5.2-public                    # commit metadata
git log --oneline 451995b -1            # 451995b (anchored commit)
md5sum experiments/v5-2-composed/data/sidecar_v5_2.bin
# expected:  ... 802152-byte bonsai V5.2 sidecar (golden fixture)
```
