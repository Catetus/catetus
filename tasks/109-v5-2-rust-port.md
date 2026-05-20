# Task #109 — V5.2 Joint-Tail Sidecar: Rust Port

**Goal:** Port the V5.2 sidecar codec from Python prototype to production Rust.
Match the Python prototype's bonsai result of **16.71 MB / 59.006 dB** (72-view orbit).

**Format reality:** The Python prototype produces a **`SFV51TAL`** format
(40-byte header, variant=2 per-cell affine) — NOT the `SFV5TAIL` format
sketched in `experiments/v5-format-spec/SPEC.md`. The Python format won
59.006 dB; per task spec "if Python and spec disagree, Python wins". This
port implements the **`SFV51TAL` variant-2** format (the actually-shipped
V5.2 wire format).

## Phase A — encoder skeleton + format I/O (THIS COMMIT)

- [x] Read Python prototype (`encode_v5_1.write_variant_per_cell_affine` +
      `compose_v5_2.py`) to understand exact byte layout.
- [x] New Rust module `crates/catetus-optimize/src/v5_tail.rs`:
  - [x] `V5TailHeader` struct (40 bytes: 8B magic + u16 version + u8 variant
        + u8 flags + u32 N + u32 K + u8 n_groups + u8 sh_rest_coefs +
        u16 n_cells + 8B reserved).
  - [x] `morton_sort_indices(positions: &[[f32; 3]])` — 21-bit per-axis Morton.
  - [x] `pack_bitmask_lsb_first` — `numpy.packbits(bitorder="little")` equiv.
  - [x] `bit_pack_fast(values, bit_depth)` — packs uint values at 8/10/12 bits.
  - [x] `per_cell_affine_quantize` — per-cell (min,max) → uniform quant.
  - [x] `write_variant_per_cell_affine` — full encoder for V5.2 variant-2.
  - [x] `encode_v5_2_sidecar(...)` — high-level "GT, recon → sidecar bytes".
- [x] Unit tests: round-trip on synthetic 10-splat scene.
- [x] Optional: golden test diffs first 64 bytes of synthetic encode against
      a Python-produced reference (skipped — Python-prototype golden file
      would require checked-in fixtures; the round-trip + per-cell stats
      tests cover correctness).

## Phase B — decoder + CLI plumbing (COMPLETE)

- [x] Decoder in `crates/catetus-gltf/src/v5_tail.rs` (the v5_tail module
      was moved out of `catetus-optimize` into `catetus-gltf` so
      `read_glb` can apply the residuals; optimize re-exports it):
  - [x] `GltfError::MissingTailSidecar { uri, tried }` (mirrors `MissingPaletteSidecar`).
  - [x] `ReadOpts::allow_missing_tail` + `CATETUS_ALLOW_MISSING_TAIL` env.
  - [x] `decode_v5tail_bytes` parser (reverse of write_variant_per_cell_affine).
  - [x] `apply_v5tail_to_scene` — adds per-attribute residuals to selected
        splats in raw 3DGS-PLY space.
- [x] Write-side: `WriteOpts::v5_tail: Option<V5TailRef>` so writer can emit the
      `SF_v5_tail_residual` extension entry alongside `.shpal` (mirrors
      `ShRestPaletteRef` exactly).
- [x] CLI plumbing in `catetus-cli/src/main.rs`:
  - [x] `--emit-v5-tail <gt-ply>` flag on the optimize subcommand.
  - [x] Joint Jacobian sidecar loader (multi-array .npz auto-detected:
        J_position, J_rotation, J_opacity, J_scale, J_dc, J_sh_rest;
        also accepts a precomputed `J_joint_sum`). Computes
        `J_joint_sum[i] = Σ_c J_c[i] / max(J_c)`.
  - [x] After GLB write, also writes `.glb.v5tail` sidecar.

Tests landed (19 total, all passing):
- 11 Phase A unit tests (encoder + golden header).
- 4 Phase B unit tests (decoder round-trip, bitmask unpack, bit-unpack,
  decoder golden against the 802 KB Python sidecar).
- 4 Phase B e2e tests (sidecar apply, missing-sidecar hard-fail,
  missing-sidecar permissive-warn, full GLB write+sidecar+read round-trip).

## Phase C — integration bench on 4090 (LANDED — 0.33 dB short of 0.1-dB target)

- [x] Re-encoded bonsai with Rust CLI (commits ec7bd96, 914a75e, 9e22c51,
      4 iterations to chase the residual-space coordinate bug).
- [x] Decoded with `catetus convert --to ply` (auto-applied sidecar
      via the GLB-resident `SF_v5_tail_residual` extension).
- [x] 72-view orbit bench on 4090 (`bench_repaired.py` 24 az × 3 el,
      512 px, sh=3, gsplat).
- [x] **Bonsai PSNR: 58.679 dB** (Python target 59.006 dB; miss 0.33 dB).
      Sidecar bytes within 0.5% of Python's. The remaining 0.33 dB is
      explained by the Rust-baseline-vs-Python-baseline drift (1.52 dB
      pre-V5.2; the V5.2 codec recovered 1.19 dB of it).

The 0.1-dB acceptance criterion was missed by 0.23 dB. Per the brief,
this is grounds to keep task #109 `in_progress` rather than auto-close.
See `experiments/v5-2-rust-port/RESULT.md` for the per-phase commits,
debug chain, and the residual two follow-ups (baseline-drift hunt;
optional V5.2 bit-depth bump).

## Phase D — preset + docs (DEFERRED until Phase C closes the last 0.33 dB)

- [ ] New CLI preset `wmv-v52-tight` bundling all flags.
- [ ] Update CANONICAL_11_LEADERBOARD.md with V5.2 column.
- [ ] Update zstd-wrap allow-list / target_glb force list per commit f03e927 precedent.

## Constraints

- NEVER `git add -A`
- NEVER write to /tmp
- Push autonomy authorized
- Multi-day work; commit per-phase
