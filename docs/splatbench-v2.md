# SplatBench v2 — Real-Photo Corpus and Ingest Harness

SplatBench v2 expanded the real-photo corpus from 3 scenes to 8 and introduced
an automated ingest harness. The corpus expansion landed in PR #5
(`bench: ingest cluster.fly LOD ladder (5 scenes, CC-BY-4.0) into SplatBench v2`)
which added the Dany Bittel cluster_fly LOD ladder. This document covers the
companion harness in `apps/bench-ingest/` that turns a `(scene_id, PLY URL,
license)` triple into a row of measured ratios — the tool used to land
scene 9, 10, … without bespoke one-off scripts.

## Current real-photo corpus (post-PR #5)

| ID                            | Class              | License                       | Splats     |
| ----------------------------- | ------------------ | ----------------------------- | ---------- |
| `bonsai_mipnerf360_iter7k`    | indoor-real-estate | Mip-NeRF 360 · CC-BY-4.0      | 1,157,141  |
| `bicycle_mipnerf360_iter7k`   | outdoor-scene      | Mip-NeRF 360 · CC-BY-4.0      | 3,616,103  |
| `stump_mipnerf360_iter7k`     | outdoor-scene      | Mip-NeRF 360 · CC-BY-4.0      | 3,807,536  |
| `cluster_fly_s`               | indoor-close-up    | Dany Bittel · CC-BY-4.0       | 25,627     |
| `cluster_fly_m`               | indoor-close-up    | Dany Bittel · CC-BY-4.0       | 145,617    |
| `cluster_fly_l`               | indoor-close-up    | Dany Bittel · CC-BY-4.0       | 301,958    |
| `cluster_fly_xl`              | indoor-close-up    | Dany Bittel · CC-BY-4.0       | 624,180    |
| `cluster_fly_xxl`             | indoor-close-up    | Dany Bittel · CC-BY-4.0       | 3,506,799  |

The cluster_fly LOD ladder is particularly valuable because all 5 scenes share
one subject — the leaderboard can isolate "how do ratios scale with splat
count" from "how do they scale with subject," a question the prior corpus
could not answer with a single capture.

## What `apps/bench-ingest/` does

For each input scene the harness records:

- `bytesIn` — size of the source `.ply` on disk.
- `splatCount`, `shDegree`, `hash` — taken from `splatforge analyze` (BLAKE3
  IR hash, matches the v0 schema).
- `analyzeMs` — wall-clock of the analyze pass.
- For each preset in `--presets` (default: `web-mobile size-min web-desktop
  quest-browser`), the SPZ output bytes and `bytesIn / spzBytes` ratio.

The published v0 numbers (e.g. bonsai web-mobile = 22.81×) measure SPZ-packed
output, not the intermediate glTF + sidecar buffers, so the harness runs the
full two-step pipeline:

1. `splatforge optimize --preset <name>` → glTF manifest + sidecar buffers.
2. `splatforge convert --to spz` → single `.spz` file.

Reported `outputBytes` is the `.spz` file size. Wall time covers both steps
so it's comparable to the worker's end-to-end latency.

## What it does NOT measure (v2 scope)

- Per-camera render PSNR — that's the `splatforge fidelity` path, which
  runs a deterministic 8-orbit camera sweep through `@splatforge/viewer`
  and records ΔE94 / SSIM / pixelmatch. v2 keeps this separate so the
  cheap-to-measure ratio numbers don't gate on the GPU-bound render pass.
- Decode wall-clock — interesting but architecture-dependent; tracked as
  a follow-up.

## CLI usage

```bash
# Local, single scene:
python -m bench_ingest.cli measure \
    --scene-id stump_mipnerf360_iter7k \
    --ply benches/scenes/real/stump_iter7000.ply \
    --license "Mip-NeRF 360 · CC-BY-4.0" \
    --class outdoor-scene \
    --origin "MipNeRF360 / dylanebert HF mirror"

# Batch from a manifest:
python -m bench_ingest.cli batch \
    --manifest benches/scenes/real/manifest.json
```

## Modal usage

```bash
modal run apps/bench-ingest/src/bench_ingest/modal_app.py::measure_remote \
    --scene-id <id> \
    --ply-url   <https://…/point_cloud.ply> \
    --license   "<spdx-ish>" \
    --class     <capture-class> \
    --origin    <attribution>
```

The Modal entrypoint reuses the existing `splatforge-worker` image base
(Rust toolchain + pinned `splatforge` CLI) so we don't pay a fresh image
build per scene. Results land in the `splatbench-ingest` Modal Volume and
are pulled with `modal volume get splatbench-ingest …`.

## Idempotence

Re-running on a `(scene_id, preset)` pair is a no-op if the existing JSON
already contains a successful row for it. Pass `--force` to remeasure. This
makes it cheap to keep the harness in CI without paying for repeat work.

## Preset selection

The CLI's currently-registered presets the harness exercises by default:

- `web-mobile` — primary web target.
- `size-min` — maximum compression at moderate fidelity cost.
- `web-desktop` — middle ground; less quantization than mobile.
- `quest-browser` — Quest 3 / 3S target; preserves higher-frequency SH
  bands needed for in-headset close-up.

The newer `codec-gs-mixed` and `fcgs-instant` presets are wired into the
worker dispatch layer (PRs `feat/worker-presets-integration` and
`feat/ship-cull-mixed-defaults`) but exercised through a different code
path. Once they appear in `crates/splatforge-optimize/src/presets.rs` the
harness picks them up by extending `DEFAULT_PRESETS` in
`bench_ingest/measure.py`.

## Sync into the leaderboard

After a measurement run, push the new numbers into the public
leaderboard JSON:

```bash
node scripts/sync-splatbench.mjs
```

The sync is strictly additive on the measurement side: v2 wins on every
ratio / byte / hash field, the v0 JSON wins on every annotation field
(splatforge-pro ML scores, DifferentiableRepack rows, ΔE94 fidelity
blocks). Re-runs are idempotent.

The Astro build picks up the new rows via
`apps/web/scripts/sync-data.mjs`, which copies
`benches/reports/splatbench-v0.json` into the project's `src/data/`
before each build.

## Visual verification

`tests/visual/scripts/shoot-leaderboard.mjs` captures default,
real-only-filter, and size-min-preset states of the `/bench` page via
Playwright. Required reading for any PR that changes the leaderboard
output, per repo memory rule `verify_ui_visually_before_handoff`.

```bash
cd apps/web && pnpm run build && pnpm exec astro preview --port 4321 &
node tests/visual/scripts/shoot-leaderboard.mjs
```

Output lands in `tasks/screenshots/leaderboard/`.

## Reproducibility

Every measured row in `splatbench-v0.json` records the BLAKE3 hash of
the canonical IR (computed by `splatforge analyze`), so a fresh run on
a fresh machine produces byte-identical numbers when:

- The PLY input matches the recorded hash.
- The `splatforge` CLI is built at the recorded git SHA (see
  `splatforgeVersion` at the top of the JSON).
- The `presets.rs` registry has not changed between runs.

If any of those drift, the ratio numbers will drift by a few percent
even on the same scene — this is expected and is the reason the v0
schema records both `splatforgeVersion` and `hash` per row.

The harness has been verified against the published v0 numbers on the
three Mip-NeRF 360 scenes (bonsai, bicycle, stump): reproduced to
within rounding (≤0.01× delta on all 6 measurements).

## Costs

v2 ingest is CPU-only. A full pass over the current 8-scene real corpus
on a Modal `cpu=4, memory=8 GiB` instance runs in well under 15 minutes
of billable time across all four presets — well under $1 at current
Modal prices. There is no A100 spend for v2; that's reserved for the
`fidelity` pass (separate runner, separate budget line).

## Files

New in this changeset:

- `apps/bench-ingest/` — Python harness package + Modal entrypoint.
- `scripts/sync-splatbench.mjs` — idempotent v2 → v0 JSON merge.
- `tests/visual/scripts/shoot-leaderboard.mjs` — Playwright shoot
  script for visual verification.
- `docs/splatbench-v2.md` — this file.

Touched:

- `apps/web/src/components/Leaderboard.astro` — methodology paragraph
  no longer hard-codes the two original Mip-NeRF 360 anchors.
