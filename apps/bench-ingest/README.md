SplatBench Ingest Harness
=========================

Automated pipeline that turns a `(scene_id, PLY URL, license)` triple into a
row of measured compression ratios for the SplatBench leaderboard. Runs as
either a local CLI or a Modal entrypoint.

Status: corpus expansion for SplatBench v2 (real-photo corpus, 8-10 scenes).

What it measures
----------------

For each input scene the harness records:

- `bytesIn` — size of the source `.ply` on disk.
- `splatCount`, `shDegree`, `hash` — taken from `splatforge analyze` (BLAKE3
  IR hash, matches v0 schema).
- `analyzeMs` — wall-clock of the analyze pass.
- For each preset in `--presets` (default: `web-mobile size-min web-desktop
  quest-browser`), the output bytes and `bytesIn / outBytes` ratio.

What it does NOT measure (v2 scope)
-----------------------------------

- Per-camera render PSNR. The v0 schema has a separate `fidelity` block that
  is populated by a different runner (`benches/run-fidelity.mjs` /
  `splatforge fidelity`). v2 ingest deliberately keeps the two paths
  decoupled — ratios first, fidelity later, so the cheap-to-measure
  compression numbers don't gate on the GPU-bound rendering pass.

Idempotence
-----------

Re-running on the same `(scene_id, preset)` is a no-op if the existing JSON
already contains a matching numeric row AND the input file's size + mtime
match the cached values. Pass `--force` to remeasure.

CLI usage
---------

```bash
# Local, single scene:
python -m bench_ingest.cli measure \
    --scene-id stump_mipnerf360_iter7k \
    --ply benches/scenes/real/stump_iter7000.ply \
    --license "MipNeRF360, CC-BY-4.0" \
    --class outdoor-scene \
    --origin "MipNeRF360 / dylanebert HF mirror"

# Batch from a manifest:
python -m bench_ingest.cli batch \
    --manifest benches/scenes/real/manifest.json
```

Modal usage
-----------

```bash
modal run apps/bench-ingest/src/bench_ingest/modal_app.py::measure_remote \
    --scene-id counter_mipnerf360_iter7k \
    --ply-url "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/counter/point_cloud/iteration_7000/point_cloud.ply"
```

The Modal entrypoint reuses the existing `splatforge-worker` image base (Rust
toolchain + pinned `splatforge` CLI) so we don't pay a fresh image build per
scene. Results land in the `splatbench-ingest` Modal Volume and are pulled to
local disk via `bench_ingest.cli pull`.

Output
------

Single JSON file: `benches/reports/splatbench-v2.json`. Schema is the v0
schema plus a `presetRuns` map keyed by preset name. The `sync-splatbench.mjs`
script merges this into the published `benches/reports/splatbench-v0.json`
without clobbering human-annotated columns (ML pro/premium scores, repack).
