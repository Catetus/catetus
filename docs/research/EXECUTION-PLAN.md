# 3DGS Frontier Research Plan — May 2026

**Goal**: systematically evaluate the 15-item menu from 4 cross-LLM deep research
responses (`catetus-private/research/llm-brainstorm-*` for the raw inputs)
and integrate the winning techniques into the Catetus stack. Each item is
either *built*, *killed with measured reason*, or *parked with a re-eval
trigger*. Nothing dies of neglect.

## Tracks

- **Track A — Compression** (Modal A100 + local CPU). Push lossless attribute
  compression past the v0.1 hash-grid hyperprior plateau (currently ~4-5×).
- **Track B — Rendering** (4090 over Tailscale + WebGPU bench). Push 10 M
  splats from 6.5 fps → 60 fps in the browser.
- **Track C — Profiling + audit** (local). Real timing data and code audits
  that gate which Track A/B candidates actually land on bottlenecks.
- **Track D — Streaming + perceptual** (cross-cutting). UX-shape wins that
  compound with whatever compression + rendering ship.

## Execution discipline

1. Every experiment appends to `catetus-private/research/EXECUTION-LOG.md`
   in the format `## YYYY-MM-DD <track>.<n> <name> — <BUILT|KILL|PARK|RUNNING>`
   followed by *Hypothesis / Method / Result / Decision*.
2. Modal budget cap per experiment: **$5** unless the previous experiment in
   the same track already paid off. Hard ceiling: **$50/week** across all
   tracks.
3. 4090 GPU is single-tenant; only one Track B experiment at a time
   (see memory: `feedback_serialize_4090_gpu_tasks`).
4. Stealth: no third-party `gh issue create`. Anything that names humans,
   partners, or outreach lives in `catetus-private/`.
5. When a candidate is killed, the kill is permanent unless the *measured*
   reason is invalidated by a later experiment. "We didn't try hard enough"
   does not invalidate a kill.

## First wave (firing now)

| ID  | Track | Item                                              | Effort  | Status |
|-----|-------|---------------------------------------------------|---------|--------|
| A1  | A     | WD-R loss spike (vs MSE on bonsai, Modal)         | 2 weeks | RUNNING |
| A2  | A     | MesonGS++ post-training codec on existing corpus  | 1 week  | RUNNING |
| B1  | B     | Audit `scan_multiblock.wgsl` for spin-wait deadlock | 1 hour | DOING |
| B2  | B     | Drill profile: histogram / scan / scatter sub-stages | 1 day | DOING |
| B3  | B     | WebSplatter design memo + WGSL port spec          | 3 days  | RUNNING |
| C1  | C     | Scaffold-GS readiness audit (HAC++ prereq)        | 1 day   | RUNNING |

## Sequenced wave 2 (fires when wave 1 returns)

| ID  | Track | Item                                              | Trigger |
|-----|-------|---------------------------------------------------|---------|
| A3  | A     | Scaffold-GS retrain on bonsai + HAC++ context     | C1 done AND A1 result |
| A4  | A     | CodecGS feature-plane prototype                   | A2 done (sequencing on Modal budget) |
| B4  | B     | WebSplatter WGSL port (radix sort replacement)    | B3 done |
| B5  | B     | StopThePop hierarchical resort port               | B4 done |
| D1  | D     | PCGS progressive bitstream design                 | A3 done |

## Wave 3 (frontier)

| ID  | Track | Item                                              | Trigger |
|-----|-------|---------------------------------------------------|---------|
| A5  | A     | GeoHCC graph-signal-processing eval               | A3 BUILT |
| A6  | A     | Feed-Forward Long-Context (zero per-scene train)  | A3 BUILT or KILL |
| A7  | A     | NeuralGS per-cluster tiny MLPs                    | A3 result |
| B6  | B     | LODGE hierarchical LOD pipeline                   | B5 BUILT |
| D2  | D     | 3DoF+Quantization for catetus.com/explore      | A3 BUILT |

## Killed before starting (per DRA consensus)

- ❌ Larger MLP / wider hash-grid hyperprior (saturates by ~5×)
- ❌ Diffusion priors directly on splat parameters (no published 3DGS work)
- ❌ Naive CUDA tile-raster → WebGPU without sort redesign (deadlock)
- ❌ Uniform low-bit quant ≤6-bit (-6.74 dB on outdoor scenes)
- ❌ Latent VAE on attribute vector (dominated by HAC++ in 3DGS.zip)
- ❌ MPEG G-PCC traditional point-cloud codecs (no view dependence)

## Living log

All experiment results: `catetus-private/research/EXECUTION-LOG.md`.
Raw DRA responses: `catetus-private/research/llm-brainstorm-*.md`.
