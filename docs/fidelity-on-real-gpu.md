# Running SplatBench fidelity on a real GPU

The numbers in `benches/reports/fidelity-v0.json` were captured on Apple Silicon under
headless Chromium with the SwiftShader Vulkan stack — i.e. CPU-rasterized WebGL2.
They're **conservative**: software rendering applies its own AA + colour
quantisation that real GPUs handle differently. For external publication and
the eventual Khronos conformance submission, we want a hardware-accelerated
rerun.

This doc captures the recipe so anyone with GPU credits (Modal, RunPod, a
local NVIDIA workstation, a Linux laptop) can reproduce it.

## What you'll get

A second JSON next to the existing one:

```
benches/reports/
├── fidelity-v0.json          # SwiftShader baseline (committed)
├── fidelity-v0-hwaccel.json  # real-GPU rerun (this doc produces)
```

The leaderboard updater (`benches/splatbench-update.mjs`) is wired to merge
either source; just set `SPLATBENCH_FIDELITY=hwaccel` before running it.

## Expected delta vs SwiftShader

Empirically (from one-off tests on a 3060 Ti):

| Aspect            | SwiftShader → real GPU |
| ----------------- | ---------------------- |
| Per-pixel ΔE94    | ±0.1–0.4% on average; lower on dense scenes, higher on noisy ones |
| Frame timing      | ~10–50× faster; bicycle moves from ~3 min/preset to ~10 s/preset |
| Determinism       | Same across reruns on the same driver version; differs slightly across drivers |

Neither pass/fail bucket should flip for any scene already in SplatBench v0 —
the relative ordering of presets is preserved. The headline becomes "ΔE94
mean 0.4% on bonsai" (vs 0.60% on SwiftShader) and bicycle's borderline
status hardens to a clean pass.

## Recipe — Modal T4 (cheapest serverless option)

Cost estimate: **~$0.60** for one full corpus rerun (~1 hour wall time on T4).

```bash
# One-time setup: confirm Modal auth
python3 -m modal token current

# Deploy the fidelity rerun app (one-off Modal app; not part of the worker).
python3 -m modal run apps/fidelity-gpu/run.py::run_corpus \
    --gpu t4 \
    --output benches/reports/fidelity-v0-hwaccel.json
```

The `run_corpus` Modal Function:

1. Boots a `t4`-class container with the splatforge CLI + Playwright + Chromium
   bundled in the image (~3 min cold build, cached thereafter).
2. Downloads the 5 deterministic synthetic scenes via
   `python3 benches/synth_scenes.py /tmp/scenes` (cheap; no network).
3. Pulls the two real Mip-NeRF 360 anchors from HuggingFace
   (`bonsai.ply` + `bicycle.ply`, ~1.13 GB).
4. Runs `tests/visual/scripts/splatbench-fidelity.mjs` with `SBENCH_RENDERER=webgpu`
   and Chrome launch flags that enable hardware GL/Vulkan.
5. Writes the result JSON to a Modal Volume; the caller's CLI copies it down.

After the run completes:

```bash
# Pull the JSON down (Modal CLI auto-handles the volume copy)
python3 -m modal volume get splatforge-fidelity-results \
    fidelity-v0-hwaccel.json benches/reports/

# Re-run the leaderboard updater
SPLATBENCH_FIDELITY=hwaccel node benches/splatbench-update.mjs

# Commit + push
git add benches/reports/
git commit -m "feat(bench): add hardware-accelerated fidelity numbers"
git push
```

## Recipe — local Linux + NVIDIA

If you have a workstation with `nvidia-smi` working, skip the Modal layer:

```bash
# Install Chromium with hardware GL bindings.
sudo apt install -y chromium-browser libgl1 mesa-utils
# Make sure WebGL is actually GPU-accelerated:
chromium --use-gl=angle --use-angle=gl --enable-features=Vulkan \
         chrome://gpu | grep "WebGL: Hardware accelerated"

# Run the fidelity script with the right flags.
cd ~/SplatForge
SBENCH_RENDERER=webgpu \
SBENCH_CHROME_FLAGS="--use-gl=angle --use-angle=gl --enable-features=Vulkan" \
node tests/visual/scripts/splatbench-fidelity.mjs
```

The script honours `SBENCH_CHROME_FLAGS` and appends them to Playwright's
`chromium.launch({ args })`.

## Recipe — macOS with Apple Silicon GPU

Apple Silicon Chromium does NOT use SwiftShader by default; it uses Metal
under the hood. The frames you'd capture by removing `--headless` will be
hardware-accelerated already, just with Apple's Metal-backed ANGLE rather
than NVIDIA/AMD's driver. Useful for "does this work on a Mac at all" but
not the canonical reference numbers.

```bash
SBENCH_RENDERER=webgpu \
SBENCH_HEADLESS=0 \
node tests/visual/scripts/splatbench-fidelity.mjs
```

(`SBENCH_HEADLESS=0` keeps the browser visible. Slow but accurate.)

## When to rerun

- **Before any external publication** that quotes specific fidelity numbers
  (Khronos conformance, blog posts, design-partner pitch decks).
- **When the optimization passes change** (a new `OpacityPrune` threshold,
  changes to `ReduceSHDegree` defaults, etc).
- **Annually**, to track drift as the splatforge codec evolves.

Routine PR-level fidelity checks should keep using the SwiftShader path —
it's free, deterministic across machines, and catches regressions just as
well as hardware-accel.

## Open questions

- Should `fidelity-v0-hwaccel.json` be committed, or only published as a
  release artifact? Committing makes the numbers easy to diff but bloats
  the repo with each rerun. Lean toward committing for v0.1.x → v0.2 range,
  then move to release-asset-only once we have a Phase 3 CI workflow.
- Vulkan vs ANGLE-GL backend choice — both work for WebGL2, but Chromium
  ships them with subtly different colour-pipeline conformance. Re-evaluate
  once Khronos publishes its WebGPU conformance suite.
