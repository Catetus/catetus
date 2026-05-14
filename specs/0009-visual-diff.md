# SPEC-0009 — Visual Diff Harness

**Status:** Implemented (Phase 2)
**Package:** `tests/visual`

## Goal

Compare source and optimized outputs using deterministic camera paths. Generate machine-readable JSON + human-readable HTML reports.

## Inputs

```bash
splatforge diff before.ply after.gltf \
  --camera-path orbit-8 \
  --frame-size 512x512 \
  --threshold 0.03 \
  --out reports/diff/
```

## Metrics

* per-frame pixel L1
* per-frame perceptual `ΔE94` via OKLab
* aggregate: `max`, `mean`, `p95`
* PSNR & SSIM (when feasible)

## Output

```
reports/diff/
  diff.json
  diff.html
  frames/
    before/0001.png ... 0008.png
    after/0001.png  ... 0008.png
    diff/0001.png   ... 0008.png   (overlay)
```

## Acceptance tests

```gherkin
Feature: Visual diff

Scenario: Identical scene has near-zero diff
  Given the same scene as before and after
  When I run visual diff
  Then max frame diff is below epsilon

Scenario: Aggressive pruning fails quality threshold
  Given a scene optimized with size-min preset
  When I run visual diff
  Then the report marks quality as failed if threshold is exceeded

Scenario: HTML report includes before and after frames
  Given a completed visual diff
  Then report.html includes side-by-side frames
```
