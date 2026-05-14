# SPEC-0006 — Optimization Pass Framework

**Status:** Implemented (Phase 1)
**Crate:** `splatforge-optimize`

## Goal

Composable optimization pipeline. Each pass is independently testable, reports stats, and is toggled through pipeline config.

## Pass interface

```rust
pub trait Pass {
    fn name(&self) -> &'static str;
    fn run(&self, scene: &mut SplatScene, ctx: &mut PassContext) -> Result<PassStats>;
}

pub struct PassStats {
    pub removed: usize,
    pub modified: usize,
    pub duration_ms: u64,
    pub notes: Vec<String>,
}
```

## Initial passes

| Order | Pass                            | Phase | Config |
| ----- | ------------------------------- | ----- | ------ |
| 1     | `RemoveInvalidSplats`           | P1    | (none) |
| 2     | `OpacityPrune`                  | P1    | `threshold: f32 = 0.01` |
| 3     | `FloaterPrune`                  | P1    | `kNeighbors`, `distSigma` |
| 4     | `QuantizePosition`              | P1    | `bits: u8 = 16` |
| 5     | `QuantizeScale`                 | P1    | `bits: u8 = 8` |
| 6     | `QuantizeRotation`              | P1    | `bits: u8 = 8` |
| 7     | `ReduceSHDegree`                | P1    | `targetDegree: u8` |
| 8     | `MortonSort`                    | P1    | (none) |
| 9     | `BuildLOD`                      | P2    | `levels: Vec<f32>` |
| 10    | `ObjectAwarePruneExperimental`  | P2    | `protectLabels: Vec<String>` |

## Presets

| Name | Passes |
| ---- | ------ |
| `lossless-repack` | RemoveInvalidSplats, MortonSort |
| `web-mobile` | RemoveInvalidSplats, OpacityPrune(0.02), FloaterPrune, QuantizePosition(15), QuantizeScale(8), QuantizeRotation(8), ReduceSHDegree(0), MortonSort, BuildLOD([0.5, 0.25]) |
| `web-desktop` | similar but bits=16, SH=1 |
| `quest-browser` | aggressive: bits=14, SH=0, BuildLOD([0.3]) |
| `visionos-preview` | mid: bits=15, SH=0 |
| `thumbnail-preview` | aggressive: bits=12, opacity=0.05 |
| `quality-max` | RemoveInvalidSplats, MortonSort only |
| `size-min` | aggressive everything |

## Acceptance tests

```gherkin
Feature: Optimization passes

Scenario: Opacity pruning removes low-opacity splats
  Given a synthetic scene with 100 splats
  And 20 splats have opacity below threshold
  When I run OpacityPrune with threshold 0.01
  Then output has 80 splats
  And the pass report says removed = 20

Scenario: Morton sort is deterministic
  Given the same scene twice
  When I run MortonSort
  Then output ordering is identical

Scenario: ObjectAwarePrune preserves protected labels
  Given a scene with semantic label "product" on 10 splats
  When I run ObjectAwarePruneExperimental
  Then no splats labeled "product" are removed
```
