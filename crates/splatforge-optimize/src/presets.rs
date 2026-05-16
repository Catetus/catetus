//! Built-in optimization presets (SPEC-0006).

use anyhow::{anyhow, Result};

use crate::passes::{
    AspectRatioPrune, BackgroundOverdrawPrune, BuildLOD, FloaterPrune, MortonSort, OpacityPrune,
    QuantizePosition, QuantizeRotation, QuantizeScale, ReduceSHDegree, RemoveInvalidSplats,
};
use crate::pipeline::Pipeline;

/// All recognized preset names.
pub const PRESETS: &[&str] = &[
    "lossless-repack",
    "web-mobile",
    "web-desktop",
    "quest-browser",
    "visionos-preview",
    "thumbnail-preview",
    "quality-max",
    "size-min",
    "geospatial",
    "hero",
    "hero-quality",
];

/// Build a `Pipeline` from a named preset.
pub fn preset(name: &str) -> Result<Pipeline> {
    let pipe = match name {
        "lossless-repack" | "quality-max" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(MortonSort)),
        // `web-mobile`: default web target. AspectRatioPrune (max_ratio=10)
        // drops Inria-3DGS needle splats before quantization so 12-bit scale
        // /rotation quant can't visibly snap thin gaussians into spikes.
        // Threshold 10.0 is slightly looser than the 8.0 prototype default
        // because mobile bandwidth budget rewards keeping a few moderately
        // anisotropic detail splats; 12-bit quant absorbs the residual risk.
        "web-mobile" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale { bits: 12 }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "web-desktop" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 16 }))
            .push(Box::new(QuantizeScale { bits: 12 }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 1 }))
            .push(Box::new(MortonSort)),
        "quest-browser" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.03 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 14 }))
            .push(Box::new(QuantizeScale { bits: 12 }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD { levels: vec![0.3] })),
        "visionos-preview" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale { bits: 8 }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        "thumbnail-preview" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.05 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 12 }))
            .push(Box::new(QuantizeScale { bits: 12 }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        // `geospatial`: produces a Cesium 3D Tiles 1.1 tileset (tileset.json +
        // per-LOD GLBs) when paired with the tileset emitter. The pipeline
        // itself is the `web-mobile` baseline (prune + quantize + Morton),
        // followed by a four-level LOD pyramid where each LOD halves the splat
        // count of the previous one: LOD0=1.0, LOD1=0.5, LOD2=0.25, LOD3=0.125.
        // The downstream tileset writer maps each LOD to a tile with halved
        // `geometricError` and a `REPLACE` refinement chain, matching Cesium's
        // screen-space-error model.
        "geospatial" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale { bits: 8 }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25, 0.125],
            })),
        "size-min" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.05 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 12 }))
            .push(Box::new(QuantizeScale { bits: 12 }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.25, 0.1],
            })),
        // `hero`: marketing-hero / showcase preset. Optimized for visual
        // quality on a single above-the-fold slow-orbit scene, not for raw
        // byte size. No opacity prune, no aspect-ratio prune, no LOD chain,
        // 16-bit scale quant (vs 8-bit elsewhere) so anisotropic detail
        // splats render without spike artifacts. SH reduced to degree 0 only
        // because the current viewer renders DC-only; bump target_degree
        // back to 1+ once `packages/viewer` learns view-dependent shading.
        "hero" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 16 }))
            .push(Box::new(QuantizeScale { bits: 16 }))
            .push(Box::new(QuantizeRotation { bits: 16 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        // `hero-quality`: marketing-hero / showcase preset tuned for tight
        // framing on a single subject (e.g. the bonsai). Drops the
        // low-opacity isotropic background ring that dominates fillrate at
        // hero framing (the post-mortem from the previous rebuild traced
        // jank + dim-blob artefacts to fillrate from those splats, not the
        // anisotropic needles that AspectRatioPrune handles).
        //
        // Order is deliberate:
        //   1. RemoveInvalidSplats — strip NaN/Inf before any geometric work.
        //   2. OpacityPrune(0.04) — drop the faintest splats that pure
        //      overdraw cost can't justify.
        //   3. FloaterPrune — k-NN isolation pass for sparse-densification halo.
        //   4. AspectRatioPrune(8.0) — kills the Inria needle artefacts.
        //   5. BackgroundOverdrawPrune — drops the top 5% by screen-coverage
        //      cost when also faint (opacity < 0.5). This is the new pass
        //      that handles the background ring specifically.
        //   6-8. Quant: 14-bit position / 12-bit scale / 12-bit rotation.
        //      12-bit scale/rot matches the spike-fix from the AspectRatioPrune
        //      commit; 14-bit position because the camera is tight and we
        //      benefit from a hair more spatial precision than web-desktop.
        //   9. ReduceSHDegree(1) — viewer has SH-1 since dd3dae9.
        //   10. MortonSort — render order for tile-based viewer.
        "hero-quality" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.04 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(AspectRatioPrune { max_ratio: 8.0 }))
            .push(Box::new(BackgroundOverdrawPrune::default()))
            .push(Box::new(QuantizePosition { bits: 14 }))
            .push(Box::new(QuantizeScale { bits: 12 }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 1 }))
            .push(Box::new(MortonSort)),
        other => return Err(anyhow!("unknown preset '{other}'")),
    };
    Ok(pipe)
}
