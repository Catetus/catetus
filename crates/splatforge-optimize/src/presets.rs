//! Built-in optimization presets (SPEC-0006).

use anyhow::{anyhow, Result};

use crate::passes::{
    AspectRatioPrune, BuildLOD, FloaterPrune, MortonSort, OpacityPrune, QuantizePosition,
    QuantizeRotation, QuantizeScale, ReduceSHDegree, RemoveInvalidSplats,
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
        other => return Err(anyhow!("unknown preset '{other}'")),
    };
    Ok(pipe)
}
