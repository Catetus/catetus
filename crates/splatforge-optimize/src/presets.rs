//! Built-in optimization presets (SPEC-0006).

use anyhow::{anyhow, Result};

use crate::passes::{
    BuildLOD, FloaterPrune, MortonSort, OpacityPrune, QuantizePosition, QuantizeRotation,
    QuantizeScale, ReduceSHDegree, RemoveInvalidSplats,
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
];

/// Build a `Pipeline` from a named preset.
pub fn preset(name: &str) -> Result<Pipeline> {
    let pipe = match name {
        "lossless-repack" | "quality-max" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(MortonSort)),
        "web-mobile" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale { bits: 8 }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "web-desktop" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 16 }))
            .push(Box::new(QuantizeScale { bits: 8 }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 1 }))
            .push(Box::new(MortonSort)),
        "quest-browser" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.03 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 14 }))
            .push(Box::new(QuantizeScale { bits: 8 }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
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
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 12 }))
            .push(Box::new(QuantizeScale { bits: 8 }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        "size-min" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.05 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 12 }))
            .push(Box::new(QuantizeScale { bits: 8 }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.25, 0.1],
            })),
        other => return Err(anyhow!("unknown preset '{other}'")),
    };
    Ok(pipe)
}
