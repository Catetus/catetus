#![deny(clippy::all)]
//! Optimization-pass framework for SplatForge. See `specs/0006-optimization-passes.md`.

pub mod passes;
pub mod pipeline;
pub mod presets;

pub use passes::{
    BuildLOD, FloaterPrune, MortonSort, ObjectAwarePruneExperimental, OpacityPrune, Pass,
    PassContext, PassStats, QuantizePosition, QuantizeRotation, QuantizeScale, ReduceSHDegree,
    RemoveInvalidSplats,
};
pub use pipeline::{Pipeline, PipelineReport};
pub use presets::preset;
