#![deny(clippy::all)]
//! Optimization-pass framework for SplatForge. See `specs/0006-optimization-passes.md`.

pub mod passes;
pub mod pipeline;
pub mod presets;
pub mod tileset;

pub use passes::{
    AspectRatioPrune, BackgroundOverdrawPrune, BuildLOD, FloaterPrune, MortonSort,
    ObjectAwarePruneExperimental, OpacityPrune, Pass, PassContext, PassStats, QuantizePosition,
    QuantizeRotation, QuantizeScale, ReduceSHDegree, RemoveInvalidSplats, SubjectCrop,
};
pub use pipeline::{Pipeline, PipelineReport};
pub use presets::preset;
pub use tileset::{write_tileset, TileReport, TilesetOpts, TilesetReport};
