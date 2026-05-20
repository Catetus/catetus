#![deny(clippy::all)]
//! Optimization-pass framework for Catetus. See `specs/0006-optimization-passes.md`.

pub mod lod_merge;
pub mod passes;
pub mod pipeline;
pub mod presets;
pub mod splat_delta;
pub mod tileset;
pub mod vq_palette;

/// Re-export of the V5.2 sidecar codec, which physically lives in
/// `catetus-gltf` (the GLB reader needs to apply the residuals at
/// decode time). Optimize re-exports it under the historical
/// `catetus_optimize::v5_tail` path so the encoder side and its
/// tests keep their original module location.
pub use catetus_gltf::v5_tail;

pub use lod_merge::{LodMergeV4, LodMergeV4Stats};
pub use passes::{
    take_last_dc_quant_table, take_last_rotation_quant_table, take_last_rotation_smallest3_table,
    take_last_sh_rest_quant_table, AspectRatioPrune, BuildLOD, BundleNeighbors, CodecGSKind,
    CodecGSLite, DcQuantTable, FloaterPrune, MortonSort, ObjectAwarePruneExperimental,
    OpacityPrune, Pass, PassContext, PassStats, QuantizeDCPacked, QuantizePosition,
    QuantizeRotation, QuantizeRotationPacked, QuantizeRotationSmallest3, QuantizeSHRest,
    QuantizeScale, RDPrune, ReduceSHDegree, RemoveInvalidSplats, RotationQuantTable,
    RotationSmallest3Table, SHDCTQuantize, ShRestQuantTable,
};
pub use pipeline::{Pipeline, PipelineReport};
pub use presets::preset;
pub use splat_delta::{take_last_delta_stream, DeltaStreamBlob, SplatDelta, SplatDeltaStats};
pub use tileset::{write_tileset, TileReport, TilesetOpts, TilesetReport};
pub use vq_palette::{
    take_last_sh_rest_palette, ShRestPaletteSidetable, VQPaletteShRest, VQPaletteShRestStats,
    VQ_SH_REST_DIM,
};
