//! # catetus-tileset — octree LOD tileset encoder
//!
//! Turns a single [`SplatCloud`] into a **streaming-first multi-tile LOD
//! octree**: a spatial tree of nodes, each node carrying progressively-finer
//! tile payloads, plus two interoperable manifests:
//!
//! 1. **`lod-meta.json`** — byte-compatible with SuperSplat's format
//!    (`d28zzqy0iyovbz.cloudfront.net/<id>/v1/lod-meta.json`). This is the
//!    interop target: a tree of `{bound, lods:[{file,count}...], children}`
//!    nodes plus a flat `filenames` array. Lets a SuperSplat-style viewer
//!    fetch root + visible tiles first for ~instant first-paint.
//! 2. **`tileset.json`** — a 3D-Tiles-1.1-shaped manifest (`{asset,
//!    geometricError, root:{boundingVolume, geometricError, refine, content,
//!    children}}`) for Cesium / generic 3D-Tiles tooling and the Catetus
//!    WebGPU viewer.
//!
//! Both describe the *same* octree; they differ only in serialization shape.
//!
//! ## Why this exists (strategic)
//!
//! SuperSplat's competitive moat is **streaming**, not compression ratio. They
//! ship a 7-level LOD octree of ~87 SOG tiles from a CDN; the viewer fetches
//! root + visible tiles first so a 492 MB scene first-paints in ~1 s. Catetus
//! historically shipped a single all-or-nothing GLB blob. This crate closes
//! that gap by producing the multi-tile octree.
//!
//! ## Pipeline
//!
//! ```text
//!   SplatCloud
//!     │  build_octree()      depth-limited spatial subdivision (octree),
//!     │                      each leaf holds its splat indices
//!     ▼
//!   Octree<indices>
//!     │  assign_lods()       per node, pick 1..=LODS_PER_NODE decimated
//!     │                      representative sets (coarse → fine) by
//!     │                      importance (opacity × scale-volume)
//!     ▼
//!   TilesetPlan
//!     │  encode payloads → bytes (TilePayloadCodec; MVP = `.sftile` v1)
//!     │  emit lod-meta.json + tileset.json
//!     ▼
//!   out/
//!     ├── lod-meta.json
//!     ├── tileset.json
//!     └── tiles/<file_index>.sftile
//! ```
//!
//! ## MVP scope (this version)
//!
//! - Real octree subdivision (configurable `max_depth`, `max_splats_per_leaf`).
//! - Importance-weighted decimation for coarse LODs (not just stride).
//! - Both manifests emitted and self-consistent.
//! - Per-tile payloads in a minimal self-describing `.sftile` binary so the
//!   library compiles and round-trips with **zero** dependency on the GLB/SOG
//!   writers (those plug in behind the [`TilePayloadCodec`] trait — see
//!   `STATUS.md` for the wiring task).
//! - No splat is dropped or duplicated across leaves (conservation invariant,
//!   asserted in tests).
//!
//! See `STATUS.md` for what's done and the multi-week roadmap.

mod octree;
mod manifest;
mod codec;
mod glb_codec;
mod shared_palette;
mod plan;

pub use codec::{SfTileCodec, TilePayloadCodec, TileBytes};
pub use glb_codec::{GlbTileCodec, TilePreset};
pub use shared_palette::{
    decode_tile_indices, dc_term, SharedCodebook, SharedPaletteTileCodec,
    SHARED_PALETTE_FILENAME, TILE_INDEX_EXT,
};
pub use manifest::{
    Aabb, LodMeta, LodMetaNode, LodRef, TileNode, TilesetManifest, TilesetAsset,
    BoundingVolume, TileContent,
};
pub use octree::{Octree, OctreeConfig, OctreeNode};
pub use plan::{
    plan_tileset, write_tileset, write_tileset_shared, SharedPaletteConfig, SharedTilesetWritten,
    TilePayload, TilesetConfig, TilesetError, TilesetPlan,
};

/// Default number of LOD levels carried per octree node (matches SuperSplat,
/// which stores exactly 3 `{file,count}` entries per node).
pub const LODS_PER_NODE: usize = 3;
