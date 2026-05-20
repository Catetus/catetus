//! `catetus-viewer-mobile` — shared CPU render core for the native mobile
//! viewers (iOS + Android).
//!
//! # Why this crate exists
//!
//! The web viewer at `packages/viewer/` ships a WebGPU/WebGL2 stack written in
//! TypeScript. On mobile we want the same math but driven by Metal (iOS) and
//! Vulkan/GLES (Android). Rather than re-port the math twice, this crate is
//! the single source of truth for:
//!
//! * Decoding a `.glb` with `KHR_gaussian_splatting` into a CPU vertex buffer
//!   (`SplatVertex`). The bytes are the wire format the GPU will eventually
//!   consume.
//! * Building view / projection matrices from a [`Camera`].
//! * Computing per-splat view-space depth for back-to-front sorting.
//! * Projecting world positions to NDC for the simple point-sprite renderer
//!   that the iOS/Android skeletons paint today.
//!
//! Compute kernels (radix sort, 2D covariance projection, tile binning) live
//! in the platform shaders (`Shaders/*.metal`, `*.glsl`). This crate is the
//! oracle they must match — see `tests/glb_roundtrip.rs`.
//!
//! # C ABI
//!
//! The functions in [`ffi`] are `extern "C"` and re-exported into a header by
//! `scripts/regen-headers.sh` (cbindgen). They are deliberately small and
//! pointer-based so Swift and Kotlin can call them with zero glue.

#![deny(clippy::all)]
#![warn(missing_docs)]

pub mod camera;
pub mod decode;
pub mod ffi;
pub mod math;
pub mod sort;
pub mod vertex;

pub use camera::Camera;
pub use decode::{decode_glb_bytes, DecodeError};
pub use sort::sort_by_depth;
pub use vertex::SplatVertex;
