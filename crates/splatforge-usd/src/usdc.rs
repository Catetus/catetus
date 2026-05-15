//! Placeholder for the USDC (binary crate) writer/reader.
//!
//! Stub kept in tree to satisfy `pub mod usdc` while the bit-exact writer
//! is delivered by the in-flight `usdc-bit-exact` agent. Callers get
//! `UsdError::Malformed` on either entry point.

use std::path::Path;

use splatforge_core::SplatScene;

use crate::{UsdError, UsdWriteOpts};

/// Write `scene` to `path` as USDC. Not yet implemented.
pub fn write_usdc(_scene: &SplatScene, _path: &Path, _opts: &UsdWriteOpts) -> Result<(), UsdError> {
    Err(UsdError::Malformed("usdc writer not yet implemented".into()))
}

/// Read a USDC file. Not yet implemented.
pub fn read_usdc(_path: &Path) -> Result<SplatScene, UsdError> {
    Err(UsdError::Malformed("usdc reader not yet implemented".into()))
}
