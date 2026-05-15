//! Route modules split out from `main.rs` so each surface can grow its own
//! validators, errors, and integration tests without bloating the bin crate.
//!
//! Currently only `import` lives here; future routes (e.g. dataset registry,
//! presigned-URL re-issue) will follow the same shape.

pub mod import;
pub mod pricing;
pub mod sdk_license;
