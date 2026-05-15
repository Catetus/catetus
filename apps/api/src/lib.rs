//! Library surface for `splatforge-api`.
//!
//! The crate is primarily a binary (`src/main.rs`), but exposing a thin
//! library makes the billing + store internals reachable from integration
//! tests under `tests/`. Only the modules that have stable, test-relevant
//! APIs are re-exported; the HTTP handlers stay private to `main.rs`.
//!
//! Tests in `tests/billing.rs` rely on this surface to exercise the
//! no-double-charge invariant against a Stripe-mock server without
//! spinning up the whole Axum app.

pub mod billing;
pub mod checkout;
pub mod modal_client;
pub mod store;
