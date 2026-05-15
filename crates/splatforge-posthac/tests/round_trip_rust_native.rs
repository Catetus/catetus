//! Rust-native round-trip: feed Rust predict_all() into BOTH encode and
//! decode and verify bit-exact symbol recovery.
//!
//! This avoids the cross-language ULP drift entirely — Python is only
//! ever responsible for *training the hyperprior weights*. Both encoding
//! and decoding are Rust-side, sharing the same `predict()` function.

use splatforge_posthac::*;
use std::env;
use std::fs;
use std::path::PathBuf;

#[test]
fn rust_native_round_trip() {
    let Ok(path) = env::var("SPLATFORGE_PYTHON_PTHC") else {
        eprintln!("SPLATFORGE_PYTHON_PTHC not set; skipping");
        return;
    };
    let bytes = fs::read(PathBuf::from(path)).expect("read .pthc");
    let container = read_container(&bytes).expect("parse container");
    let n = container.header.n as usize;
    let d = container.header.d as usize;
    eprintln!("[rust-roundtrip] N={n} D={d}");

    let predictions = predict_all(
        &container.positions,
        container.header.pos_mn,
        container.header.pos_mx,
        &container.header.config,
        &container.weights,
    );
    eprintln!(
        "[rust-roundtrip] computed predict_all() for {} splats",
        predictions.len()
    );

    // Encode some synthetic codes deterministically (we don't have the
    // original 8-bit codes in this test — they're inside Python's
    // bitstream which we can't replay symbol-by-symbol here without
    // the input PLY). Use a reproducible pseudo-random stream instead.
    let mut codes = vec![0u8; n * d];
    let mut seed: u32 = 1234567;
    for i in 0..codes.len() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        codes[i] = ((seed >> 16) & 0xff) as u8;
    }

    let compressed = encode_codes(&codes, n, d, &predictions).expect("rust encode");
    eprintln!(
        "[rust-roundtrip] encoded → {} u32 ({:.2} MB)",
        compressed.len(),
        compressed.len() as f32 * 4.0 / 1e6
    );

    let decoded = decode_codes(&compressed, n, d, &predictions).expect("rust decode");
    eprintln!("[rust-roundtrip] decoded {} bytes", decoded.len());
    assert_eq!(decoded.len(), codes.len());
    for (i, (&a, &b)) in codes.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(a, b, "mismatch at byte {i}: encoded {a} ≠ decoded {b}");
    }
    eprintln!(
        "[rust-roundtrip] BIT-EXACT round-trip on {} symbols",
        codes.len()
    );
}
