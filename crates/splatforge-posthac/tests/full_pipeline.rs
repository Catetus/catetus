//! Full Rust encode → decode pipeline on real Python-trained weights + real
//! 8-bit codes from a scene. This is the production validation: bit-exact
//! round-trip across the entire codec.
//!
//! Inputs:
//!   SPLATFORGE_PYTHON_PTHC = a .pthc file with hyperprior weights + positions
//!   SPLATFORGE_RAW_CODES   = a binary dump of the raw 8-bit codes
//!                           layout: u32 n, u32 d, then n*d uint8 codes (row-major)

use byteorder::{LittleEndian, ReadBytesExt};
use splatforge_posthac::*;
use std::env;
use std::fs;
use std::io::{Cursor, Read};
use std::path::PathBuf;

#[test]
fn full_pipeline_round_trip() {
    let Ok(pthc_path) = env::var("SPLATFORGE_PYTHON_PTHC") else {
        return;
    };
    let Ok(codes_path) = env::var("SPLATFORGE_RAW_CODES") else {
        return;
    };

    let bytes = fs::read(PathBuf::from(pthc_path)).expect("read .pthc");
    let container = read_container(&bytes).expect("parse container");
    let n = container.header.n as usize;
    let d = container.header.d as usize;
    eprintln!("[pipeline] container N={n} D={d}");

    // Read raw codes
    let codes_bytes = fs::read(PathBuf::from(codes_path)).expect("read codes");
    let mut cur = Cursor::new(&codes_bytes);
    let n_codes = cur.read_u32::<LittleEndian>().unwrap() as usize;
    let d_codes = cur.read_u32::<LittleEndian>().unwrap() as usize;
    assert_eq!(n, n_codes, "container N != codes N");
    assert_eq!(d, d_codes, "container D != codes D");
    let mut raw_codes = vec![0u8; n * d];
    cur.read_exact(&mut raw_codes).unwrap();
    eprintln!("[pipeline] loaded raw codes: {} bytes", raw_codes.len());

    // Rust predict() for every splat
    let predictions = predict_all(
        &container.positions,
        container.header.pos_mn,
        container.header.pos_mx,
        &container.header.config,
        &container.weights,
    );

    // Rust encode
    let compressed = encode_codes(&raw_codes, n, d, &predictions).expect("encode");
    let raw_bytes = (n * d) as f32;
    let comp_bytes = (compressed.len() * 4) as f32;
    eprintln!(
        "[pipeline] encoded: {:.2} MB raw → {:.2} MB compressed ({:.2}× over 8-bit)",
        raw_bytes / 1e6,
        comp_bytes / 1e6,
        raw_bytes / comp_bytes
    );

    // Rust decode and verify bit-exact
    let decoded = decode_codes(&compressed, n, d, &predictions).expect("decode");
    assert_eq!(decoded.len(), raw_codes.len());
    let mut mismatches = 0usize;
    for i in 0..raw_codes.len() {
        if decoded[i] != raw_codes[i] {
            if mismatches < 5 {
                eprintln!(
                    "mismatch at {i}: raw={} decoded={}",
                    raw_codes[i], decoded[i]
                );
            }
            mismatches += 1;
        }
    }
    eprintln!(
        "[pipeline] mismatches: {} / {} bytes",
        mismatches,
        raw_codes.len()
    );
    assert_eq!(mismatches, 0, "Rust encode → decode is NOT bit-exact");
    eprintln!("[pipeline] BIT-EXACT round-trip end-to-end!");
}
