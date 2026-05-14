//! Cross-language interop test.
//!
//! Reads a `.pthc` file produced by `apps/diff-repack/posthac_codec.py`
//! and verifies the Rust crate can parse the header, restore weights,
//! re-run the hash-grid + MLP forward pass for every splat, and
//! range-decode the compressed stream to recover the original 8-bit codes.
//!
//! Run with:
//!     SPLATFORGE_PYTHON_PTHC=/path/to/file.pthc cargo test -p splatforge-posthac --test interop_python -- --nocapture

use byteorder::{LittleEndian, ReadBytesExt};
use splatforge_posthac::*;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

fn read_pthc(path: &PathBuf) -> (PostHacHeader, HyperpriorWeights, Vec<f32>, Vec<u32>) {
    let mut r = BufReader::new(File::open(path).expect("open .pthc"));
    let header = read_header(&mut r).expect("read header");
    let weights = read_weights(&mut r, &header.config).expect("read weights");

    // Position stream: 3 * n f32
    let mut positions = vec![0f32; 3 * header.n as usize];
    for v in positions.iter_mut() {
        *v = r.read_f32::<LittleEndian>().expect("read positions");
    }

    // Compressed stream length + payload
    let comp_len = r.read_u32::<LittleEndian>().expect("read comp_len") as usize;
    let mut compressed = vec![0u32; comp_len];
    for v in compressed.iter_mut() {
        *v = r.read_u32::<LittleEndian>().expect("read compressed");
    }

    (header, weights, positions, compressed)
}

#[test]
fn interop_load_pthc_or_skip() {
    let Ok(p) = env::var("SPLATFORGE_PYTHON_PTHC") else {
        eprintln!("SPLATFORGE_PYTHON_PTHC not set; skipping interop test");
        return;
    };
    let path = PathBuf::from(p);
    let (header, weights, positions, compressed) = read_pthc(&path);
    eprintln!(
        "[interop] loaded N={} D={} sh_deg={} grid={}x{}x2^{} mlp_hidden={}",
        header.n,
        header.d,
        header.sh_degree,
        header.config.grid_levels,
        header.config.features_per_level,
        header.config.log2_hashmap_size,
        header.config.mlp_hidden,
    );
    eprintln!(
        "[interop] weights: grid_tables={} mb, mlp fc1_w={} fc1_b={} fc2_w={} fc2_b={}",
        weights.grid_tables.len() as f32 * 4.0 / 1e6,
        weights.fc1_w.len(),
        weights.fc1_b.len(),
        weights.fc2_w.len(),
        weights.fc2_b.len(),
    );
    eprintln!(
        "[interop] positions: {} f32 = {:.2} MB",
        positions.len(),
        positions.len() as f32 * 4.0 / 1e6,
    );
    eprintln!(
        "[interop] compressed: {} u32 = {:.2} MB",
        compressed.len(),
        compressed.len() as f32 * 4.0 / 1e6,
    );

    // Sanity: predict() for splat 0 should produce in-range means/stds.
    let p0 = [
        (positions[0] - header.pos_mn[0]) / (header.pos_mx[0] - header.pos_mn[0]).max(1e-9),
        (positions[1] - header.pos_mn[1]) / (header.pos_mx[1] - header.pos_mn[1]).max(1e-9),
        (positions[2] - header.pos_mn[2]) / (header.pos_mx[2] - header.pos_mn[2]).max(1e-9),
    ];
    let pred0 = predict(p0, &header.config, &weights);
    eprintln!(
        "[interop] splat 0 prediction: mean[:4]={:?} std[:4]={:?}",
        &pred0.mean[..pred0.mean.len().min(4)],
        &pred0.std[..pred0.std.len().min(4)],
    );
    assert!(pred0.mean.iter().all(|&m| (0.0..=255.0).contains(&m)));
    assert!(pred0.std.iter().all(|&s| (0.5..=128.0).contains(&s)));

    // Build the full prediction array — required for decode.
    let n = header.n as usize;
    let d = header.d as usize;
    let mut predictions = Vec::with_capacity(n);
    for i in 0..n {
        let p = [
            (positions[3 * i] - header.pos_mn[0])
                / (header.pos_mx[0] - header.pos_mn[0]).max(1e-9),
            (positions[3 * i + 1] - header.pos_mn[1])
                / (header.pos_mx[1] - header.pos_mn[1]).max(1e-9),
            (positions[3 * i + 2] - header.pos_mn[2])
                / (header.pos_mx[2] - header.pos_mn[2]).max(1e-9),
        ];
        predictions.push(predict(p, &header.config, &weights));
    }

    eprintln!("[interop] running Rust range-decoder on {} splats × {} attrs...", n, d);
    let decoded = decode_codes(&compressed, n, d, &predictions).expect("decode");
    assert_eq!(decoded.len(), n * d);
    eprintln!("[interop] decoded {} bytes successfully", decoded.len());
}
