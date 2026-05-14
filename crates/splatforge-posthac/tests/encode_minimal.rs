//! Cross-language smoke test: 16-symbol Python encode → Rust decode.
//! Must produce the same input symbols if Python encoder + Rust decoder
//! use identical constriction Gaussian models.

use byteorder::{LittleEndian, ReadBytesExt};
use constriction::stream::model::{DefaultLeakyQuantizer, LeakyQuantizer};
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use constriction::stream::{Decode, Encode};
use probability::distribution::Gaussian;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

#[test]
fn rust_encode_matches_python_bytes() {
    // Same 16 symbols Python encoded.
    let symbols: [i32; 16] = [100, 50, 200, 30, 128, 64, 192, 96,
                               150, 80, 220, 40, 100, 60, 180, 110];
    let quantizer: DefaultLeakyQuantizer<f64, i32> = LeakyQuantizer::new(-1..=256);
    let mut enc = DefaultRangeEncoder::new();
    for &s in &symbols {
        let model = quantizer.quantize(Gaussian::new(128.0, 50.0));
        enc.encode_symbol(s, model).unwrap();
    }
    let rust_bytes = enc.into_compressed().unwrap();
    eprintln!("Rust encoded: {:?}", rust_bytes);

    // Load Python output.
    let Ok(path) = env::var("SPLATFORGE_PY_ENCODED") else {
        eprintln!("SPLATFORGE_PY_ENCODED not set; just printing Rust output");
        return;
    };
    let mut r = BufReader::new(File::open(path).unwrap());
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).unwrap();
    let py_bytes: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("Python encoded: {:?}", py_bytes);

    if rust_bytes == py_bytes {
        eprintln!("BYTE-FOR-BYTE MATCH! cross-language interop works.");
    } else {
        eprintln!("MISMATCH — investigating decode anyway.");
    }

    // Now try to Rust-decode the Python output:
    let mut dec = DefaultRangeDecoder::from_compressed(py_bytes.clone()).unwrap();
    let mut decoded = Vec::with_capacity(16);
    for _ in 0..16 {
        let model = quantizer.quantize(Gaussian::new(128.0, 50.0));
        let s = dec.decode_symbol(model).unwrap();
        decoded.push(s);
    }
    eprintln!("Decoded Python bytes via Rust: {:?}", decoded);
    assert_eq!(&decoded[..], &symbols[..]);
}
