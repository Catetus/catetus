//! Cross-language two-batch encode/decode test.
use constriction::stream::model::{DefaultLeakyQuantizer, LeakyQuantizer};
use constriction::stream::queue::DefaultRangeDecoder;
use constriction::stream::Decode;
use probability::distribution::Gaussian;
use std::env;
use std::fs::File;
use std::io::Read;

#[test]
fn rust_decodes_python_two_batches() {
    let Ok(path) = env::var("SPLATFORGE_PY_BATCHES") else {
        return;
    };
    let mut buf = Vec::new();
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let py_bytes: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let quantizer: DefaultLeakyQuantizer<f64, i32> = LeakyQuantizer::new(-1..=256);
    let mut dec = DefaultRangeDecoder::from_compressed(py_bytes).unwrap();

    let mut decoded = Vec::new();
    // Batch A: 8 symbols with mu=128, std=50
    for _ in 0..8 {
        let model = quantizer.quantize(Gaussian::new(128.0, 50.0));
        decoded.push(dec.decode_symbol(model).unwrap());
    }
    // Batch B: 8 symbols with mu=100, std=30
    for _ in 0..8 {
        let model = quantizer.quantize(Gaussian::new(100.0, 30.0));
        decoded.push(dec.decode_symbol(model).unwrap());
    }
    eprintln!("decoded: {:?}", decoded);
    let expected = [100, 50, 200, 30, 128, 64, 192, 96, 150, 80, 220, 40, 100, 60, 180, 110];
    assert_eq!(decoded, expected);
}
