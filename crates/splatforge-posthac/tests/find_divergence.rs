//! Scan all splats and find the first one where Rust predict() diverges
//! from Python's prediction.

use byteorder::{LittleEndian, ReadBytesExt};
use splatforge_posthac::*;
use std::env;
use std::fs::File;
use std::io::BufReader;

#[test]
fn find_first_prediction_divergence() {
    let Ok(pthc) = env::var("SPLATFORGE_PYTHON_PTHC") else {
        return;
    };
    let Ok(preds) = env::var("SPLATFORGE_PY_ALLPREDS") else {
        return;
    };

    let mut r = BufReader::new(File::open(pthc).unwrap());
    let header = read_header(&mut r).unwrap();
    let weights = read_weights(&mut r, &header.config).unwrap();
    let n = header.n as usize;
    let d = header.d as usize;

    let mut positions = vec![0f32; 3 * n];
    for v in positions.iter_mut() {
        *v = r.read_f32::<LittleEndian>().unwrap();
    }

    let mut pr = BufReader::new(File::open(preds).unwrap());
    let n_dump = pr.read_u32::<LittleEndian>().unwrap() as usize;
    let d_dump = pr.read_u32::<LittleEndian>().unwrap() as usize;
    assert_eq!(n_dump, n);
    assert_eq!(d_dump, d);
    let mut py_mu = vec![0f32; n * d];
    let mut py_sigma = vec![0f32; n * d];
    for v in py_mu.iter_mut() {
        *v = pr.read_f32::<LittleEndian>().unwrap();
    }
    for v in py_sigma.iter_mut() {
        *v = pr.read_f32::<LittleEndian>().unwrap();
    }

    let mut first_diverge: Option<(usize, usize, f64, f32, f64, f32)> = None;
    for i in 0..n {
        let p = [
            (positions[3 * i] - header.pos_mn[0]) / (header.pos_mx[0] - header.pos_mn[0]).max(1e-9),
            (positions[3 * i + 1] - header.pos_mn[1])
                / (header.pos_mx[1] - header.pos_mn[1]).max(1e-9),
            (positions[3 * i + 2] - header.pos_mn[2])
                / (header.pos_mx[2] - header.pos_mn[2]).max(1e-9),
        ];
        let pred = predict(p, &header.config, &weights);
        for c in 0..d {
            let mr = pred.mean[c];
            let sr = pred.std[c];
            let mp = py_mu[i * d + c];
            let sp = py_sigma[i * d + c];
            if (mr as f32 - mp).abs() > 1e-4 || (sr as f32 - sp).abs() > 1e-4 {
                first_diverge = Some((i, c, mr, mp, sr, sp));
                break;
            }
        }
        if first_diverge.is_some() {
            break;
        }
    }
    match first_diverge {
        Some((i, c, mr, mp, sr, sp)) => {
            eprintln!("FIRST DIVERGENCE at splat {i} col {c}:");
            eprintln!("  Rust   mu={mr} std={sr}");
            eprintln!("  Python mu={mp} std={sp}");
            panic!("predictions diverge — that's why decode fails");
        }
        None => {
            eprintln!("NO DIVERGENCE found across all {n} splats × {d} attrs");
        }
    }
}
