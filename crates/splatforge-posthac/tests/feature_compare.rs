//! Compare Rust hash-grid forward pass to Python's for splat 0.
use byteorder::{LittleEndian, ReadBytesExt};
use splatforge_posthac::*;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

#[test]
fn compare_splat0_features() {
    let Ok(pthc) = env::var("SPLATFORGE_PYTHON_PTHC") else {
        eprintln!("SPLATFORGE_PYTHON_PTHC not set; skipping");
        return;
    };
    let Ok(feats_bin) = env::var("SPLATFORGE_FEATS_BIN") else {
        eprintln!("SPLATFORGE_FEATS_BIN not set; skipping");
        return;
    };

    let mut r = BufReader::new(File::open(PathBuf::from(&pthc)).expect("open .pthc"));
    let header = read_header(&mut r).expect("read header");
    let weights = read_weights(&mut r, &header.config).expect("read weights");
    // Read pos 0 only
    let mut pos = [0f32; 3];
    pos[0] = r.read_f32::<LittleEndian>().unwrap();
    pos[1] = r.read_f32::<LittleEndian>().unwrap();
    pos[2] = r.read_f32::<LittleEndian>().unwrap();

    let pos_norm = [
        (pos[0] - header.pos_mn[0]) / (header.pos_mx[0] - header.pos_mn[0]).max(1e-9),
        (pos[1] - header.pos_mn[1]) / (header.pos_mx[1] - header.pos_mn[1]).max(1e-9),
        (pos[2] - header.pos_mn[2]) / (header.pos_mx[2] - header.pos_mn[2]).max(1e-9),
    ];
    eprintln!("pos[0] = {pos:?}");
    eprintln!("pos_norm[0] = {pos_norm:?}");

    // Compute features identically to predict()'s loop.
    let cfg = &header.config;
    let hashmap_size = 1usize << cfg.log2_hashmap_size;
    let f_per_lvl = cfg.features_per_level as usize;
    let n_feats = (cfg.grid_levels as usize) * f_per_lvl;
    let mut feats = vec![0f32; n_feats];
    for lvl in 0..(cfg.grid_levels as usize) {
        let scale_f = (cfg.base_resolution as f32) * 1.5f32.powi(lvl as i32);
        let scale = scale_f.trunc();
        let mut g = [0f32; 3];
        for d in 0..3 {
            g[d] = pos_norm[d] * scale;
        }
        let x0 = [
            g[0].floor() as i64,
            g[1].floor() as i64,
            g[2].floor() as i64,
        ];
        let xf = [
            g[0] - x0[0] as f32,
            g[1] - x0[1] as f32,
            g[2] - x0[2] as f32,
        ];
        let level_offset = lvl * hashmap_size * f_per_lvl;
        for dx in 0..2_i64 {
            for dy in 0..2_i64 {
                for dz in 0..2_i64 {
                    let corner = [x0[0] + dx, x0[1] + dy, x0[2] + dz];
                    let wx = if dx == 1 { xf[0] } else { 1.0 - xf[0] };
                    let wy = if dy == 1 { xf[1] } else { 1.0 - xf[1] };
                    let wz = if dz == 1 { xf[2] } else { 1.0 - xf[2] };
                    let w = wx * wy * wz;
                    let bucket = hash3(corner, hashmap_size);
                    for k in 0..f_per_lvl {
                        let off = level_offset + bucket * f_per_lvl + k;
                        feats[lvl * f_per_lvl + k] += w * weights.grid_tables[off];
                    }
                }
            }
        }
    }
    eprintln!("Rust features = {:?}", feats);

    // Read Python features
    let mut r2 = BufReader::new(File::open(PathBuf::from(&feats_bin)).expect("open feats"));
    let _gl = r2.read_u32::<LittleEndian>().unwrap();
    let _fpl = r2.read_u32::<LittleEndian>().unwrap();
    let mut py_feats = vec![0f32; n_feats];
    for v in py_feats.iter_mut() {
        *v = r2.read_f32::<LittleEndian>().unwrap();
    }
    eprintln!("Python features = {:?}", py_feats);

    let mut max_diff: f32 = 0.0;
    for i in 0..n_feats {
        let diff = (feats[i] - py_feats[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            eprintln!("feat {i}: rust={} py={} diff={diff}", feats[i], py_feats[i]);
        }
    }
    eprintln!("features max-diff = {max_diff}");

    // Now run the full predict() and dump first 4 (μ, σ).
    let p = predict(pos_norm, &header.config, &weights);
    eprintln!("Rust predict mu[:4] = {:?}", &p.mean[..4]);
    eprintln!("Rust predict std[:4] = {:?}", &p.std[..4]);

    // Also predict for splat 4920 (the one that fails decode).
    let idx = 4920;
    // Re-read positions and find splat idx — would need second pass. Use env var
    // to inject if provided.
    if let Ok(ps) = env::var("SPLATFORGE_POS_4920") {
        let nums: Vec<f32> = ps.split(',').map(|s| s.trim().parse().unwrap()).collect();
        let pos_n = [
            (nums[0] - header.pos_mn[0]) / (header.pos_mx[0] - header.pos_mn[0]).max(1e-9),
            (nums[1] - header.pos_mn[1]) / (header.pos_mx[1] - header.pos_mn[1]).max(1e-9),
            (nums[2] - header.pos_mn[2]) / (header.pos_mx[2] - header.pos_mn[2]).max(1e-9),
        ];
        let p = predict(pos_n, &header.config, &weights);
        eprintln!("Rust splat {idx} mu[:4] = {:?}", &p.mean[..4]);
        eprintln!("Rust splat {idx} std[:4] = {:?}", &p.std[..4]);
    }
}
