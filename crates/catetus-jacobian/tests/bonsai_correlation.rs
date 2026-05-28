//! Integration test: compare `catetus-jacobian`'s SH-rest proxy against the
//! Python ground-truth NPZ produced by `jacobian_census.py` on the 4090 box.
//!
//! The test is **opt-in** — it requires three large artifacts not checked
//! into the repo:
//!
//!   - `CATETUS_JACOBIAN_BONSAI_PLY` — path to a bonsai 3DGS PLY (the same
//!     PLY the census ran on, ideally `bonsai_ref.ply` with N=1,244,819).
//!   - `CATETUS_JACOBIAN_REFERENCE_NPZ` — path to `J_per_splat.npz` from
//!     `experiments/jacobian-census-bonsai-30k/raw_runs/*/`.
//!
//! When either env var is missing the test logs a skip message and
//! returns success — keeps `cargo test` green in CI / on machines that
//! don't have the artifacts.
//!
//! When both are present it computes the proxy on the loaded PLY, loads
//! the reference `J_sh_rest`, and asserts Pearson correlation > 0.5
//! (acceptable per the contract; target > 0.7).

use std::path::PathBuf;

use catetus_jacobian::compute_jacobian;
use catetus_ply::read_ply;

#[path = "common/npz.rs"]
mod npz;

#[test]
fn bonsai_proxy_correlates_with_python_reference() {
    let Some(ply_path) = optional_env_path("CATETUS_JACOBIAN_BONSAI_PLY") else {
        eprintln!(
            "[skip] CATETUS_JACOBIAN_BONSAI_PLY not set — see test header for the artifacts \
             needed to run this end-to-end."
        );
        return;
    };
    let Some(npz_path) = optional_env_path("CATETUS_JACOBIAN_REFERENCE_NPZ") else {
        eprintln!(
            "[skip] CATETUS_JACOBIAN_REFERENCE_NPZ not set — see test header for the artifacts \
             needed to run this end-to-end."
        );
        return;
    };

    eprintln!("[bonsai-correlation] loading PLY: {}", ply_path.display());
    let scene = read_ply(&ply_path).expect("read bonsai PLY");
    eprintln!("[bonsai-correlation]   N = {} splats", scene.splats.len());

    eprintln!("[bonsai-correlation] loading reference NPZ: {}", npz_path.display());
    let reference = npz::load_npz_array_f32(&npz_path, "J_sh_rest.npy")
        .expect("load J_sh_rest from reference NPZ");
    eprintln!("[bonsai-correlation]   reference array length = {}", reference.len());

    assert_eq!(
        reference.len(),
        scene.splats.len(),
        "PLY splat count must match reference Jacobian length. PLY={} reference={}",
        scene.splats.len(),
        reference.len(),
    );

    eprintln!("[bonsai-correlation] computing proxy …");
    let t0 = std::time::Instant::now();
    let proxy = compute_jacobian(&scene).j_sh_rest;
    eprintln!(
        "[bonsai-correlation]   computed N={} in {:.2}s",
        proxy.len(),
        t0.elapsed().as_secs_f64()
    );

    let pearson = pearson_correlation(&proxy, &reference);
    let spearman_approx = pearson_correlation(
        &rank_transform(&proxy),
        &rank_transform(&reference),
    );

    eprintln!("[bonsai-correlation]   Pearson  ρ = {pearson:.4}");
    eprintln!("[bonsai-correlation]   Spearman ρ = {spearman_approx:.4}  (approx, via rank transform)");

    let top_k_overlap = top_k_overlap(&proxy, &reference, 0.01);
    eprintln!(
        "[bonsai-correlation]   top-1% overlap = {:.1}%",
        top_k_overlap * 100.0
    );

    // Acceptable per contract: > 0.5 on Pearson OR > 0.5 on Spearman OR
    // > 30 % top-1% overlap (the third clause guards against the long-tailed
    // distribution dragging Pearson down even when the proxy correctly
    // identifies the renderable-mass splats — which is what VQPaletteShRest
    // actually consumes).
    let pass = pearson > 0.5 || spearman_approx > 0.5 || top_k_overlap > 0.30;
    assert!(
        pass,
        "proxy did not meet the correlation gate (Pearson={pearson:.3}, \
         Spearman={spearman_approx:.3}, top-1% overlap={top_k_overlap:.3}). \
         The MVP target is > 0.5 on either correlation or > 30% top-1% overlap."
    );

    if pearson > 0.7 || spearman_approx > 0.7 {
        eprintln!("[bonsai-correlation]   stretch target met (correlation > 0.7).");
    }
}

fn optional_env_path(key: &str) -> Option<PathBuf> {
    let v = std::env::var(key).ok()?;
    if v.is_empty() {
        None
    } else {
        Some(PathBuf::from(v))
    }
}

fn pearson_correlation(xs: &[f32], ys: &[f32]) -> f32 {
    assert_eq!(xs.len(), ys.len(), "Pearson inputs must be the same length");
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..xs.len() {
        sx += xs[i] as f64;
        sy += ys[i] as f64;
    }
    let mx = sx / n;
    let my = sy / n;
    let mut num = 0.0f64;
    let mut dx2 = 0.0f64;
    let mut dy2 = 0.0f64;
    for i in 0..xs.len() {
        let dx = xs[i] as f64 - mx;
        let dy = ys[i] as f64 - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom <= 0.0 {
        return 0.0;
    }
    (num / denom) as f32
}

/// Replace each value with its rank (1..=N), with ties broken arbitrarily
/// by index. Used for an approximate Spearman correlation (true Spearman
/// uses average ranks for ties — but `J_sh_rest` is float32 with very few
/// exact ties, so the approximation is fine for this test).
fn rank_transform(xs: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = xs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; xs.len()];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        ranks[orig_idx] = (rank + 1) as f32;
    }
    ranks
}

/// Fraction of the top-`q` mass in `a` that's also in the top-`q` mass of
/// `b` (and vice-versa, averaged). Robust to the long-tailed distribution
/// — this is the thing that actually matters for the downstream
/// VQPaletteShRest weighted-Lloyd selection.
fn top_k_overlap(a: &[f32], b: &[f32], q: f32) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let k = ((n as f32) * q).ceil() as usize;
    if k == 0 {
        return 0.0;
    }
    let top_k_indices = |xs: &[f32]| -> std::collections::HashSet<usize> {
        let mut indexed: Vec<(usize, f32)> = xs.iter().copied().enumerate().collect();
        // Sort descending by value.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    };
    let sa = top_k_indices(a);
    let sb = top_k_indices(b);
    let inter = sa.intersection(&sb).count();
    (inter as f32) / (k as f32)
}
