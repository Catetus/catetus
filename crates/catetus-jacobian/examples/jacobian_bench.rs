//! `cargo run -p catetus-jacobian --example jacobian-bench -- <PLY>`
//!
//! Loads a PLY, computes the SH-rest Jacobian proxy, prints summary
//! statistics that mirror the Python census `summary.json` schema so a
//! human can eyeball whether the proxy is in the right ballpark.

use std::env;
use std::path::PathBuf;

use catetus_jacobian::compute_jacobian;
use catetus_ply::read_ply;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: jacobian-bench <PLY>");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    eprintln!("[bench] loading {}", path.display());
    let scene = read_ply(&path)?;
    eprintln!("[bench]   N = {}", scene.splats.len());

    let t0 = std::time::Instant::now();
    let result = compute_jacobian(&scene);
    let elapsed = t0.elapsed();
    eprintln!(
        "[bench] computed J_sh_rest in {:.3}s ({:.1} M splats/s)",
        elapsed.as_secs_f64(),
        scene.splats.len() as f64 / elapsed.as_secs_f64() / 1e6,
    );

    let j = result.j_sh_rest;
    print_summary(&j);
    Ok(())
}

fn print_summary(j: &[f32]) {
    let n = j.len();
    if n == 0 {
        println!("(empty)");
        return;
    }
    let mut sorted = j.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = j.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let median = sorted[n / 2];
    let max = *sorted.last().unwrap();
    let min = sorted[0];
    let sum = j.iter().map(|&v| v as f64).sum::<f64>();

    println!("{{");
    println!("  \"method\": \"GeometricProxyV1\",");
    println!("  \"N\": {n},");
    println!("  \"mean\":   {mean:.6e},");
    println!("  \"median\": {median:.6e},");
    println!("  \"max\":    {max:.6e},");
    println!("  \"min\":    {min:.6e},");
    println!("  \"sum\":    {sum:.6e},");
    println!("  \"topk_share\": {{");
    for q in [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50] {
        let k = ((n as f32) * q).ceil() as usize;
        let k = k.max(1);
        let head_sum: f64 = sorted[n - k..].iter().map(|&v| v as f64).sum();
        let share = head_sum / sum.max(1e-30);
        println!("    \"top_{:.1}%\": {:.6},", q * 100.0, share);
    }
    println!("  }}");
    println!("}}");
}
