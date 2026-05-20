//! Micro-bench for `read_ply` against a real PLY on disk.
//!
//! Usage:
//!   ply-read-bench <path-to-ply> [iterations]
//!
//! Reports wall time and splats/second per iteration. The first iteration is
//! cold-cache; subsequent iterations are warm-cache. Useful for before/after
//! comparisons of the binary-body decode path.

use std::env;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next().map(PathBuf::from) else {
        eprintln!("usage: ply-read-bench <path-to-ply> [iterations]");
        process::exit(2);
    };
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let bytes_on_disk = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!(
        "ply-read-bench: file={} size={:.2} MB iters={}",
        path.display(),
        bytes_on_disk as f64 / (1024.0 * 1024.0),
        iters,
    );

    for i in 0..iters {
        let t0 = Instant::now();
        let scene = match catetus_ply::read_ply(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("read_ply failed: {e}");
                process::exit(1);
            }
        };
        let dt = t0.elapsed();
        let secs = dt.as_secs_f64();
        let n = scene.len() as f64;
        let mb = bytes_on_disk as f64 / (1024.0 * 1024.0);
        let label = if i == 0 { "cold" } else { "warm" };
        println!(
            "iter[{i}] {label}: {:.3}s | {:.0} splats | {:.1} M splats/s | {:.1} MB/s",
            secs,
            n,
            n / 1.0e6 / secs,
            mb / secs,
        );
        // Drop scene explicitly so the allocator returns memory between iters.
        drop(scene);
    }
}
