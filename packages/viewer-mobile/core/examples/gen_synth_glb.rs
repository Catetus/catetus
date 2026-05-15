//! Generate a small synthetic `KHR_gaussian_splatting` `.glb` fixture for the
//! iOS / Android demo apps. The output mimics a Gaussian-splatted point cloud
//! at modest splat counts (~10k) so the demo app shows something interesting
//! out of the box without depending on the ~22 MB `bonsai-7k.glb` asset.
//!
//! Run:
//!     cargo run --example gen_synth_glb --release -- <output-path>
//!
//! Suggested default:
//!     packages/viewer-mobile/examples/iOS-Demo/Sources/iOSDemo/Assets/synth.glb

use std::env;
use std::path::PathBuf;

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_gltf::{write_glb, WriteOpts};

fn main() {
    let args: Vec<String> = env::args().collect();
    let out: PathBuf = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("synth.glb"));
    let count: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let mut scene = SplatScene::new();
    scene.splats.reserve(count);

    // Build a colorful gaussian-distributed cloud roughly in a unit sphere so
    // the demo's auto-frame routine pulls the camera back to ~1.6 units.
    // Hand-rolled LCG so the example is fully deterministic and dependency-free.
    let mut state: u64 = 0xC0FFEE_u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (state >> 33) as f32 / (1u32 << 31) as f32;
        v - 1.0 // ~[-1, 1)
    };

    for i in 0..count {
        // Two interleaved blobs offset along X for visual interest.
        let blob = (i % 2) as f32 * 1.5 - 0.75;
        let p = [
            next() * 0.6 + blob,
            next() * 0.6,
            next() * 0.6,
        ];
        let t = (i as f32 / count as f32).clamp(0.0, 1.0);
        let color = if i % 2 == 0 {
            [0.95, 0.30 + 0.50 * t, 0.15]
        } else {
            [0.15, 0.40 + 0.40 * t, 0.95]
        };
        scene.splats.push(Splat {
            position: p,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.02, 0.02, 0.02],
            opacity: 0.85,
            color: Color::Rgb(color),
        });
    }

    write_glb(&scene, &out, &WriteOpts::default()).expect("write_glb");
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("wrote {count} splats → {} ({size} bytes)", out.display());
}
