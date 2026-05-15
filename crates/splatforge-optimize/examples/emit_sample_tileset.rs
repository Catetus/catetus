//! Emit the committed `crates/splatforge-optimize/tests/fixtures/geospatial-sample/`
//! reference tileset deterministically. Run with:
//!
//! ```bash
//! cargo run -p splatforge-optimize --example emit_sample_tileset
//! ```
//!
//! Produces ~450 splats split into 4 LODs. The output is byte-deterministic, so
//! re-running the example is a no-op in `git diff` unless this generator
//! changes.

use std::path::PathBuf;

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_optimize::{preset, write_tileset, TilesetOpts};

fn synthetic_scene(n: usize) -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..n {
        let f = i as f32;
        // Two-radius torus-like distribution. Lots of structure -> nicer Morton
        // sort and bounding box, but still deterministic.
        let theta = f * 0.197;
        let phi = f * 0.071;
        let r1 = 3.0;
        let r2 = 0.6 + (f * 0.013).sin() * 0.2;
        let x = (r1 + r2 * phi.cos()) * theta.cos();
        let y = (r1 + r2 * phi.cos()) * theta.sin();
        let z = r2 * phi.sin();
        scene.splats.push(Splat {
            position: [x, y, z],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.04, 0.04, 0.04],
            opacity: 0.85,
            color: Color::Rgb([
                0.5 + 0.4 * (f * 0.031).sin(),
                0.5 + 0.4 * (f * 0.041).cos(),
                0.5 + 0.4 * (f * 0.053).sin(),
            ]),
        });
    }
    scene
}

fn main() -> anyhow::Result<()> {
    let n = 450;
    let mut scene = synthetic_scene(n);
    let pipe = preset("geospatial")?;
    let report = pipe.run(&mut scene)?;

    // Walk up from the manifest to find the crate root, then point at the
    // committed fixture path.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let out: PathBuf = [manifest, "tests", "fixtures", "geospatial-sample"]
        .iter()
        .collect();
    std::fs::create_dir_all(&out)?;

    let tr = write_tileset(&scene, &out, &TilesetOpts::default())?;
    let summary = serde_json::json!({
        "synthetic_splat_count_input": n,
        "splats_after_pipeline": report.splats_after,
        "tiles": tr.tiles.iter().map(|t| serde_json::json!({
            "lod": t.lod_index,
            "fraction": t.fraction,
            "splats": t.splat_count,
            "geometric_error": t.geometric_error,
            "glb": t.glb,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        out.join("README.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("wrote sample tileset to {}", out.display());
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
