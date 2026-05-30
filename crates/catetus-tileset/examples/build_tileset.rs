//! Build a tileset from a synthetic scene and print a summary.
//!
//! Run: `cargo run -p catetus-tileset --example build_tileset -- /tmp/out`
//!
//! This example does not depend on the PLY reader; it generates a deterministic
//! synthetic cube scene so the whole emit path can be exercised with zero
//! inputs. Once the CLI wiring lands, `catetus optimize --target tileset
//! scene.ply -o out/` will replace the synthetic source with a real decoded
//! scene (via `catetus-ply::read_ply`).

use std::path::PathBuf;

use catetus_core::ir::{Color, Splat, SplatScene};
use catetus_tileset::{plan_tileset, write_tileset, OctreeConfig, SfTileCodec, TilesetConfig};

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tileset-out"));

    // Synthetic scene: a 40^3 lattice (64,000 splats) so the octree subdivides
    // several levels.
    let n = 40usize;
    let mut scene = SplatScene::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                scene.splats.push(Splat {
                    position: [x as f32, y as f32, z as f32],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.5, 0.5, 0.5],
                    opacity: 0.7,
                    color: Color::Rgb([0.4, 0.5, 0.6]),
                });
            }
        }
    }
    eprintln!("synthetic scene: {} splats", scene.len());

    let cfg = TilesetConfig {
        octree: OctreeConfig { max_depth: 6, max_splats_per_leaf: 4_000 },
        lods_per_node: 3,
        coarsest_target: 500,
        proxy_cap: 20_000,
    };

    let plan = plan_tileset(&scene, &cfg).expect("plan");
    eprintln!(
        "octree: lod_levels={} tiles={}",
        plan.lod_meta.lod_levels,
        plan.payloads.len()
    );

    let bytes = write_tileset(&plan, &SfTileCodec, &out).expect("write");
    eprintln!("wrote tileset to {} ({} tile bytes)", out.display(), bytes);
    eprintln!("  - {}/lod-meta.json   (SuperSplat-compatible)", out.display());
    eprintln!("  - {}/tileset.json    (3D Tiles 1.1)", out.display());
    eprintln!("  - {}/tiles/*.sftile", out.display());
}
