//! D-WIRE driver: emit a REAL on-disk shared-palette tileset for bonsai.
//!
//! All local. Uses the same scene + default configs as `tile_codec_bench.rs`
//! (TilesetConfig::default, SharedPaletteConfig::default = K=4096/5/0xC0DE) so
//! the on-disk total is directly comparable to the bench's 95,439,411 B.
//!
//!   cargo run --release --example dwire_emit -p catetus-tileset -- <out_dir>

use std::path::{Path, PathBuf};

use catetus_tileset::{plan_tileset, write_tileset_shared, SharedPaletteConfig, TilesetConfig};

const PRIMARY_PLY: &str = "/Users/montabano1/Downloads/bonsai_comparison/bonsai_original.ply";
const FALLBACK_PLY: &str =
    "/Users/montabano1/Desktop/SplatForgeProject/SplatForge/benches/scenes/bonsai_mipnerf360_iter7k.ply";

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: dwire_emit <out_dir>");

    let ply = if Path::new(PRIMARY_PLY).exists() { PRIMARY_PLY } else { FALLBACK_PLY };
    eprintln!("PLY: {ply}");
    let scene = catetus_ply::read_ply(&PathBuf::from(ply)).expect("read_ply");
    eprintln!("loaded {} splats", scene.splats.len());

    let plan = plan_tileset(&scene, &TilesetConfig::default()).expect("plan");
    eprintln!("planned {} tiles", plan.payloads.len());

    let w = write_tileset_shared(&plan, &scene, &out, &SharedPaletteConfig::default())
        .expect("write_tileset_shared");

    // Machine-readable sentinels (small, one per line).
    println!("DWIRE_OUT={}", out.display());
    println!("DWIRE_N_TILES={}", w.n_tiles);
    println!("DWIRE_PALETTE_K={:?}", w.palette_size);
    println!("DWIRE_CODEBOOK_B={}", w.codebook_bytes);
    println!("DWIRE_TILE_GLB_B={}", w.tile_glb_bytes);
    println!("DWIRE_INDEX_B={}", w.index_bytes);
    println!("DWIRE_TOTAL_B={}", w.total_bytes);
    println!("DWIRE_DONE=1");
}
