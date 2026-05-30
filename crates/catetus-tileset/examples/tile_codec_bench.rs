//! TILE-CODEC-1 streaming-compression gate benchmark (throwaway).
//!
//! Question: does our GLB tile codec still beat the competitor SOG format when a
//! scene is chopped into many small streaming tiles, and what is the per-tile
//! "overhead tax" vs encoding the scene as one monolithic file?
//!
//! All local. Run:
//!   cargo run --release --example tile_codec_bench -p catetus-tileset 2>&1 | tail -60
//!
//! Verified APIs (file:line at authorship time):
//!   - catetus_ply::read_ply(path: impl AsRef<Path>) -> Result<SplatScene, PlyError>
//!         (crates/catetus-ply/src/lib.rs:14)
//!   - catetus_tileset::plan_tileset(&SplatScene, &TilesetConfig) -> Result<TilesetPlan,_>
//!         (crates/catetus-tileset/src/plan.rs:126)
//!   - catetus_tileset::TilesetConfig { lods_per_node, coarsest_target, proxy_cap, octree }
//!         (plan.rs:16-44 ; Default = lods_per_node 3, coarsest_target 2_000, proxy_cap 20_000)
//!   - GlbTileCodec::new(TilePreset).encode(&SplatScene) -> TileBytes{bytes, sidecar,..}
//!         (crates/catetus-tileset/src/glb_codec.rs:83,115 ; codec.rs:81-90)
//!   - catetus_gltf::write_glb(scene: &SplatScene, path: &Path, opts: &WriteOpts) -> Result<(),GltfError>
//!         (crates/catetus-gltf/src/lib.rs ; same call used by glb_codec.rs:108)
//!
//! SOG: the SOG encoder crate (`catetus-sog`) was MOVED to the private
//! `catetus/catetus-private` repo on 2026-05-19 (open-core split). It does NOT
//! exist in this public tree — `crates/catetus-sog/` is absent and the CLI's
//! `Target::Sog` arm just `anyhow::bail!`s pointing at api.catetus.com
//! (crates/catetus-cli/src/main.rs:462-471). So we CANNOT encode SOG locally.
//! The HARD RULES forbid network calls, so the per-tile SOG numbers are not
//! locally obtainable. For the MONOLITHIC head-to-head we use the real,
//! pre-existing SuperSplat-produced SOG of the SAME bonsai scene on disk
//! (bonsai_supersplat.sog, a genuine competitor artifact). Per-tile SOG is
//! reported as BLOCKED (no local encoder) — never fabricated.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use catetus_core::ir::SplatScene;
use catetus_gltf::{write_glb, ShRestQuantTable, WriteOpts};
use catetus_tileset::{
    plan_tileset, GlbTileCodec, SharedCodebook, SharedPaletteTileCodec, TilePayloadCodec,
    TilePreset, TilesetConfig, SHARED_PALETTE_FILENAME,
};

const PRIMARY_PLY: &str = "/Users/montabano1/Downloads/bonsai_comparison/bonsai_original.ply";
const FALLBACK_PLY: &str =
    "/Users/montabano1/Desktop/SplatForgeProject/SplatForge/benches/scenes/bonsai_mipnerf360_iter7k.ply";
/// Real SuperSplat-produced SOG of the same bonsai scene (genuine competitor
/// artifact). Used for the MONOLITHIC head-to-head only; per-tile SOG cannot be
/// produced locally (encoder is private/hosted-only).
const MONO_SOG_DISK: &str = "/Users/montabano1/Downloads/bonsai_comparison/bonsai_supersplat.sog";
const LOG: &str = "/tmp/tilecodec_result.txt";

fn log_line(f: &mut std::fs::File, s: &str) {
    println!("{s}");
    let _ = writeln!(f, "{s}");
    let _ = f.flush();
}

/// The same writer-side knobs as TilePreset::Balanced (private to the crate),
/// so the monolithic GLB is apples-to-apples with the tiles.
fn balanced_opts() -> WriteOpts {
    let mut o = WriteOpts::default();
    o.quantize = true;
    o.log_quant_attrs = true;
    o
}

/// Monolithic GLB byte count via the same path-based writer the tile codec uses.
fn mono_glb_bytes(scene: &SplatScene, opts: &WriteOpts) -> Result<u64, String> {
    let tmp = std::env::temp_dir().join(format!("tilecodec-mono-{}.glb", std::process::id()));
    write_glb(scene, &tmp, opts).map_err(|e| format!("write_glb: {e}"))?;
    let n = std::fs::metadata(&tmp).map_err(|e| format!("stat: {e}"))?.len();
    let _ = std::fs::remove_file(&tmp);
    Ok(n)
}

/// Per-tile `sh_rest_quant` ranges: per-coefficient absolute-max over THIS
/// tile's SH-rest (45 scalars). 8-bit BYTE accessors → 45 B/splat for SH-rest
/// (down from 180 B FP32), no shared codebook.
fn tile_sh_rest_ranges(scene: &SplatScene) -> Vec<f32> {
    let mut ranges = vec![1e-9f32; 45];
    for s in &scene.splats {
        if let catetus_core::ir::Color::Sh { coeffs, .. } = &s.color {
            if coeffs.len() >= 48 {
                for d in 0..45 {
                    let v = coeffs[3 + d].abs();
                    if v > ranges[d] {
                        ranges[d] = v;
                    }
                }
            }
        }
    }
    ranges
}

/// Write a tile GLB with per-tile 8-bit `sh_rest_quant` (the cheaper, no-shared-
/// codebook rung) and return total bytes.
fn sh_rest_quant_tile_bytes(scene: &SplatScene) -> usize {
    let mut o = WriteOpts::default();
    o.quantize = true;
    o.log_quant_attrs = true;
    o.sh_rest_quant = Some(ShRestQuantTable {
        bits: 8,
        ranges: tile_sh_rest_ranges(scene),
    });
    let tmp =
        std::env::temp_dir().join(format!("tilecodec-shq-{}.glb", std::process::id()));
    if write_glb(scene, &tmp, &o).is_err() {
        return 0;
    }
    let n = std::fs::metadata(&tmp).map(|m| m.len() as usize).unwrap_or(0);
    let _ = std::fs::remove_file(&tmp);
    n
}

fn main() {
    let mut f = std::fs::File::create(LOG).expect("create log");
    let t_start = Instant::now();

    let ply = if Path::new(PRIMARY_PLY).exists() { PRIMARY_PLY } else { FALLBACK_PLY };
    log_line(&mut f, "=== TILE-CODEC-1 bench ===");
    log_line(&mut f, &format!("PLY: {ply}"));

    let scene = match catetus_ply::read_ply(&PathBuf::from(ply)) {
        Ok(s) => s,
        Err(e) => {
            log_line(&mut f, &format!("FATAL: read_ply failed: {e}"));
            std::process::exit(2);
        }
    };
    let n_total = scene.splats.len();
    log_line(&mut f, &format!("loaded {n_total} splats in {:.1}s", t_start.elapsed().as_secs_f64()));

    // --- 2. Plan the tileset (default config) ---
    let cfg = TilesetConfig::default();
    log_line(
        &mut f,
        &format!(
            "TilesetConfig::default {{ lods_per_node:{}, coarsest_target:{}, proxy_cap:{} }}",
            cfg.lods_per_node, cfg.coarsest_target, cfg.proxy_cap
        ),
    );
    let plan = match plan_tileset(&scene, &cfg) {
        Ok(p) => p,
        Err(e) => {
            log_line(&mut f, &format!("FATAL: plan_tileset failed: {e}"));
            std::process::exit(3);
        }
    };
    let n_tiles = plan.payloads.len();
    let counts: Vec<usize> = plan.payloads.iter().map(|p| p.scene.splats.len()).collect();
    let sum_tile_splats: usize = counts.iter().sum();
    let min_c = counts.iter().copied().min().unwrap_or(0);
    let max_c = counts.iter().copied().max().unwrap_or(0);
    log_line(
        &mut f,
        &format!(
            "tileset: {n_tiles} tiles, lod_levels={}, per-tile splats min={min_c} max={max_c} sum={sum_tile_splats}",
            plan.lod_meta.lod_levels
        ),
    );

    let glb_codec = GlbTileCodec::new(TilePreset::Balanced);
    let glb_bytes = |s: &SplatScene| -> usize {
        let tb = glb_codec.encode(s);
        tb.bytes.len() + tb.sidecar.as_ref().map(|v| v.len()).unwrap_or(0)
    };

    // --- 3. Representative sample for the per-tile OURS table ---
    // smallest-count tile (root coarse proxy class), largest (~proxy_cap leaf),
    // plus ~10 spanning the count range.
    let mut idx_by_count: Vec<usize> = (0..n_tiles).collect();
    idx_by_count.sort_by_key(|&i| counts[i]);
    let mut sample: Vec<usize> = Vec::new();
    if !idx_by_count.is_empty() {
        sample.push(idx_by_count[0]);
        sample.push(*idx_by_count.last().unwrap());
        let k = 10usize.min(idx_by_count.len());
        for j in 0..k {
            let pos = (j * idx_by_count.len().saturating_sub(1)) / k.max(1);
            sample.push(idx_by_count[pos]);
        }
    }
    sample.sort_unstable();
    sample.dedup();

    log_line(&mut f, "");
    log_line(&mut f, "--- TABLE 1: per-tile OURS sizes (sampled). SOG-per-tile: BLOCKED (no local encoder) ---");
    log_line(&mut f, "file_idx   splats    ours_glb(B)   bytes_per_splat");
    let mut sample_ours_total = 0usize;
    for &i in &sample {
        let s = &plan.payloads[i].scene;
        let o = glb_bytes(s);
        sample_ours_total += o;
        let bps = if counts[i] > 0 { o as f64 / counts[i] as f64 } else { 0.0 };
        log_line(
            &mut f,
            &format!("{:>7}  {:>7}  {:>13}  {:>12.2}", plan.payloads[i].file_index, counts[i], o, bps),
        );
    }
    log_line(&mut f, &format!("sample ours total: {sample_ours_total} B over {} tiles", sample.len()));

    // --- 4. Monolithic single-file baselines ---
    log_line(&mut f, "");
    log_line(&mut f, "--- monolithic single-file baselines (whole scene) ---");
    let mono_t = Instant::now();
    let mono_glb = mono_glb_bytes(&scene, &balanced_opts()).unwrap_or_else(|e| {
        log_line(&mut f, &format!("WARN mono GLB failed: {e}"));
        0
    });
    log_line(
        &mut f,
        &format!("mono GLB (ours, Balanced opts): {mono_glb} B  ({:.1}s)", mono_t.elapsed().as_secs_f64()),
    );
    // Real competitor SOG of the same scene, measured from disk (not encoded here).
    let mono_sog = std::fs::metadata(MONO_SOG_DISK).map(|m| m.len()).unwrap_or(0);
    log_line(
        &mut f,
        &format!("mono SOG (SuperSplat, from disk {MONO_SOG_DISK}): {mono_sog} B"),
    );

    // --- 5. Overhead tax (OURS): sum over ALL tiles / monolithic ---
    log_line(&mut f, "");
    log_line(&mut f, "--- ALL-tiles OURS total + overhead tax ---");
    let all_t = Instant::now();
    let mut all_ours = 0u64;
    for (k, p) in plan.payloads.iter().enumerate() {
        all_ours += glb_bytes(&p.scene) as u64;
        if k % 50 == 0 {
            let _ = writeln!(
                f,
                "  progress {k}/{n_tiles}  ours_so_far={all_ours} ({:.0}s)",
                all_t.elapsed().as_secs_f64()
            );
            let _ = f.flush();
        }
    }
    log_line(&mut f, &format!("all-tiles OURS encode done in {:.1}s", all_t.elapsed().as_secs_f64()));

    let tax_ours = if mono_glb > 0 { all_ours as f64 / mono_glb as f64 } else { f64::NAN };
    let mono_ratio = if mono_glb > 0 { mono_sog as f64 / mono_glb as f64 } else { f64::NAN };
    // What SOG-tiled would need to be for ours to still win after tiling:
    // ours-tiled = all_ours. SOG mono = mono_sog. If SOG's per-tile tax were T_sog,
    // SOG-tiled = mono_sog * T_sog. Ours beats SOG-tiled iff all_ours <= mono_sog*T_sog,
    // i.e. T_sog >= all_ours/mono_sog. Report that break-even SOG tax.
    let sog_breakeven_tax = if mono_sog > 0 { all_ours as f64 / mono_sog as f64 } else { f64::NAN };

    log_line(&mut f, "");
    log_line(&mut f, &format!("MONO   ours_glb={mono_glb} B   sog={mono_sog} B   sog/ours={mono_ratio:.3}"));
    log_line(&mut f, &format!("TILED  ours_glb(all tiles)={all_ours} B   (SOG-tiled: NOT LOCALLY MEASURABLE)"));
    log_line(&mut f, "");
    log_line(
        &mut f,
        &format!("OVERHEAD TAX (ours): sum(tiles)/mono = {tax_ours:.4}  (= +{:.1}%)", (tax_ours - 1.0) * 100.0),
    );
    log_line(
        &mut f,
        &format!(
            "BREAK-EVEN: ours-tiled beats SOG-tiled only if SOG's own tiling tax >= {sog_breakeven_tax:.3} (= +{:.1}%)",
            (sog_breakeven_tax - 1.0) * 100.0
        ),
    );
    log_line(&mut f, "");
    log_line(
        &mut f,
        &format!(
            "VERDICT mono: ours {} SOG (sog/ours={mono_ratio:.3})",
            if mono_ratio >= 1.0 { "BEATS" } else { "LOSES to" }
        ),
    );

    // ====================================================================
    // SHARED-CODEBOOK-STREAM: the three-way tiled comparison.
    //   (a) FP32 GlbTileCodec  — all_ours above (180 B/splat SH-rest)
    //   (b) per-tile sh_rest_quant — 45 B/splat SH-rest, no shared codebook
    //   (c) shared-palette codec — 2 B/splat indices + ONE root codebook
    // ====================================================================
    log_line(&mut f, "");
    log_line(&mut f, "=== SHARED-CODEBOOK-STREAM three-way tiled comparison ===");

    // --- Build the scene-global VQ45 codebook ONCE over the whole scene. ---
    let cb_t = Instant::now();
    let codebook = match SharedCodebook::build(&scene, 4096, 5, 0xC0DE) {
        Ok(Some(cb)) => Some(cb),
        Ok(None) => {
            log_line(&mut f, "scene is DC-only (no SH-rest) — shared palette N/A");
            None
        }
        Err(e) => {
            log_line(&mut f, &format!("WARN SharedCodebook::build failed: {e}"));
            None
        }
    };
    let mut root_codebook_bytes = 0usize;
    if let Some(cb) = &codebook {
        root_codebook_bytes = cb.root_sidecar_bytes().len();
        log_line(
            &mut f,
            &format!(
                "shared codebook built in {:.1}s: K={} N={} sh_degree={} \
                 root {} = {} B (written ONCE for the whole tileset)",
                cb_t.elapsed().as_secs_f64(),
                cb.palette_size(),
                cb.n_splats(),
                cb.sh_degree(),
                SHARED_PALETTE_FILENAME,
                root_codebook_bytes,
            ),
        );
    }

    // --- Per-tile sh_rest_quant (b) + shared-palette (c) over ALL tiles. ---
    let mut all_shq = 0u64;
    let mut all_shp_glb = 0u64; // tile GLBs (SH-rest elided)
    let mut all_shp_idx = 0u64; // per-tile .shpalx index sidecars
    let shp_codec = codebook.as_ref().map(SharedPaletteTileCodec::new);
    let rung_t = Instant::now();
    for (k, p) in plan.payloads.iter().enumerate() {
        all_shq += sh_rest_quant_tile_bytes(&p.scene) as u64;
        if let Some(codec) = &shp_codec {
            let tb = codec.encode_tile(&p.scene, &p.origins);
            all_shp_glb += tb.bytes.len() as u64;
            all_shp_idx += tb.sidecar.as_ref().map(|v| v.len()).unwrap_or(0) as u64;
        }
        if k % 50 == 0 {
            let _ = writeln!(
                f,
                "  rung progress {k}/{n_tiles} ({:.0}s)",
                rung_t.elapsed().as_secs_f64()
            );
            let _ = f.flush();
        }
    }
    log_line(
        &mut f,
        &format!("rungs (b)+(c) encoded in {:.1}s", rung_t.elapsed().as_secs_f64()),
    );

    let all_shp = all_shp_glb + all_shp_idx + root_codebook_bytes as u64;
    let sum_splats = sum_tile_splats.max(1) as f64;

    log_line(&mut f, "");
    log_line(&mut f, "--- THREE-WAY TILED BYTE TABLE (bonsai, all tiles) ---");
    log_line(
        &mut f,
        &format!(
            "{:<28} {:>16} {:>16}",
            "rung", "total_bytes", "bytes_per_splat"
        ),
    );
    log_line(
        &mut f,
        &format!(
            "{:<28} {:>16} {:>16.2}",
            "(a) FP32 GlbTileCodec", all_ours, all_ours as f64 / sum_splats
        ),
    );
    log_line(
        &mut f,
        &format!(
            "{:<28} {:>16} {:>16.2}",
            "(b) per-tile sh_rest_quant", all_shq, all_shq as f64 / sum_splats
        ),
    );
    if codebook.is_some() {
        log_line(
            &mut f,
            &format!(
                "{:<28} {:>16} {:>16.2}",
                "(c) shared-palette TOTAL", all_shp, all_shp as f64 / sum_splats
            ),
        );
        log_line(
            &mut f,
            &format!(
                "      ├─ tile GLBs (SH elided): {} B ({:.2} B/splat)",
                all_shp_glb,
                all_shp_glb as f64 / sum_splats
            ),
        );
        log_line(
            &mut f,
            &format!(
                "      ├─ per-tile .shpalx idx : {} B ({:.2} B/splat)",
                all_shp_idx,
                all_shp_idx as f64 / sum_splats
            ),
        );
        log_line(
            &mut f,
            &format!("      └─ shared root codebook : {root_codebook_bytes} B (once)"),
        );
    }

    log_line(&mut f, "");
    log_line(
        &mut f,
        &format!("mono SF-with-palette (from disk): see bonsai_sf.glb + .shpal if present"),
    );
    // Reference single-file SF-with-palette artifacts on disk, for context.
    let sf_glb = std::fs::metadata("/Users/montabano1/Downloads/bonsai_comparison/bonsai_sf.glb")
        .map(|m| m.len())
        .unwrap_or(0);
    let sf_shpal =
        std::fs::metadata("/Users/montabano1/Downloads/bonsai_comparison/bonsai_sf.glb.shpal")
            .map(|m| m.len())
            .unwrap_or(0);
    if sf_glb > 0 {
        log_line(
            &mut f,
            &format!(
                "mono SF-with-palette on disk: glb={sf_glb} + shpal={sf_shpal} = {} B",
                sf_glb + sf_shpal
            ),
        );
    }

    if codebook.is_some() && all_shp > 0 {
        let vs_fp32 = all_ours as f64 / all_shp as f64;
        let vs_shq = all_shq as f64 / all_shp.max(1) as f64;
        let vs_sog = mono_sog as f64 / all_shp as f64;
        log_line(&mut f, "");
        log_line(
            &mut f,
            &format!(
                "REDUCTION shared-palette vs FP32 tiled: {vs_fp32:.2}× smaller \
                 ({all_ours} → {all_shp} B)"
            ),
        );
        log_line(
            &mut f,
            &format!("REDUCTION shared-palette vs sh_rest_quant tiled: {vs_shq:.2}×"),
        );
        log_line(
            &mut f,
            &format!(
                "VERDICT shared-palette-tiled vs SOG-mono ({mono_sog} B): \
                 shared={all_shp} B → sog/shared={vs_sog:.3} ({})",
                if vs_sog >= 1.0 {
                    "shared BEATS SOG-mono"
                } else if vs_sog >= 0.8 {
                    "shared APPROACHES SOG-mono"
                } else {
                    "shared still larger than SOG-mono"
                }
            ),
        );
    }

    // Machine-readable sentinels.
    let _ = writeln!(f, "RESULT_N_TILES={n_tiles}");
    let _ = writeln!(f, "RESULT_TOTAL_SPLATS={n_total}");
    let _ = writeln!(f, "RESULT_SUM_TILE_SPLATS={sum_tile_splats}");
    let _ = writeln!(f, "RESULT_MAX_TILE_SPLATS={max_c}");
    let _ = writeln!(f, "RESULT_MONO_OURS={mono_glb}");
    let _ = writeln!(f, "RESULT_MONO_SOG={mono_sog}");
    let _ = writeln!(f, "RESULT_TILED_OURS={all_ours}");
    let _ = writeln!(f, "RESULT_TAX_OURS_X1000={}", (tax_ours * 1000.0) as i64);
    let _ = writeln!(f, "RESULT_MONO_RATIO_X1000={}", (mono_ratio * 1000.0) as i64);
    let _ = writeln!(f, "RESULT_SOG_BREAKEVEN_TAX_X1000={}", (sog_breakeven_tax * 1000.0) as i64);
    let _ = writeln!(f, "RESULT_TILED_SHQ={all_shq}");
    let _ = writeln!(f, "RESULT_TILED_SHP_TOTAL={all_shp}");
    let _ = writeln!(f, "RESULT_TILED_SHP_GLB={all_shp_glb}");
    let _ = writeln!(f, "RESULT_TILED_SHP_IDX={all_shp_idx}");
    let _ = writeln!(f, "RESULT_SHP_ROOT_CODEBOOK={root_codebook_bytes}");
    let _ = writeln!(f, "RESULT_DONE=1");
    let _ = f.flush();
    log_line(&mut f, &format!("TOTAL wall {:.1}s", t_start.elapsed().as_secs_f64()));
}
