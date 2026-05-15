#![deny(clippy::all)]
//! `splatforge-lodge` — offline builder for the LODGE-style hierarchical
//! level-of-detail pyramid used by the SplatForge viewer.
//!
//! See `docs/perf/lodge-lod-spec.md` for the design memo. This crate
//! ships **Phase A.1** of that spec: the offline chunker. It takes a
//! trained 3DGS PLY and emits a `.lodge` directory containing
//!
//!   - a `manifest.json` describing the LOD pyramid + spatial chunking,
//!   - per-level PLY files (`level_<l>/chunk_<c>.ply`) sized to be
//!     individually streamable.
//!
//! The chunker does NOT do the full LODGE depth-aware smoothing +
//! prune-+-fine-tune cycle from the paper. That requires gsplat
//! rasterization on a GPU and is Phase A.2 of the spec. Phase A.1
//! ships the *structure* — manifest schema, chunked layout, importance-
//! weighted decimation — so that:
//!
//!   1. The viewer team can start wiring the runtime side
//!      (`packages/viewer/src/lodge/`) against a real manifest format.
//!   2. The 4-tile Sweet Corals 11.4M OOM scenario has a path forward:
//!      pick a coarser level for far chunks and the per-frame VRAM
//!      footprint drops.
//!
//! Decimation algorithm (Phase A.1):
//!
//!   - Per-level uniform 3D grid bin (sized so the expected
//!     splat-per-cell count is the level's coarsening ratio, default 2×).
//!   - Per cell, keep the splat with the highest *importance score*:
//!     `s_i = opacity_i * det(scale_i)^(2/3)`. This is the same score
//!     LODGE uses for RadSplat-style importance pruning (paper §3.1)
//!     and the same one MesonGS++ uses for high-CRF mask selection.
//!   - Drop the rest. No re-centering / Gaussian merging — we KEEP the
//!     winning splat verbatim. This is the conservative "smooth-then-
//!     prune" approximation: we skip the smooth step entirely (which
//!     requires fine-tune to recover quality) and just prune. PSNR
//!     loss vs. full LODGE pipeline is bounded by ~0.3-0.5 dB at L1
//!     according to LODGE table 3 (the prune-only "LOD d=10" row is
//!     0.17 dB below their full pipeline at SmallCity).
//!
//! Spatial chunking (per level):
//!
//!   - We compute a 3D Morton code per splat over the level's bounding
//!     box (16 bits/axis = 48-bit code).
//!   - Sort by Morton code, then split into chunks of
//!     `chunk_target_splats` (default 100_000). This gives roughly-cubic
//!     spatial groupings that map cleanly onto LODGE's K-means-over-
//!     camera-positions chunking once the runtime selector lands
//!     (Phase A.2 — we'll re-cluster by camera proximity then, but the
//!     on-disk layout is identical: a fixed list of per-chunk PLY
//!     pointers in the manifest).
//!
//! The manifest schema is documented in
//! `docs/perf/lodge-lod-spec.md` §A.1 (appended by the Phase A.1 patch).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splatforge_core::{Splat, SplatScene};
use splatforge_ply::{read_ply, write_ply};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LodgeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("ply: {0}")]
    Ply(#[from] splatforge_ply::PlyError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid input: {0}")]
    Invalid(String),
}

/// Build-time options for [`build`].
#[derive(Debug, Clone)]
pub struct BuildOpts {
    /// Number of LOD levels including level 0 (the original). Must be ≥ 1.
    /// Default: 6 (gives a 32× decimation at the top).
    pub levels: usize,
    /// Target splat count at the coarsest level. The number of levels is
    /// also bounded from above by this (we stop subdividing once the
    /// level's count falls below `target_top`).
    /// Default: 100_000.
    pub target_top: usize,
    /// Coarsening ratio between consecutive levels: level `l+1` has
    /// `~level_l_count / coarsen_ratio` splats. The paper-faithful choice
    /// is 2.0 (geometric pyramid); larger values produce a steeper pyramid
    /// at the cost of bigger PSNR jumps between levels.
    /// Default: 2.0.
    pub coarsen_ratio: f32,
    /// Target splat count per spatial chunk. Each level is split into
    /// roughly `level_count / chunk_target_splats` chunks via Morton-
    /// sorted slicing.
    /// Default: 100_000.
    pub chunk_target_splats: usize,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            levels: 6,
            target_top: 100_000,
            coarsen_ratio: 2.0,
            chunk_target_splats: 100_000,
        }
    }
}

/// On-disk manifest format (`manifest.json`). Versioned for forward
/// compatibility with Phase A.2 (per-view selector) and Phase A.3
/// (per-frame LOD selection compute pass).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LodgeManifest {
    /// Manifest schema version. Bump on breaking changes.
    pub version: u32,
    /// Source PLY filename (for provenance only — readers don't open it).
    pub source: String,
    /// Splat count of the original PLY (= level 0).
    pub original_splat_count: usize,
    /// Scene axis-aligned bounding box `[min_xyz, max_xyz]`. Reused by
    /// the runtime selector to clip the camera-distance bands to scene
    /// extent.
    pub bbox: [[f32; 3]; 2],
    /// Pyramid levels, ordered fine → coarse. `levels[0]` is the
    /// original; `levels.last()` is the coarsest.
    pub levels: Vec<LevelDescriptor>,
}

/// One level of the LOD pyramid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelDescriptor {
    /// Level index (0 = finest = original).
    pub level: u32,
    /// Total splat count at this level (sum of all chunk splat counts).
    pub splat_count: usize,
    /// Reduction ratio relative to level 0 (e.g. 0.5 means this level
    /// has half as many splats as the original).
    pub reduction: f32,
    /// Approximate Nyquist depth `d_l` for this level — the camera
    /// distance at which a splat at this level is visually sufficient.
    /// Phase A.1 derives this heuristically from level index and the
    /// scene's bounding-box diagonal. The runtime selector (Phase A.2)
    /// uses these as the band edges in eq. 2 of LODGE.
    pub depth_threshold: f32,
    /// Chunk descriptors, ordered by Morton sweep.
    pub chunks: Vec<ChunkDescriptor>,
}

/// One spatial chunk inside one LOD level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDescriptor {
    /// Chunk index within the level.
    pub index: u32,
    /// Relative path (from the manifest's directory) to the chunk PLY.
    pub path: String,
    /// Splat count in this chunk.
    pub splat_count: usize,
    /// Chunk-local AABB `[min_xyz, max_xyz]`.
    pub bbox: [[f32; 3]; 2],
    /// Centroid `[x, y, z]` of the splats in this chunk. Used by the
    /// runtime selector to pick the two nearest chunks per frame
    /// (LODGE eq. 4 boundary blend).
    pub centroid: [f32; 3],
    /// Bounding-sphere radius (max distance from centroid to any splat
    /// in the chunk). Used to expand the per-chunk active set for
    /// cameras near the chunk edge (LODGE eq. 2 with `d_l` offset by
    /// `r_k`).
    pub radius: f32,
    /// blake3 hex digest of the chunk PLY bytes. Lets streaming clients
    /// detect a stale/swapped chunk without re-fetching the whole
    /// manifest.
    pub blake3: String,
}

/// Importance score used to decide which splat survives per cell during
/// decimation. LODGE / RadSplat use `max-over-training-views alpha`;
/// without renderer + cameras at this stage we approximate with
/// `opacity * det(scale)^(2/3)`. The exponent makes the score scale
/// linearly with on-screen area at a fixed view distance, which is the
/// quantity LODGE's depth-aware-smooth filter is also tracking.
#[inline]
fn importance(s: &Splat) -> f32 {
    let det = (s.scale[0].abs() * s.scale[1].abs() * s.scale[2].abs()).max(f32::MIN_POSITIVE);
    // (det)^(2/3) = cbrt(det^2)
    let area_scale = det.cbrt().powi(2);
    s.opacity.clamp(0.0, 1.0) * area_scale
}

fn scene_bbox(splats: &[Splat]) -> ([f32; 3], [f32; 3]) {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for s in splats {
        for i in 0..3 {
            let p = s.position[i];
            if p < mn[i] {
                mn[i] = p;
            }
            if p > mx[i] {
                mx[i] = p;
            }
        }
    }
    if !mn[0].is_finite() {
        // Empty scene: clamp to zero so downstream math doesn't NaN.
        mn = [0.0; 3];
        mx = [0.0; 3];
    }
    (mn, mx)
}

/// Compute the 48-bit 3D Morton code of `position` over `[mn, mx]`,
/// 16 bits per axis. Identical to the morton-permute path in the CLI;
/// kept private here to avoid an extra crate dependency.
fn morton48(position: [f32; 3], mn: [f32; 3], extent: [f32; 3]) -> u64 {
    #[inline]
    fn spread16(v: u32) -> u64 {
        let mut x = (v & 0xFFFF) as u64;
        x = (x | (x << 32)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555;
        x
    }
    let qx = (((position[0] - mn[0]) / extent[0]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32;
    let qy = (((position[1] - mn[1]) / extent[1]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32;
    let qz = (((position[2] - mn[2]) / extent[2]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32;
    let qx = qx.min(65535);
    let qy = qy.min(65535);
    let qz = qz.min(65535);
    spread16(qx) | (spread16(qy) << 1) | (spread16(qz) << 2)
}

/// Decimate `indices` (into `splats`) to roughly `target_count` survivors
/// via importance-weighted uniform 3D-grid binning. Returns a new index
/// list.
///
/// The grid is sized so that the expected splat-per-cell count is
/// `splats_in / target_count` — i.e. the reduction ratio. Per occupied
/// cell, we keep the highest-importance splat. The cell side length is
/// derived from the bounding box of the *input indices*, not the full
/// scene — this prevents the grid from collapsing onto a single cell
/// when we decimate a small spatial subset.
fn decimate_to(splats: &[Splat], indices: &[u32], target_count: usize) -> Vec<u32> {
    if indices.is_empty() || target_count == 0 {
        return Vec::new();
    }
    if target_count >= indices.len() {
        return indices.to_vec();
    }

    // Bounding box over this subset.
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for &idx in indices {
        let p = splats[idx as usize].position;
        for i in 0..3 {
            if p[i] < mn[i] {
                mn[i] = p[i];
            }
            if p[i] > mx[i] {
                mx[i] = p[i];
            }
        }
    }
    let extent = [
        (mx[0] - mn[0]).max(f32::MIN_POSITIVE),
        (mx[1] - mn[1]).max(f32::MIN_POSITIVE),
        (mx[2] - mn[2]).max(f32::MIN_POSITIVE),
    ];

    // Cells along the longest axis = cbrt(target_count) is the ideal
    // for a perfectly uniform distribution. Real scenes are clustered,
    // so we OVER-resolve by ~1.5× and let empty cells take care of the
    // density variance. After the per-cell argmax, the survivor count
    // is bounded above by the number of OCCUPIED cells, which is what
    // we actually want at the target.
    //
    // We iterate the resolution: pick a starting grid edge, count
    // occupied cells, halve or double the edge until the survivor count
    // lands inside [target/1.3, target*1.3]. Caps at 8 iterations so
    // pathological distributions terminate.
    let mut cells_per_axis: u32 = ((target_count as f32).powf(1.0 / 3.0) * 1.5).ceil() as u32;
    cells_per_axis = cells_per_axis.max(1);

    // Track best-so-far so we don't oscillate forever on lattice-like
    // inputs (where every grid resolution either over- or under-shoots
    // the target). Best = closest to `target_count` measured in
    // |log2(n/target)|.
    let mut best: Option<(f32, Vec<u32>)> = None;
    let log_target = (target_count as f32).ln();
    for _ in 0..12 {
        let cand = grid_argmax(splats, indices, mn, extent, cells_per_axis);
        let n = cand.len();
        if n > 0 {
            let dist = ((n as f32).ln() - log_target).abs();
            let take = match &best {
                None => true,
                Some((d, _)) => dist < *d,
            };
            if take {
                best = Some((dist, cand));
            }
        }
        let n = best.as_ref().map(|(_, v)| v.len()).unwrap_or(n);
        if n > (target_count as f32 * 1.15) as usize {
            let next = ((cells_per_axis as f32) * 0.85).floor() as u32;
            if next == cells_per_axis || next == 0 {
                break;
            }
            cells_per_axis = next.max(1);
        } else if n < (target_count as f32 / 1.15) as usize {
            let next = ((cells_per_axis as f32) * 1.18).ceil() as u32;
            if next == cells_per_axis {
                break;
            }
            cells_per_axis = next;
        } else {
            break;
        }
    }
    best.map(|(_, v)| v)
        .unwrap_or_else(|| grid_argmax(splats, indices, mn, extent, cells_per_axis))
}

/// Per occupied cell of an `n³` axis-aligned grid over `[mn, mn+extent]`,
/// keep the highest-importance splat. Returns the indices of survivors
/// in Morton order (within the cell, deterministic by input index).
fn grid_argmax(
    splats: &[Splat],
    indices: &[u32],
    mn: [f32; 3],
    extent: [f32; 3],
    cells_per_axis: u32,
) -> Vec<u32> {
    use std::collections::HashMap;
    let n = cells_per_axis.max(1);
    let nf = n as f32;
    let mut best: HashMap<u64, (f32, u32)> = HashMap::with_capacity(indices.len().min(1 << 20));
    for &idx in indices {
        let s = &splats[idx as usize];
        let cx = ((((s.position[0] - mn[0]) / extent[0]).clamp(0.0, 0.999999)) * nf) as u32;
        let cy = ((((s.position[1] - mn[1]) / extent[1]).clamp(0.0, 0.999999)) * nf) as u32;
        let cz = ((((s.position[2] - mn[2]) / extent[2]).clamp(0.0, 0.999999)) * nf) as u32;
        let key = (cx as u64) | ((cy as u64) << 22) | ((cz as u64) << 44);
        let score = importance(s);
        match best.get_mut(&key) {
            None => {
                best.insert(key, (score, idx));
            }
            Some((bs, bi)) => {
                if score > *bs || (score == *bs && idx < *bi) {
                    *bs = score;
                    *bi = idx;
                }
            }
        }
    }
    let mut out: Vec<u32> = best.into_values().map(|(_, idx)| idx).collect();
    out.sort_unstable();
    out
}

/// Split `indices` (already in arbitrary order) into Morton-sorted
/// spatial chunks of at most `chunk_target` splats each. Returns a
/// vector of per-chunk index lists.
fn morton_chunk(
    splats: &[Splat],
    indices: &[u32],
    mn: [f32; 3],
    extent: [f32; 3],
    chunk_target: usize,
) -> Vec<Vec<u32>> {
    if indices.is_empty() {
        return Vec::new();
    }
    let chunk_target = chunk_target.max(1);
    let mut keyed: Vec<(u64, u32)> = indices
        .iter()
        .map(|&idx| (morton48(splats[idx as usize].position, mn, extent), idx))
        .collect();
    keyed.sort_unstable_by_key(|&(k, _)| k);

    // Aim for chunks of size ~chunk_target. Compute the chunk count
    // first so we don't end up with one tiny tail chunk: ceil(n /
    // chunk_target) chunks, evenly split.
    let n = keyed.len();
    let nchunks = n.div_ceil(chunk_target).max(1);
    let mut chunks = Vec::with_capacity(nchunks);
    let base = n / nchunks;
    let rem = n % nchunks;
    let mut start = 0usize;
    for i in 0..nchunks {
        let take = base + if i < rem { 1 } else { 0 };
        let slice = &keyed[start..start + take];
        chunks.push(slice.iter().map(|&(_, idx)| idx).collect::<Vec<u32>>());
        start += take;
    }
    chunks
}

/// Compute level depth thresholds.
///
/// Phase A.1 heuristic: spread the per-level `d_l` linearly between
/// `0.0` (= "anywhere") at level 0 and `scene_diag * 1.5` (= "from
/// outside the scene") at the coarsest level. Phase A.2 will replace
/// this with the LODGE greedy-search-over-training-views routine, but
/// the manifest field is in place now so the runtime can consume it
/// without a format change.
fn depth_thresholds(levels: usize, scene_diag: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(levels);
    let max_depth = scene_diag * 1.5;
    if levels <= 1 {
        out.push(0.0);
        return out;
    }
    for l in 0..levels {
        let frac = l as f32 / (levels - 1) as f32;
        out.push(frac * max_depth);
    }
    out
}

fn chunk_stats(splats: &[Splat], indices: &[u32]) -> ([[f32; 3]; 2], [f32; 3], f32) {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    let mut sum = [0.0f64; 3];
    for &idx in indices {
        let p = splats[idx as usize].position;
        for i in 0..3 {
            if p[i] < mn[i] {
                mn[i] = p[i];
            }
            if p[i] > mx[i] {
                mx[i] = p[i];
            }
            sum[i] += p[i] as f64;
        }
    }
    let n = indices.len().max(1) as f64;
    let centroid = [
        (sum[0] / n) as f32,
        (sum[1] / n) as f32,
        (sum[2] / n) as f32,
    ];
    let mut radius2 = 0.0f32;
    for &idx in indices {
        let p = splats[idx as usize].position;
        let dx = p[0] - centroid[0];
        let dy = p[1] - centroid[1];
        let dz = p[2] - centroid[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 > radius2 {
            radius2 = r2;
        }
    }
    let radius = radius2.sqrt();
    ([mn, mx], centroid, radius)
}

/// Build a chunk-only `SplatScene` referencing the requested indices.
fn subset_scene(scene: &SplatScene, indices: &[u32]) -> SplatScene {
    let mut out = SplatScene::new();
    out.coordinate_system = scene.coordinate_system;
    out.temporal_mode = scene.temporal_mode;
    out.splats.reserve(indices.len());
    for &idx in indices {
        out.splats.push(scene.splats[idx as usize].clone());
    }
    out
}

/// Build a LODGE pyramid from `scene` into `output_dir`. The directory
/// is created if it doesn't exist; existing contents may be overwritten.
///
/// Returns the manifest as written, with paths relative to `output_dir`.
pub fn build(scene: &SplatScene, output_dir: &Path, opts: &BuildOpts) -> Result<LodgeManifest, LodgeError> {
    if scene.is_empty() {
        return Err(LodgeError::Invalid("empty scene".into()));
    }
    if opts.levels == 0 {
        return Err(LodgeError::Invalid("levels must be ≥ 1".into()));
    }
    if opts.coarsen_ratio <= 1.0 {
        return Err(LodgeError::Invalid(
            "coarsen_ratio must be > 1.0 (each coarser level reduces count)".into(),
        ));
    }

    std::fs::create_dir_all(output_dir)?;

    let n0 = scene.splats.len();
    let (mn, mx) = scene_bbox(&scene.splats);
    let extent = [
        (mx[0] - mn[0]).max(f32::MIN_POSITIVE),
        (mx[1] - mn[1]).max(f32::MIN_POSITIVE),
        (mx[2] - mn[2]).max(f32::MIN_POSITIVE),
    ];
    let scene_diag = (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).sqrt();

    // Build per-level index lists. Level 0 = all splats.
    let level_indices: Vec<Vec<u32>> = {
        let mut all = Vec::with_capacity(n0);
        for i in 0..n0 {
            all.push(i as u32);
        }
        let mut levels: Vec<Vec<u32>> = vec![all];
        for _l in 1..opts.levels {
            let prev = levels.last().unwrap();
            let target =
                ((prev.len() as f32) / opts.coarsen_ratio).round() as usize;
            // Stop subdividing once we hit the target_top floor.
            if target < opts.target_top.max(1) {
                let final_target = opts.target_top.max(1);
                if prev.len() <= final_target {
                    break;
                }
                let dec = decimate_to(&scene.splats, prev, final_target);
                levels.push(dec);
                break;
            }
            let dec = decimate_to(&scene.splats, prev, target);
            if dec.len() == prev.len() {
                // Can't decimate further (probably extreme clustering).
                break;
            }
            levels.push(dec);
        }
        levels
    };

    let thresholds = depth_thresholds(level_indices.len(), scene_diag);

    // Write each (level, chunk) PLY and assemble level descriptors.
    let mut level_descs: Vec<LevelDescriptor> = Vec::with_capacity(level_indices.len());
    for (l, idxs) in level_indices.iter().enumerate() {
        let level_dir_rel = format!("level_{l}");
        let level_dir_abs = output_dir.join(&level_dir_rel);
        std::fs::create_dir_all(&level_dir_abs)?;

        let chunks = morton_chunk(&scene.splats, idxs, mn, extent, opts.chunk_target_splats);
        let mut chunk_descs: Vec<ChunkDescriptor> = Vec::with_capacity(chunks.len());
        for (c, cidx) in chunks.iter().enumerate() {
            let chunk_rel = format!("{level_dir_rel}/chunk_{c:04}.ply");
            let chunk_path = output_dir.join(&chunk_rel);
            let chunk_scene = subset_scene(scene, cidx);
            write_ply(&chunk_scene, &chunk_path)?;

            let bytes = std::fs::read(&chunk_path)?;
            let hash = blake3::hash(&bytes).to_hex().to_string();

            let (bbox, centroid, radius) = chunk_stats(&scene.splats, cidx);
            chunk_descs.push(ChunkDescriptor {
                index: c as u32,
                path: chunk_rel,
                splat_count: cidx.len(),
                bbox,
                centroid,
                radius,
                blake3: hash,
            });
        }
        let reduction = idxs.len() as f32 / n0 as f32;
        level_descs.push(LevelDescriptor {
            level: l as u32,
            splat_count: idxs.len(),
            reduction,
            depth_threshold: *thresholds.get(l).unwrap_or(&0.0),
            chunks: chunk_descs,
        });
    }

    let manifest = LodgeManifest {
        version: 1,
        source: "".into(), // caller fills in via build_with_source if desired
        original_splat_count: n0,
        bbox: [mn, mx],
        levels: level_descs,
    };

    let manifest_path = output_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    Ok(manifest)
}

/// Convenience wrapper that reads `input_ply`, builds the pyramid, and
/// stamps `manifest.source` with the input filename.
pub fn build_from_ply(
    input_ply: &Path,
    output_dir: &Path,
    opts: &BuildOpts,
) -> Result<LodgeManifest, LodgeError> {
    let scene = read_ply(input_ply)?;
    let mut manifest = build(&scene, output_dir, opts)?;
    if let Some(name) = input_ply.file_name().and_then(|s| s.to_str()) {
        manifest.source = name.to_string();
        let manifest_path = output_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, json)?;
    }
    Ok(manifest)
}

/// Load a manifest from `output_dir/manifest.json`.
pub fn read_manifest(output_dir: &Path) -> Result<LodgeManifest, LodgeError> {
    let manifest_path = output_dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)?;
    let m: LodgeManifest = serde_json::from_slice(&bytes)?;
    Ok(m)
}

/// Reassemble all chunks of level `level` into a flat `SplatScene`. The
/// returned scene's splats are concatenated in chunk order (chunks in
/// manifest order, splats inside each chunk in PLY-on-disk order). Use
/// this for round-trip testing or for diagnostic exports.
///
/// **Note.** Level 0's reassembly is NOT byte-identical to the input
/// PLY because the chunker re-orders splats by Morton code. The splat
/// SET is preserved exactly (no addition, no loss, no per-splat
/// modification at level 0).
pub fn reassemble_level(output_dir: &Path, level: u32) -> Result<SplatScene, LodgeError> {
    let manifest = read_manifest(output_dir)?;
    let lvl = manifest
        .levels
        .iter()
        .find(|l| l.level == level)
        .ok_or_else(|| LodgeError::Invalid(format!("level {level} not in manifest")))?;
    let mut out = SplatScene::new();
    out.splats.reserve(lvl.splat_count);
    for chunk in &lvl.chunks {
        let chunk_path = output_dir.join(&chunk.path);
        let chunk_scene = read_ply(&chunk_path)?;
        // Optional integrity check.
        let bytes = std::fs::read(&chunk_path)?;
        let hash = blake3::hash(&bytes).to_hex().to_string();
        if hash != chunk.blake3 {
            return Err(LodgeError::Invalid(format!(
                "blake3 mismatch on {}: manifest={} disk={}",
                chunk.path, chunk.blake3, hash
            )));
        }
        out.splats.extend(chunk_scene.splats);
    }
    if out.splats.len() != lvl.splat_count {
        return Err(LodgeError::Invalid(format!(
            "reassemble level {level}: expected {} splats, got {}",
            lvl.splat_count,
            out.splats.len()
        )));
    }
    Ok(out)
}

/// Reassemble level `level` of the LODGE pyramid into a flat PLY at
/// `out_ply`. Convenience for the CLI `lodge unpack` command and for
/// round-trip sanity checks.
pub fn unpack_level_to_ply(
    output_dir: &Path,
    level: u32,
    out_ply: &Path,
) -> Result<usize, LodgeError> {
    let scene = reassemble_level(output_dir, level)?;
    let n = scene.splats.len();
    write_ply(&scene, out_ply)?;
    Ok(n)
}

/// Total bytes on disk for the manifest + all chunk files inside
/// `output_dir`. Used for the CLI summary print.
pub fn total_on_disk_bytes(output_dir: &Path) -> Result<u64, LodgeError> {
    let manifest = read_manifest(output_dir)?;
    let mut total: u64 = 0;
    let manifest_bytes = std::fs::metadata(output_dir.join("manifest.json"))?.len();
    total = total.saturating_add(manifest_bytes);
    for lvl in &manifest.levels {
        for chunk in &lvl.chunks {
            let bytes = std::fs::metadata(output_dir.join(&chunk.path))?.len();
            total = total.saturating_add(bytes);
        }
    }
    Ok(total)
}

/// Helper for tests / external utilities: pull out the [`PathBuf`] of
/// every chunk file in the manifest.
pub fn all_chunk_paths(manifest: &LodgeManifest, root: &Path) -> Vec<PathBuf> {
    manifest
        .levels
        .iter()
        .flat_map(|l| l.chunks.iter().map(|c| root.join(&c.path)))
        .collect()
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use splatforge_core::{Color, Splat};
    use tempfile::tempdir;

    fn make_splat(x: f32, y: f32, z: f32, opacity: f32, scale: f32) -> Splat {
        Splat {
            position: [x, y, z],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [scale, scale, scale],
            opacity,
            color: Color::Rgb([0.5, 0.5, 0.5]),
        }
    }

    fn make_scene(n: usize) -> SplatScene {
        // Generate a deterministic point cloud on a 3D lattice plus a few
        // high-importance outliers, so the decimator has clear winners.
        let mut scene = SplatScene::new();
        let side = (n as f32).cbrt().ceil() as usize;
        let mut count = 0;
        for ix in 0..side {
            for iy in 0..side {
                for iz in 0..side {
                    if count >= n {
                        break;
                    }
                    let x = ix as f32;
                    let y = iy as f32;
                    let z = iz as f32;
                    // Importance varies so the per-cell argmax has something
                    // to do.
                    let op = 0.1 + 0.9 * ((ix * 7 + iy * 13 + iz * 19) % 17) as f32 / 17.0;
                    let sc = 0.01 + 0.05 * ((count % 7) as f32 / 7.0);
                    scene.splats.push(make_splat(x, y, z, op, sc));
                    count += 1;
                }
            }
        }
        scene
    }

    #[test]
    fn importance_is_nonnegative_and_monotone() {
        let a = make_splat(0.0, 0.0, 0.0, 0.5, 0.01);
        let b = make_splat(0.0, 0.0, 0.0, 0.5, 0.05);
        let ia = super::importance(&a);
        let ib = super::importance(&b);
        assert!(ia >= 0.0);
        assert!(ib > ia);
    }

    #[test]
    fn build_roundtrip_preserves_total_splat_count_at_level0() {
        let scene = make_scene(1000);
        let dir = tempdir().unwrap();
        let opts = BuildOpts {
            levels: 4,
            target_top: 100,
            coarsen_ratio: 2.0,
            chunk_target_splats: 250,
        };
        let manifest = build(&scene, dir.path(), &opts).unwrap();
        assert_eq!(manifest.levels[0].splat_count, scene.len());
        // The pyramid should monotonically shrink.
        let mut prev = manifest.levels[0].splat_count;
        for lvl in &manifest.levels[1..] {
            assert!(
                lvl.splat_count <= prev,
                "level {} count {} > prev {}",
                lvl.level,
                lvl.splat_count,
                prev
            );
            prev = lvl.splat_count;
        }
        // Reassemble level 0 — splat SET must be preserved.
        let reassembled = reassemble_level(dir.path(), 0).unwrap();
        assert_eq!(reassembled.len(), scene.len());
    }

    #[test]
    fn build_decimates_each_level_by_approx_coarsen_ratio() {
        let scene = make_scene(8000);
        let dir = tempdir().unwrap();
        let opts = BuildOpts {
            levels: 5,
            target_top: 100,
            coarsen_ratio: 2.0,
            chunk_target_splats: 1024,
        };
        let manifest = build(&scene, dir.path(), &opts).unwrap();
        // Each level should be within a factor of 2x of the requested
        // halving. Real distributions cluster so we allow a wide band.
        for w in manifest.levels.windows(2) {
            let ratio = w[0].splat_count as f32 / w[1].splat_count as f32;
            // Wide band: a perfect 3D lattice doesn't admit clean 2×
            // decimation because the grid argmax preserves at most one
            // splat per cell and cell counts only step in cubic units.
            // Real-scene PSNR-driven testing happens via the round-trip
            // sanity gate (see tests/ integration test).
            assert!(
                (1.1..=4.0).contains(&ratio),
                "consecutive-level ratio out of band: {ratio} (l{}={} l{}={})",
                w[0].level,
                w[0].splat_count,
                w[1].level,
                w[1].splat_count
            );
        }
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let scene = make_scene(500);
        let dir = tempdir().unwrap();
        let opts = BuildOpts {
            levels: 3,
            target_top: 50,
            coarsen_ratio: 2.0,
            chunk_target_splats: 256,
        };
        let _ = build(&scene, dir.path(), &opts).unwrap();
        let loaded = read_manifest(dir.path()).unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.original_splat_count, scene.len());
        for lvl in &loaded.levels {
            for chunk in &lvl.chunks {
                let p = dir.path().join(&chunk.path);
                let bytes = std::fs::read(&p).unwrap();
                assert_eq!(blake3::hash(&bytes).to_hex().to_string(), chunk.blake3);
            }
        }
    }

    #[test]
    fn unpack_level_writes_a_valid_ply() {
        let scene = make_scene(300);
        let dir = tempdir().unwrap();
        let opts = BuildOpts {
            levels: 2,
            target_top: 50,
            coarsen_ratio: 2.0,
            chunk_target_splats: 200,
        };
        let _ = build(&scene, dir.path(), &opts).unwrap();
        let out = dir.path().join("roundtrip.ply");
        let n = unpack_level_to_ply(dir.path(), 0, &out).unwrap();
        assert_eq!(n, scene.len());
        // Re-read the written PLY to confirm it parses.
        let back = read_ply(&out).unwrap();
        assert_eq!(back.len(), scene.len());
    }

    #[test]
    fn higher_importance_splats_survive_decimation() {
        // Construct two splats in the same grid cell; the higher-
        // importance one must survive a 2× decimation.
        let mut scene = SplatScene::new();
        scene.splats.push(make_splat(0.0, 0.0, 0.0, 0.2, 0.01)); // low
        scene.splats.push(make_splat(0.001, 0.0, 0.0, 0.9, 0.05)); // high
        scene.splats.push(make_splat(5.0, 5.0, 5.0, 0.5, 0.02)); // far away
        let indices: Vec<u32> = vec![0, 1, 2];
        let kept = decimate_to(&scene.splats, &indices, 2);
        assert!(kept.contains(&1), "high-importance splat must survive");
        assert!(kept.contains(&2), "spatially-isolated splat must survive");
        assert!(!kept.contains(&0), "low-importance neighbor should drop");
    }

    #[test]
    fn rejects_zero_levels() {
        let scene = make_scene(10);
        let dir = tempdir().unwrap();
        let opts = BuildOpts {
            levels: 0,
            ..Default::default()
        };
        assert!(build(&scene, dir.path(), &opts).is_err());
    }

    #[test]
    fn rejects_empty_scene() {
        let scene = SplatScene::new();
        let dir = tempdir().unwrap();
        let opts = BuildOpts::default();
        assert!(build(&scene, dir.path(), &opts).is_err());
    }
}
