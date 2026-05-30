//! Scene-global shared-codebook SH-rest palette for streaming tiles.
//!
//! ## The problem this solves
//!
//! [`crate::GlbTileCodec`] (the FP32 tile codec) leaves SH-rest as uncompressed
//! per-coefficient FLOAT accessors — 45 scalars × 4 B = **180 B/splat**, which
//! is ~83% of a balanced tile's bytes. For a 1.16M-splat scene chopped into
//! hundreds of LOD tiles that is the entire compression gap versus the
//! single-file SF GLB (which palette-codes SH-rest into a `.shpal` sidecar).
//!
//! ## The fix: one codebook, many tiles
//!
//! The single-file moat is the VQ45 SH-rest palette — a K-entry 45-D k-means
//! codebook built by [`catetus_optimize::VQPaletteShRest`], with each splat
//! storing a 16-bit index instead of 45 floats. The single-file path builds
//! the codebook over the whole scene and writes ONE `.shpal` sidecar next to
//! the GLB.
//!
//! For streaming we keep exactly that: build the codebook **once** over the
//! whole scene ([`SharedCodebook::build`]) and write it **once** as a
//! tileset-root `palette.shpal`. Each tile then stores only its splats' u16
//! palette indices (2 B/splat) — the codebook is shared, never duplicated.
//! The per-tile GLB carries quantized geometry + a `CT_gaussian_splatting_palette`
//! root extension pointing decoders at the shared root codebook; the tile's
//! own index stream rides in a tiny `.shpalx` sidecar next to the tile.
//!
//! ## Wire shapes
//!
//! * **Root** `palette.shpal` — the standard SHPA-v1 blob (zstd-19) the
//!   single-file path emits, here carrying the **whole-scene** index stream so
//!   the codebook header (`ranges` + centroids) is byte-identical to what the
//!   GLB writer / WebGPU viewer already decode via
//!   [`catetus_gltf::decode_shpal_bytes`]. Written exactly once.
//! * **Per tile** `<i>.glb` — geometry (16-bit pos + log-quant scale/opacity),
//!   SH-rest accessors elided, `CT_gaussian_splatting_palette` root extension.
//! * **Per tile** `<i>.glb.shpalx` — a tiny self-describing index-only blob:
//!   `magic "SPLX" | version u32 | n u32 | k u32 | indices u16×n` (zstd-19).
//!
//! ## Roundtrip
//!
//! [`SharedCodebook::reconstruct_sh_rest`] decodes a tile's `.shpalx` indices
//! against the shared codebook to rebuild each splat's 45-D SH-rest vector —
//! exactly what a streaming viewer does. Verified against the FP32 originals in
//! the crate tests within VQ tolerance.

use catetus_core::ir::{Color, SplatScene};
use catetus_gltf::{decode_shpal_bytes, write_glb, ShPalette, ShRestPaletteRef, WriteOpts};
use catetus_optimize::passes::PassContext;
use catetus_optimize::vq_palette::{
    take_last_sh_rest_palette, VQPaletteShRest, VQ_SH_REST_DIM,
};
use catetus_optimize::Pass;

use crate::codec::{TileBytes, TilePayloadCodec};

/// File name of the tileset-root shared codebook sidecar (next to
/// `lod-meta.json` / `tileset.json`). Written exactly once per tileset.
pub const SHARED_PALETTE_FILENAME: &str = "palette.shpal";

/// Per-tile index-sidecar extension appended to the tile file name, e.g.
/// `tiles/7.glb.shpalx`. Carries ONLY that tile's u16 palette indices.
pub const TILE_INDEX_EXT: &str = "shpalx";

const SPLX_MAGIC: u32 = 0x5350_4C58; // "SPLX"
const SPLX_VERSION: u32 = 1;

/// A scene-global VQ45 SH-rest palette: the shared codebook (decoded centroids
/// + per-coefficient ranges) plus the per-**original-splat** palette index.
///
/// Built once over the whole scene with [`SharedCodebook::build`]; the codebook
/// is serialized once via [`SharedCodebook::root_sidecar_bytes`] and the
/// per-tile codec ([`SharedPaletteTileCodec`]) looks up `indices[origin]` for
/// each tile splat.
pub struct SharedCodebook {
    /// Decoded codebook + per-scene-splat indices (the SHPA blob, parsed).
    palette: ShPalette,
    /// The exact compressed SHPA-v1 sidecar bytes (zstd-19) — written ONCE as
    /// the tileset-root `palette.shpal`. Byte-identical to the single-file
    /// path's `.shpal`, so the GLB writer / viewer decode it unchanged.
    root_sidecar: Vec<u8>,
    /// SH degree the codebook covers (3 for full SH-rest).
    sh_degree: u8,
}

impl SharedCodebook {
    /// Build the scene-global codebook by running [`VQPaletteShRest`] once over
    /// the whole scene. Mutates a CLONE of the scene internally (the pass
    /// rewrites SH-rest in place), so the caller's scene is untouched and can be
    /// tiled with its original FP32 SH-rest still present (the tile codec reads
    /// only positions/geometry from it; SH-rest comes from the shared indices).
    ///
    /// `palette_size` / `iterations` mirror [`VQPaletteShRest`]'s knobs. Returns
    /// `None` if the scene has no SH-rest splats (DC-only — no palette to share).
    pub fn build(
        scene: &SplatScene,
        palette_size: usize,
        iterations: usize,
        seed: u64,
    ) -> Result<Option<Self>, String> {
        let sh_degree = scene
            .splats
            .first()
            .map(|s| s.color.degree())
            .unwrap_or(0);
        if sh_degree == 0 {
            return Ok(None);
        }

        // Run the VQ45 k-means pass over a clone (it rewrites SH-rest in place).
        let mut work = scene.clone();
        let pass = VQPaletteShRest {
            palette_size,
            iterations,
            codebook_bits: 8,
            training_subsample: Some(200_000),
            posthac_indices: false,
        };
        let mut ctx = PassContext {
            seed,
            sh_rest_weights: None,
            splat_origin_idx: None,
        };
        pass.run(&mut work, &mut ctx)
            .map_err(|e| format!("VQPaletteShRest failed: {e}"))?;

        // Drain the parked sidecar (header + ranges + codebook + per-scene-splat
        // u16 indices, zstd-19). This IS the root `palette.shpal`.
        let side = take_last_sh_rest_palette()
            .ok_or_else(|| "VQPaletteShRest produced no sidecar".to_string())?;
        let root_sidecar = side.compressed.clone();

        // Decode it back so we have the centroids + per-scene-splat indices in
        // memory for per-tile lookup and roundtrip reconstruction.
        let palette = decode_shpal_bytes(
            &root_sidecar,
            Some((side.palette_size, side.n_splats, side.codebook_bits)),
            sh_degree,
        )
        .map_err(|e| format!("decode_shpal_bytes(root) failed: {e}"))?;

        Ok(Some(Self {
            palette,
            root_sidecar,
            sh_degree,
        }))
    }

    /// Number of centroids in the codebook (K).
    pub fn palette_size(&self) -> usize {
        self.palette.k
    }

    /// Number of splats the scene-global index stream covers.
    pub fn n_splats(&self) -> usize {
        self.palette.n
    }

    /// SH degree the codebook reconstructs.
    pub fn sh_degree(&self) -> u8 {
        self.sh_degree
    }

    /// The tileset-root `palette.shpal` bytes — write these ONCE next to
    /// `lod-meta.json`. Shared by every tile.
    pub fn root_sidecar_bytes(&self) -> &[u8] {
        &self.root_sidecar
    }

    /// The scene-global palette index for original-scene splat `origin`.
    #[inline]
    pub fn index_for_origin(&self, origin: u32) -> u16 {
        self.palette
            .indices
            .get(origin as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Reconstruct a single splat's 45-D SH-rest vector from a palette index by
    /// looking the centroid up in the shared codebook. This is exactly what a
    /// streaming viewer does (`index → centroid`).
    #[inline]
    pub fn centroid(&self, index: u16) -> &[f32] {
        let c = index as usize;
        let off = c * VQ_SH_REST_DIM;
        &self.palette.codebook[off..off + VQ_SH_REST_DIM]
    }

    /// Decode a `.shpalx` tile index blob (the per-tile index stream) and
    /// reconstruct full `Color::Sh` colors for `geom`, whose DC term is read
    /// from `geom` (the tile GLB's DC accessor) and SH-rest from the shared
    /// codebook. Returns the reconstructed colors in tile order. Used by the
    /// roundtrip test and any decoder that wants the recon without a full
    /// `read_glb` (which would also resolve the palette via the root sidecar).
    pub fn reconstruct_sh_rest(&self, splx_blob: &[u8], dc_terms: &[[f32; 3]]) -> Result<Vec<Color>, String> {
        let idx = decode_tile_indices(splx_blob)?;
        if idx.len() != dc_terms.len() {
            return Err(format!(
                "shpalx index count {} != tile splat count {}",
                idx.len(),
                dc_terms.len()
            ));
        }
        let mut out = Vec::with_capacity(idx.len());
        for (i, &pi) in idx.iter().enumerate() {
            let mut coeffs = vec![0.0f32; 3 + VQ_SH_REST_DIM];
            coeffs[0] = dc_terms[i][0];
            coeffs[1] = dc_terms[i][1];
            coeffs[2] = dc_terms[i][2];
            let cen = self.centroid(pi);
            coeffs[3..3 + VQ_SH_REST_DIM].copy_from_slice(cen);
            out.push(Color::Sh {
                degree: self.sh_degree,
                coeffs,
            });
        }
        Ok(out)
    }
}

/// Encode a tile's u16 palette index stream into a self-describing `.shpalx`
/// blob (zstd-19): `magic "SPLX" | version u32 | n u32 | k u32 | indices u16×n`.
fn encode_tile_indices(indices: &[u16], k: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(16 + indices.len() * 2);
    raw.extend_from_slice(&SPLX_MAGIC.to_le_bytes());
    raw.extend_from_slice(&SPLX_VERSION.to_le_bytes());
    raw.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    raw.extend_from_slice(&(k as u32).to_le_bytes());
    for &v in indices {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    // zstd-19 to match the root `.shpal` compression and squeeze repeated /
    // low-entropy index runs (spatially coherent tiles reuse few centroids).
    zstd::stream::encode_all(raw.as_slice(), 19).unwrap_or(raw)
}

/// Decode a `.shpalx` tile index blob back into a `Vec<u16>`.
pub fn decode_tile_indices(blob: &[u8]) -> Result<Vec<u16>, String> {
    let raw = zstd::stream::decode_all(blob).map_err(|e| format!(".shpalx zstd decode: {e}"))?;
    if raw.len() < 16 {
        return Err(format!(".shpalx too small: {} bytes", raw.len()));
    }
    let magic = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    if magic != SPLX_MAGIC {
        return Err(format!(".shpalx magic mismatch: 0x{magic:08x}"));
    }
    let version = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    if version != SPLX_VERSION {
        return Err(format!("unsupported .shpalx version: {version}"));
    }
    let n = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let _k = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]) as usize;
    if raw.len() < 16 + n * 2 {
        return Err(".shpalx truncated in indices".into());
    }
    let mut indices = Vec::with_capacity(n);
    let mut off = 16usize;
    for _ in 0..n {
        indices.push(u16::from_le_bytes([raw[off], raw[off + 1]]));
        off += 2;
    }
    Ok(indices)
}

/// Encodes each tile as a palette-elided SF GLB (16-bit pos + log-quant
/// scale/opacity, SH-rest accessors omitted) plus a tiny `.shpalx` index
/// sidecar, against a [`SharedCodebook`] written once at the tileset root.
///
/// Use [`SharedPaletteTileCodec::encode_tile`] (which takes the tile's origin
/// indices) rather than the index-less [`TilePayloadCodec::encode`] — the
/// latter cannot look up the shared codebook and falls back to the FP32 codec.
pub struct SharedPaletteTileCodec<'a> {
    codebook: &'a SharedCodebook,
}

impl<'a> SharedPaletteTileCodec<'a> {
    /// Build a tile codec bound to a scene-global codebook.
    pub fn new(codebook: &'a SharedCodebook) -> Self {
        Self { codebook }
    }

    /// WriteOpts for a balanced palette-elided tile: 16-bit positions,
    /// log-quant scale/opacity, SH-rest elided into the shared codebook.
    fn write_opts(&self, n_tile: usize) -> WriteOpts {
        let mut o = WriteOpts::default();
        o.quantize = true;
        o.log_quant_attrs = true;
        // Palette elision: the writer SKIPS all SH_DEGREE_l_COEF_n accessors and
        // emits a CT_gaussian_splatting_palette root extension pointing at the
        // shared root sidecar. n_splats here is the TILE's count (its own index
        // stream lives in the `.shpalx`); paletteSize/codebookBits/shDegree come
        // from the shared codebook.
        o.palette = Some(ShRestPaletteRef {
            // Relative URI from a tile (in `tiles/`) up to the tileset root.
            sidecar_uri: format!("../{SHARED_PALETTE_FILENAME}"),
            palette_size: self.codebook.palette_size(),
            n_splats: n_tile,
            codebook_bits: self.codebook.palette.codebook_bits,
            sh_degree: self.codebook.sh_degree(),
        });
        o
    }

    /// Encode one tile against the shared codebook. `origins[j]` is the original
    /// scene index of `scene.splats[j]` (from [`crate::TilePayload::origins`]).
    /// Returns the tile GLB bytes (SH-rest elided) plus a `.shpalx` sidecar of
    /// this tile's u16 palette indices.
    pub fn encode_tile(&self, scene: &SplatScene, origins: &[u32]) -> TileBytes {
        // Per-tile index stream: look up each tile splat's scene-global palette
        // index by its origin. O(n_tile) — no per-tile k-means.
        let indices: Vec<u16> = origins
            .iter()
            .map(|&o| self.codebook.index_for_origin(o))
            .collect();
        let splx = encode_tile_indices(&indices, self.codebook.palette_size());

        let opts = self.write_opts(scene.len());
        match write_to_bytes(scene, &opts) {
            Ok(glb) => TileBytes {
                bytes: glb,
                ext: "glb",
                sidecar: Some(splx),
                sidecar_ext: Some(TILE_INDEX_EXT),
            },
            Err(e) => {
                tracing::error!("shared-palette tile encode failed: {e}; emitting empty tile");
                let empty = write_to_bytes(&SplatScene::new(), &WriteOpts::default())
                    .unwrap_or_default();
                TileBytes::simple(empty, "glb")
            }
        }
    }
}

impl TilePayloadCodec for SharedPaletteTileCodec<'_> {
    /// Index-less fallback. A scene-global palette needs the tile's ORIGIN
    /// indices to look up the shared codebook, which `encode(&scene)` does not
    /// provide — so this path cannot palette-code and would silently lose
    /// SH-rest. To avoid a wrong-but-valid GLB, callers must use
    /// [`SharedPaletteTileCodec::encode_tile`]; this fallback writes the tile
    /// with FP32 SH-rest (no palette) so it stays correct, not tiny.
    fn encode(&self, scene: &SplatScene) -> TileBytes {
        let mut o = WriteOpts::default();
        o.quantize = true;
        o.log_quant_attrs = true;
        match write_to_bytes(scene, &o) {
            Ok(glb) => TileBytes::simple(glb, "glb"),
            Err(_) => TileBytes::simple(Vec::new(), "glb"),
        }
    }

    fn extension(&self) -> &'static str {
        "glb"
    }
}

/// Write a scene to GLB bytes via the path-based `write_glb` (the only public
/// GLB encoder) using write-to-tempfile-then-read — same trick as
/// [`crate::GlbTileCodec`].
fn write_to_bytes(scene: &SplatScene, opts: &WriteOpts) -> Result<Vec<u8>, String> {
    let tmp = tempfile::Builder::new()
        .prefix("catetus-shpal-tile-")
        .suffix(".glb")
        .tempfile()
        .map_err(|e| format!("tempfile create failed: {e}"))?;
    let path = tmp.path().to_path_buf();
    write_glb(scene, &path, opts).map_err(|e| format!("write_glb failed: {e}"))?;
    std::fs::read(&path).map_err(|e| format!("re-read tile failed: {e}"))
}

/// Helper for tests / decoders: read a splat's DC RGB term regardless of
/// whether its `Color` is `Rgb` or `Sh` (DC = coeffs[0..3]).
pub fn dc_term(s: &catetus_core::ir::Splat) -> [f32; 3] {
    match &s.color {
        Color::Rgb(c) => *c,
        Color::Sh { coeffs, .. } => [
            coeffs.first().copied().unwrap_or(0.0),
            coeffs.get(1).copied().unwrap_or(0.0),
            coeffs.get(2).copied().unwrap_or(0.0),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catetus_core::ir::{Color, Splat, SplatScene};

    /// A synthetic SH=3 scene with `n_clusters` distinct SH-rest patterns so the
    /// VQ palette has real structure to cluster (not all-identical vectors).
    fn sh3_scene(n: usize, n_clusters: usize) -> SplatScene {
        let mut s = SplatScene::new();
        for i in 0..n {
            let cluster = i % n_clusters;
            let mut coeffs = vec![0.0f32; 3 + VQ_SH_REST_DIM];
            // DC term: distinct per splat so reconstruction must use the tile DC.
            coeffs[0] = (i as f32 * 0.001).sin();
            coeffs[1] = (i as f32 * 0.002).cos();
            coeffs[2] = (i as f32 * 0.003).sin();
            // SH-rest: a per-cluster pattern.
            for d in 0..VQ_SH_REST_DIM {
                coeffs[3 + d] = ((cluster * 31 + d * 7) as f32 / 17.0).sin();
            }
            let f = i as f32;
            s.splats.push(Splat {
                position: [(f * 0.13).sin() * 50.0, (f * 0.17).cos() * 50.0, (f * 0.19).sin() * 50.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.05, 0.05, 0.05],
                opacity: 0.8,
                color: Color::Sh { degree: 3, coeffs },
            });
        }
        s
    }

    #[test]
    fn shared_codebook_builds_and_root_sidecar_decodes() {
        let scene = sh3_scene(2000, 8);
        let cb = SharedCodebook::build(&scene, 16, 8, 0xABCD)
            .expect("build ok")
            .expect("scene has SH-rest");
        assert_eq!(cb.palette_size(), 16);
        assert_eq!(cb.n_splats(), 2000);
        assert_eq!(cb.sh_degree(), 3);
        assert!(!cb.root_sidecar_bytes().is_empty());
        // Every scene splat must have a valid index into the codebook.
        for i in 0..scene.len() {
            assert!((cb.index_for_origin(i as u32) as usize) < cb.palette_size());
        }
    }

    #[test]
    fn dc_only_scene_has_no_shared_codebook() {
        let mut s = SplatScene::new();
        for _ in 0..10 {
            s.splats.push(Splat {
                position: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
                opacity: 1.0,
                color: Color::Rgb([0.5, 0.4, 0.3]),
            });
        }
        assert!(SharedCodebook::build(&s, 16, 4, 1).unwrap().is_none());
    }

    #[test]
    fn tile_index_blob_roundtrips() {
        let idx = vec![0u16, 5, 5, 12, 3, 0, 65535, 7];
        let blob = encode_tile_indices(&idx, 65536);
        let back = decode_tile_indices(&blob).unwrap();
        assert_eq!(idx, back);
    }

    /// The core acceptance test: a shared-palette tile + the shared codebook
    /// reconstruct SH-rest within VQ tolerance of the FP32 originals.
    #[test]
    fn shared_palette_tile_roundtrips_sh_rest_within_vq_tolerance() {
        let scene = sh3_scene(1500, 8);
        let cb = SharedCodebook::build(&scene, 64, 10, 0x1234)
            .unwrap()
            .unwrap();
        let codec = SharedPaletteTileCodec::new(&cb);

        // A "tile" = a subset of the scene (e.g. an octree leaf). Pick splats
        // [100, 400) as the tile; origins are their global indices.
        let lo = 100usize;
        let hi = 400usize;
        let mut tile = SplatScene::new();
        let mut origins = Vec::new();
        for i in lo..hi {
            tile.splats.push(scene.splats[i].clone());
            origins.push(i as u32);
        }

        let tb = codec.encode_tile(&tile, &origins);
        assert_eq!(tb.ext, "glb");
        let splx = tb.sidecar.as_ref().expect("tile carries a .shpalx index sidecar");
        assert_eq!(tb.sidecar_ext, Some(TILE_INDEX_EXT));

        // Decode the tile GLB (no palette resolution — DC only) to get DC terms.
        let decoded = catetus_gltf::read_glb_bytes(&tb.bytes).expect("tile GLB decodes");
        assert_eq!(decoded.len(), tile.len(), "tile splat count round-trips");
        let dc_terms: Vec<[f32; 3]> = decoded.splats.iter().map(dc_term).collect();

        // Reconstruct SH-rest from (shared codebook, tile indices).
        let recon = cb
            .reconstruct_sh_rest(splx, &dc_terms)
            .expect("reconstruct from shared codebook");
        assert_eq!(recon.len(), tile.len());

        // Compare reconstructed SH-rest to the centroid the SCENE-GLOBAL pass
        // assigned each origin splat (VQ is lossy vs the raw FP32 input, but
        // the tile recon must EXACTLY match the shared codebook's centroid for
        // that splat's global index — that's the contract).
        let mut max_abs_err = 0.0f32;
        for (j, &origin) in origins.iter().enumerate() {
            let expected_centroid = cb.centroid(cb.index_for_origin(origin));
            if let Color::Sh { coeffs, .. } = &recon[j] {
                for d in 0..VQ_SH_REST_DIM {
                    let e = (coeffs[3 + d] - expected_centroid[d]).abs();
                    if e > max_abs_err {
                        max_abs_err = e;
                    }
                }
            } else {
                panic!("expected SH color in reconstruction");
            }
        }
        assert!(
            max_abs_err < 1e-5,
            "tile recon must equal shared centroid exactly; max_abs_err={max_abs_err}"
        );

        // And the recon must be a reasonable VQ approximation of the ORIGINAL
        // FP32 SH-rest (sanity: K=64 over 8 true clusters → near-exact here).
        let mut sse = 0.0f64;
        for (j, &origin) in origins.iter().enumerate() {
            if let (Color::Sh { coeffs: rc, .. }, Color::Sh { coeffs: oc, .. }) =
                (&recon[j], &scene.splats[origin as usize].color)
            {
                for d in 0..VQ_SH_REST_DIM {
                    let diff = (rc[3 + d] - oc[3 + d]) as f64;
                    sse += diff * diff;
                }
            }
        }
        let mse = sse / (origins.len() * VQ_SH_REST_DIM) as f64;
        assert!(mse < 0.05, "VQ reconstruction MSE too high: {mse}");
    }

    #[test]
    fn shared_palette_tile_is_far_smaller_than_fp32_tile() {
        let scene = sh3_scene(3000, 16);
        let cb = SharedCodebook::build(&scene, 64, 6, 0x55).unwrap().unwrap();
        let codec = SharedPaletteTileCodec::new(&cb);

        let mut tile = SplatScene::new();
        let mut origins = Vec::new();
        for i in 0..1000 {
            tile.splats.push(scene.splats[i].clone());
            origins.push(i as u32);
        }

        // FP32 tile (current GlbTileCodec, Balanced).
        let fp32 = crate::GlbTileCodec::new(crate::TilePreset::Balanced).encode(&tile);
        let fp32_total = fp32.bytes.len() + fp32.sidecar.as_ref().map(|v| v.len()).unwrap_or(0);

        // Shared-palette tile: GLB (no SH-rest) + tiny index sidecar. Excludes
        // the shared root codebook (amortized once across the whole tileset).
        let shp = codec.encode_tile(&tile, &origins);
        let shp_total = shp.bytes.len() + shp.sidecar.as_ref().map(|v| v.len()).unwrap_or(0);

        assert!(
            shp_total * 3 < fp32_total,
            "shared-palette tile ({shp_total} B) should be <1/3 of FP32 tile ({fp32_total} B)"
        );
    }
}
