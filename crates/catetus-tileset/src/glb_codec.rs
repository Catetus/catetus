//! Real GLB tile payload codec.
//!
//! Each octree tile is just a small splat scene, so the most direct way to make
//! tiles viewer-loadable is to encode each one as a standalone **SF GLB**
//! (glTF 2.0 + `KHR_gaussian_splatting`) using the same GLB writer the
//! single-file `catetus optimize` path uses ([`catetus_gltf::write_glb`]). A GLB
//! tile loads in the Catetus WebGPU viewer, Cesium-style 3D-Tiles tooling, and
//! anything else that reads `KHR_gaussian_splatting` — no bespoke tile format.
//!
//! ## Why write-to-tempfile-then-read
//!
//! `catetus-gltf`'s public GLB writer is path-based (`write_glb(scene, &Path,
//! &WriteOpts)`); there is no public bytes-returning GLB encoder. So the codec
//! writes each tile to a `tempfile::NamedTempFile`, reads the bytes back, and
//! returns them. This keeps the tile bytes byte-identical to what a user would
//! get from a single-file `catetus optimize --target glb` of that sub-scene at
//! the same `WriteOpts`.
//!
//! ## Preset / quantization
//!
//! The full `catetus optimize` preset pipeline (Morton sort, opacity prune, VQ
//! palette, …) mutates the *scene*; running it per tile would be both expensive
//! and would change splat counts (breaking the manifest's per-tile count
//! contract). Instead the tile codec applies the cheap, count-preserving
//! `WriteOpts`-level knobs: [`TilePreset::Balanced`] uses 16-bit position
//! quantization (`KHR_mesh_quantization`) + log-quant scale/opacity (the same
//! writer-side losses the web presets use), [`TilePreset::QualityMax`] writes
//! full-precision FLOAT accessors. SH-rest is emitted as per-coefficient FLOAT
//! accessors (valid `KHR_gaussian_splatting`); palette/`.shpal` sidecar coding
//! is a future enhancement (it requires running the `VQPaletteShRest` optimize
//! pass per tile — see STATUS roadmap).

use catetus_core::ir::SplatScene;
use catetus_gltf::{write_glb, WriteOpts};

use crate::codec::{TileBytes, TilePayloadCodec};

/// Per-tile encoding preset. A deliberately small surface that maps onto the
/// count-preserving `WriteOpts` knobs (the full optimize-pipeline presets change
/// splat counts, which a tileset's per-tile manifest counts forbid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilePreset {
    /// 16-bit quantized positions + log-quant scale/opacity. Smaller tiles,
    /// the default web-streaming profile.
    Balanced,
    /// Full FLOAT accessors, no quantization. Largest tiles, lossless geometry.
    QualityMax,
}

impl TilePreset {
    /// Parse a CLI `--preset` string into a tile preset. Unknown / size-* /
    /// web-* presets all map to [`TilePreset::Balanced`] (quantized streaming);
    /// `quality-max` / `lossless` map to [`TilePreset::QualityMax`].
    pub fn from_cli(name: &str) -> Self {
        match name {
            "quality-max" | "lossless" | "quality" => TilePreset::QualityMax,
            _ => TilePreset::Balanced,
        }
    }

    fn write_opts(self) -> WriteOpts {
        let mut o = WriteOpts::default();
        match self {
            TilePreset::Balanced => {
                o.quantize = true; // 16-bit positions (KHR_mesh_quantization)
                o.log_quant_attrs = true; // log-space scale, logit-space opacity
            }
            TilePreset::QualityMax => {
                o.quantize = false; // full FLOAT geometry
            }
        }
        o
    }
}

/// Encodes each tile as a standalone SF GLB via [`catetus_gltf::write_glb`].
pub struct GlbTileCodec {
    preset: TilePreset,
}

impl GlbTileCodec {
    /// Build a GLB tile codec for the given preset.
    pub fn new(preset: TilePreset) -> Self {
        Self { preset }
    }

    /// Convenience: build from a CLI `--preset` string.
    pub fn from_cli_preset(name: &str) -> Self {
        Self::new(TilePreset::from_cli(name))
    }

    /// The preset this codec encodes tiles with.
    pub fn preset(&self) -> TilePreset {
        self.preset
    }

    /// Encode one tile scene to GLB bytes via write-to-tempfile-then-read.
    /// Returns the GLB bytes on success.
    fn encode_to_bytes(&self, scene: &SplatScene) -> Result<Vec<u8>, String> {
        let opts = self.preset.write_opts();
        // Unique temp path; `write_glb` writes the JSON+BIN GLB container here.
        let tmp = tempfile::Builder::new()
            .prefix("catetus-tile-")
            .suffix(".glb")
            .tempfile()
            .map_err(|e| format!("tempfile create failed: {e}"))?;
        let path = tmp.path().to_path_buf();
        write_glb(scene, &path, &opts).map_err(|e| format!("write_glb failed: {e}"))?;
        let bytes = std::fs::read(&path).map_err(|e| format!("re-read tile failed: {e}"))?;
        Ok(bytes)
    }
}

impl TilePayloadCodec for GlbTileCodec {
    fn encode(&self, scene: &SplatScene) -> TileBytes {
        match self.encode_to_bytes(scene) {
            Ok(bytes) => TileBytes::simple(bytes, "glb"),
            // Encoding a tile should not fail for well-formed IR. If it does we
            // surface it as an empty-but-valid GLB so the tileset still writes
            // and the failure is visible as a zero-splat tile rather than a
            // panic mid-build.
            Err(e) => {
                tracing::error!("GLB tile encode failed: {e}; emitting empty tile");
                let empty = self.encode_to_bytes(&SplatScene::new()).unwrap_or_default();
                TileBytes::simple(empty, "glb")
            }
        }
    }

    fn extension(&self) -> &'static str {
        "glb"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catetus_core::ir::{Color, Splat, SplatScene};
    use catetus_gltf::read_glb_bytes;

    fn dc_scene(n: usize) -> SplatScene {
        let mut s = SplatScene::new();
        for i in 0..n {
            let f = i as f32;
            s.splats.push(Splat {
                position: [f * 0.1, (f * 0.2).cos(), (f * 0.3).sin()],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.05, 0.05, 0.05],
                opacity: 0.7,
                color: Color::Rgb([0.4, 0.5, 0.6]),
            });
        }
        s
    }

    #[test]
    fn glb_tile_is_a_valid_decodable_glb() {
        let s = dc_scene(64);
        let codec = GlbTileCodec::new(TilePreset::Balanced);
        let tb = codec.encode(&s);
        assert_eq!(tb.ext, "glb");
        assert_eq!(&tb.bytes[0..4], b"glTF", "GLB container magic");
        // The cardinal test: the emitted bytes are a real GLB the reader loads.
        let decoded = read_glb_bytes(&tb.bytes).expect("tile must be a valid GLB");
        assert_eq!(decoded.len(), s.len(), "tile splat count must round-trip");
    }

    #[test]
    fn quality_max_round_trips_geometry_losslessly() {
        let s = dc_scene(32);
        let tb = GlbTileCodec::new(TilePreset::QualityMax).encode(&s);
        let decoded = read_glb_bytes(&tb.bytes).expect("valid GLB");
        assert_eq!(decoded.len(), s.len());
        // FLOAT accessors -> positions exact.
        for (a, b) in decoded.splats.iter().zip(s.splats.iter()) {
            for k in 0..3 {
                assert!((a.position[k] - b.position[k]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn sh_scene_tile_is_valid_glb() {
        let mut s = SplatScene::new();
        for i in 0..50 {
            let f = i as f32;
            let coeffs: Vec<f32> = (0..48).map(|k| ((k as f32 + f) * 0.01).sin()).collect();
            s.splats.push(Splat {
                position: [f * 0.1, (f * 0.2).cos(), (f * 0.3).sin()],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.02, 0.02, 0.02],
                opacity: 0.8,
                color: Color::Sh { degree: 3, coeffs },
            });
        }
        let tb = GlbTileCodec::new(TilePreset::Balanced).encode(&s);
        let decoded = read_glb_bytes(&tb.bytes).expect("SH tile is a valid GLB");
        assert_eq!(decoded.len(), s.len());
    }

    #[test]
    fn empty_tile_encodes_to_valid_glb() {
        let tb = GlbTileCodec::new(TilePreset::Balanced).encode(&SplatScene::new());
        let decoded = read_glb_bytes(&tb.bytes).expect("empty tile must still be valid");
        assert_eq!(decoded.len(), 0);
    }
}
