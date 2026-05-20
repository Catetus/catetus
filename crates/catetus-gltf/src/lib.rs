#![deny(clippy::all)]
//! glTF 2.0 + `KHR_gaussian_splatting` writer/reader, with the optional
//! `CT_spatial_streaming_index` vendor extension defined in SPEC-0007.
//!
//! We hand-roll the JSON to stay in control of the wire format (the
//! `gltf` crate doesn't know about KHR_gaussian_splatting yet).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, WriteBytesExt};
use catetus_core::{Color, CoordinateSystem, Splat, SplatScene, TemporalMode};
use catetus_spz::{encode_spz, read_spz_bytes};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// V5.2 / V5.1-F joint-tail residual sidecar codec. See module docs.
/// Encoder + decoder + format constants for the `CT_v5_tail_residual`
/// extension. The encoder is used by `catetus-optimize` (re-exported
/// at `catetus_optimize::v5_tail`) and the decoder is wired into
/// [`read_glb`] / [`read_gltf`] below.
pub mod v5_tail;

/// Variants of the `KHR_gaussian_splatting_compression_spz` extension that the
/// writer can emit. The wire-version integer flows straight into the
/// extension's `version` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpzVariant {
    /// SPZ v2 — the current wire format produced by `catetus-spz`.
    V2,
}

impl SpzVariant {
    /// Integer carried in the extension's `version` field.
    pub fn version(self) -> u32 {
        match self {
            SpzVariant::V2 => 2,
        }
    }
}

/// Lossless wrapper applied to the GLB BIN chunk after the per-attribute
/// layout has been packed. Independent of [`SpzVariant`] (which is a
/// content-aware codec); this is a pure byte-stream compressor.
///
/// The viewer side detects the chosen wrapper via the root-level
/// `CT_brotli_buffer` vendor extension and decompresses the BIN chunk
/// in-place before reading bufferView offsets — so accessor offsets remain
/// expressed in *uncompressed* coordinates and the existing accessor /
/// bufferView shape is preserved bit-for-bit.
///
/// Measured on real `catetus optimize` outputs (see
/// `experiments/lossless-zstd/RESULT.md`): brotli-11 saves ~47% on the
/// `quality-max` preset's FP32 SH coefficients and ~5-7% on the quantized
/// presets vs the current zstd-19 sidecar baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LosslessWrap {
    /// Brotli quality 11 (max ratio, slow encode). Decoder works in a single
    /// pass on the full BIN payload; a 100 MB scene decompresses in a few
    /// hundred ms on a modern desktop CPU.
    Brotli11,
    /// Byte-plane-transposed zstd-19. Each bufferView in the BIN chunk is
    /// transposed from `[splat][byte_within_stride]` layout to
    /// `[byte_within_stride][splat]` layout before compression. Adjacent
    /// splats in Morton order share most high-bytes of quantized integer
    /// attributes, so the transposed stream is highly run-length-friendly
    /// — zstd-19 picks up the ~50% extra saving that WebP-lossless gets on
    /// the SOG image-domain layout. Wrapper extension: `CT_zstd_split_buffer`.
    Zstd19Split,
}

/// Alias to keep both naming conventions working.
pub type WriteOptions = WriteOpts;

/// Which revision of `KHR_gaussian_splatting` the writer should target.
///
/// The Khronos Release Candidate landed on 2026-04-15 (commit
/// `63770cc7`) and reshaped the on-wire schema in three breaking ways:
///
/// 1. Attribute keys are namespaced (`KHR_gaussian_splatting:ROTATION`
///    instead of the pre-RC `_ROTATION`), and `attributes` moved from
///    inside the extension object to the primitive level next to
///    `mode` / `indices`.
/// 2. The primitive now MUST declare `mode = 0` (POINTS) and the
///    extension blob MUST carry string `kernel` + `colorSpace`.
/// 3. Spherical-harmonic coefficients are emitted as one VEC3 FLOAT
///    accessor per coefficient (`SH_DEGREE_l_COEF_n`) instead of a
///    single SCALAR-of-45 buffer; the DC color lives at
///    `SH_DEGREE_0_COEF_0`.
///
/// `Pre2026` keeps the historical `_ROTATION` / `_COLOR_DC` layout for
/// backwards-compatibility round-tripping; new output should use the
/// `RcMay2026` default so it passes `catetus-khr-conformance` and the
/// upstream Khronos validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpecVersion {
    /// Khronos RC text at commit `63770cc7` (2026-04-15) — the default.
    #[default]
    RcMay2026,
    /// Pre-RC layout used by Catetus v0.x and the legacy web viewer.
    Pre2026,
}

/// glTF I/O errors.
#[derive(Debug, Error)]
pub enum GltfError {
    /// Underlying IO failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON could not be parsed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// The required `KHR_gaussian_splatting` extension was absent.
    #[error("KHR_gaussian_splatting extension missing")]
    MissingExtension,
    /// One of the chunks failed its checksum.
    #[error("checksum mismatch on chunk {0}")]
    ChecksumMismatch(usize),
    /// An external buffer file could not be loaded.
    #[error("buffer not found: {0}")]
    BufferNotFound(String),
    /// The GLB declared `CT_gaussian_splatting_palette` (SH-rest lives in a
    /// `.shpal` sidecar) but the sidecar could not be located next to the
    /// GLB. Without it the decoder would silently emit all-zero SH-rest
    /// coefficients — a corruption that has, in the past, polluted PSNR
    /// benches by ~9 dB without any user-visible warning. Callers that
    /// genuinely want DC-only degradation can opt in via
    /// `ReadOpts { allow_missing_palette: true }` or by setting the
    /// `CATETUS_ALLOW_MISSING_PALETTE=1` environment variable.
    #[error(
        "missing SH-rest palette sidecar: GLB declares CT_gaussian_splatting_palette \
         pointing at `{uri}` but the file was not found at `{tried}`. The decoder \
         would otherwise silently emit all-zero SH-rest coefficients. Re-encode the \
         GLB (this regenerates the `.shpal`), restore the sidecar from a backup, or \
         set `CATETUS_ALLOW_MISSING_PALETTE=1` to accept DC-only degradation."
    )]
    MissingPaletteSidecar {
        /// The `uri` value the GLB advertised inside its
        /// `CT_gaussian_splatting_palette` root extension.
        uri: String,
        /// Absolute path the loader attempted to read.
        tried: String,
    },
    /// The GLB declared `CT_v5_tail_residual` (V5.2 joint-tail residual
    /// sidecar) but the `.v5tail` file could not be located. Mirrors the
    /// `MissingPaletteSidecar` story — silently rendering without the
    /// residual gives the user a baseline VQ45 reconstruction that's
    /// numerically valid but ~10 dB worse than what they paid the encode
    /// time for. Callers that want to accept the degradation opt in via
    /// `ReadOpts { allow_missing_tail: true }` or
    /// `CATETUS_ALLOW_MISSING_TAIL=1`. Hard-fails iff the extension is
    /// listed in `extensionsRequired`; merely listed in `extensionsUsed`
    /// + sidecar missing produces a warning and falls back to baseline.
    #[error(
        "missing V5.2 tail-residual sidecar: GLB declares CT_v5_tail_residual \
         pointing at `{uri}` but the file was not found at `{tried}`. The decoder \
         would otherwise render the baseline-VQ45 reconstruction (typically \
         ~10 dB worse than the residual-applied output). Re-encode the GLB \
         (this regenerates the `.v5tail`), restore the sidecar from a backup, or \
         set `CATETUS_ALLOW_MISSING_TAIL=1` to accept the degradation."
    )]
    MissingTailSidecar {
        /// The `uri` value the GLB advertised inside its
        /// `CT_v5_tail_residual` root extension.
        uri: String,
        /// Absolute path the loader attempted to read.
        tried: String,
    },
    /// Asset declared an unsupported extension version.
    #[error("unsupported Gaussian Splatting extension version: {0}")]
    UnsupportedVersion(String),
    /// Generic shape error.
    #[error("malformed glTF: {0}")]
    Malformed(String),
    /// Chunked output is not supported for GLB containers.
    #[error("glb_chunked_unsupported: GLB cannot embed multiple external chunks")]
    GlbChunkedUnsupported,
    /// Failure decoding or encoding a `KHR_gaussian_splatting_compression_spz` blob.
    #[error("spz codec error: {0}")]
    Spz(String),
    /// Failure compressing/decompressing the `CT_brotli_buffer` BIN wrapper.
    #[error("brotli codec error: {0}")]
    Brotli(String),
}

/// Options that control glTF export.
#[derive(Debug, Clone)]
pub struct WriteOpts {
    /// Whether to split splats into multiple chunked external buffers.
    pub chunked: bool,
    /// Target splat count per chunk when `chunked` is true.
    pub chunk_target_splats: usize,
    /// Optional LOD splat-fraction levels.
    pub lod_fractions: Vec<f32>,
    /// Enable `KHR_mesh_quantization` integer accessors for the small,
    /// quantization-friendly attributes (POSITION → u16, SCALE/OPACITY → u8,
    /// and — under `SpecVersion::Pre2026` only — the legacy `_COLOR_DC` u8
    /// path). See SPEC-0013 for the wire format and rationale.
    ///
    /// Defaults to `false` so the lossless / quality-max paths stay byte-stable.
    /// The `catetus optimize` CLI flips this on for the web-targeted presets.
    pub quantize: bool,
    /// When `quantize` is true and this is true, ROTATION is emitted as a
    /// normalized signed SHORT (`5122 / VEC4 / normalized`) per the RC
    /// quaternion-quantization table. Validators accept normalized signed
    /// BYTE or SHORT; SHORT is the safer default. Only takes effect when
    /// `quantize` is also set; no-op under `SpecVersion::Pre2026` for
    /// byte-stability reasons.
    pub quantize_rotation: bool,
    /// Target revision of `KHR_gaussian_splatting`. See [`SpecVersion`].
    pub spec_version: SpecVersion,
    /// When set, route the splat payload through `catetus-spz` and emit
    /// the `KHR_gaussian_splatting_compression_spz` extension on the output.
    /// Only meaningful for GLB output.
    pub compress: Option<SpzVariant>,
    /// When set, brotli-compress the GLB BIN chunk after packing and signal
    /// the wrapper via the `CT_brotli_buffer` root extension. Lossless and
    /// independent of `compress`; mutually exclusive with `Some(SpzVariant)`
    /// because the SPZ blob is already content-compressed (double-compressing
    /// loses ~0%). Only takes effect on `write_glb`.
    pub lossless: Option<LosslessWrap>,
    /// SH-rest per-channel quantization side table. When `Some`, each
    /// `KHR_gaussian_splatting:SH_DEGREE_l_COEF_n` accessor (degrees 1..=3,
    /// 15 coefficients × 3 channels = 45 scalars) is emitted as a normalized
    /// signed integer accessor — BYTE when `bits <= 8`, SHORT when `bits > 8`
    /// — with `min/max = [-range_ch, range_ch]`. When `None`, the legacy
    /// FP32 path is used. Cuts SH-rest from 180 b/s (fp32 × 45) to 45 b/s
    /// (b=8) or 30 b/s (b=6, packed as BYTE).
    ///
    /// Round-trip lossy: the upstream `QuantizeSHRest` pass already
    /// dequantized the values into the scene, so the writer just re-applies
    /// the same forward map and the round trip is bit-exact.
    pub sh_rest_quant: Option<ShRestQuantTable>,
    /// SOG-style smallest-3 quaternion side table. When `Some`, ROTATION is
    /// emitted as a SCALAR UNSIGNED_INT accessor (componentType 5125,
    /// 4 B/splat) holding `u32 = [q0:bits | q1:bits | q2:bits | tag:2]`.
    /// See `experiments/SOG_STUDY_RUN/SMALLEST3_QUAT_RESULT.md`.
    pub rotation_smallest3: Option<RotationSmallest3Table>,
    /// Per-component rotation quant (8-bit). Mutually exclusive with smallest3.
    pub rotation_quant: Option<RotationQuantTable>,
    /// Per-channel DC quant. Emits VEC3 UBYTE/USHORT normalized.
    pub dc_quant: Option<DcQuantTable>,
    /// When set, the writer SKIPS every `SH_DEGREE_l_COEF_n` accessor for
    /// degrees 1..=3 (45 scalars per splat) and instead emits an
    /// `CT_gaussian_splatting_palette` root extension that points decoders at
    /// the companion `.shpal` sidecar holding the 16-bit palette index per
    /// splat plus the K-entry 45-D centroid codebook. The actual sidecar file
    /// must be written separately by the caller (see
    /// `catetus-cli`'s `cmd_optimize`). This is how the VQ45 preset
    /// achieves SOG-class SH-rest sizes (~3-4 MB total for 1M splats sh=3
    /// vs. the 50-60 MB the FP32 SH-rest accessors would consume even after
    /// zstd). When `None` (default), the legacy SH-rest accessor path is used.
    pub palette: Option<ShRestPaletteRef>,
    /// When set, the writer emits an `CT_v5_tail_residual` root extension
    /// pointing at the companion `.glb.v5tail` sidecar (written separately
    /// by the caller). The sidecar carries the V5.2 joint-tail residual
    /// codec payload — see [`v5_tail`]. The extension is advertised in
    /// `extensionsUsed` (NOT `extensionsRequired`) by default so legacy
    /// decoders ignore it and render the baseline VQ45 reconstruction;
    /// set `required = true` on the ref to force-fail those clients.
    pub v5_tail: Option<V5TailRef>,
    /// When `true` and `quantize` is also `true`, the writer quantizes
    /// SCALE in **log space** and OPACITY in **logit space** rather than
    /// linear / sigmoid space. Accessor `min`/`max` metadata is set in the
    /// transformed space and the [`CT_LOG_QUANT_ATTRS`] root extension is
    /// emitted so decoders know to apply `exp(scale)` and `sigmoid(opacity)`
    /// after dequantization.
    ///
    /// This restores SOG-parity visual quality on heavy-tailed scale
    /// distributions — uniform-in-linear 8-bit quantization crushes the
    /// long tail of small splats; uniform-in-log preserves it. Same byte
    /// budget. Off by default for backwards compatibility with the
    /// existing wmv-sh3-q* baseline numbers.
    pub log_quant_attrs: bool,
}

/// Pointer + metadata for an externally-written `.shpal` sidecar. When
/// `WriteOpts::palette` is `Some`, the writer omits SH-rest accessors and
/// embeds this metadata as an `CT_gaussian_splatting_palette` root extension
/// so decoders (e.g. `experiments/w3-fidelity-harness/code/cpu-fidelity.mjs`)
/// can locate and parse the sidecar to reconstruct SH-rest from indices.
#[derive(Debug, Clone)]
pub struct ShRestPaletteRef {
    /// Relative URI to the sidecar file, resolved against the GLB's parent
    /// directory. Typically `<glb-basename>.shpal`.
    pub sidecar_uri: String,
    /// Number of centroids in the codebook (e.g. 65,536).
    pub palette_size: usize,
    /// Number of splats covered by the palette indices.
    pub n_splats: usize,
    /// Codebook quantization width in bits per coefficient column (typically 8).
    pub codebook_bits: u8,
    /// SH degree at which the palette covers all `SH_DEGREE_l_COEF_n` slots.
    /// Decoders use this to know how many coefficients to reconstruct per
    /// splat from the 45-D centroid vector.
    pub sh_degree: u8,
}

/// Pointer + metadata for an externally-written `.glb.v5tail` sidecar that
/// carries V5.2 joint-tail residuals (the per-Morton-cell affine codec from
/// the V5.1-F prototype). When `WriteOpts::v5_tail` is `Some`, the writer
/// emits an `CT_v5_tail_residual` root extension; the GLB itself is
/// unchanged otherwise (it still carries the baseline VQ45 attribute
/// accessors). At read time, [`read_glb`] decodes the sidecar and adds the
/// residuals to the selected splat indices.
#[derive(Debug, Clone)]
pub struct V5TailRef {
    /// Relative URI to the sidecar file, resolved against the GLB's parent
    /// directory. Typically `<glb-basename>.glb.v5tail`.
    pub sidecar_uri: String,
    /// Total splat count covered by the sidecar's selection bitmap (must
    /// equal the GLB's `splatCount`). Cross-checked at decode time.
    pub n_splats: usize,
    /// Number of splats actually carrying residuals (the top-K selection).
    pub k_selected: usize,
    /// SH-rest coefficient count covered by the residual stream — 15 for
    /// sh_degree=3, 8 for degree=2, 3 for degree=1, 0 otherwise.
    pub sh_rest_coefs: u8,
    /// Number of Morton cells the residuals are partitioned into.
    pub n_cells: u16,
    /// When `true`, the writer adds `CT_v5_tail_residual` to
    /// `extensionsRequired` so decoders that don't understand the extension
    /// hard-fail rather than silently rendering the baseline. Set this only
    /// when the residual is part of the asset's compliance contract.
    pub required: bool,
}

/// Per-channel SH-rest quantization side table. 45 entries: 3 (degree-1) + 5
/// (degree-2) + 7 (degree-3) coefficients × 3 channels, packed in the same
/// order as `Color::Sh::coeffs[3..]`. See `WriteOpts::sh_rest_quant`.
#[derive(Debug, Clone)]
pub struct RotationQuantTable {
    /// Bits per component. 2..=8 ⇒ UBYTE, 9..=16 ⇒ USHORT.
    pub bits: u8,
    pub mins: [f32; 4],
    pub maxs: [f32; 4],
}

/// Per-channel DC quantization side table.
#[derive(Debug, Clone)]
pub struct DcQuantTable {
    pub bits: u8,
    pub mins: [f32; 3],
    pub maxs: [f32; 3],
}

/// Per-channel SH-rest quantization side table.
#[derive(Debug, Clone)]
pub struct ShRestQuantTable {
    /// Signed-int width. 2..=8 ⇒ BYTE accessor, 9..=16 ⇒ SHORT.
    pub bits: u8,
    /// Per-channel range; the reconstructed value is `clamp(q / 2^(bits-1)-1,
    /// -1, 1) * range_ch`. Length must be `45` (writer pads / truncates).
    pub ranges: Vec<f32>,
}

/// SOG-style smallest-3 quaternion side table. See `WriteOpts::rotation_smallest3`.
#[derive(Debug, Clone)]
pub struct RotationSmallest3Table {
    /// Bits per stored component (3 stored, 1 dropped + 2-bit tag).
    /// Valid range is 6..=10. At 10 bits the layout fits in 32 bits exactly.
    pub component_bits: u8,
}

impl Default for WriteOpts {
    fn default() -> Self {
        Self {
            chunked: false,
            chunk_target_splats: 100_000,
            lod_fractions: vec![1.0],
            quantize: false,
            quantize_rotation: false,
            spec_version: SpecVersion::default(),
            compress: None,
            lossless: None,
            sh_rest_quant: None,
            rotation_smallest3: None,
            rotation_quant: None,
            dc_quant: None,
            palette: None,
            v5_tail: None,
            log_quant_attrs: false,
        }
    }
}

/// Result of `inspect_gltf`.
#[derive(Debug, Clone)]
pub struct InspectReport {
    /// Whether `KHR_gaussian_splatting` is declared.
    pub has_khr: bool,
    /// Whether the `CT_spatial_streaming_index` extension is present.
    pub has_spatial_index: bool,
    /// Number of chunk entries, when the streaming-index extension is present.
    pub chunk_count: usize,
    /// Per-chunk checksum validation outcome.
    pub checksum_ok: bool,
    /// Splat count reported by the asset.
    pub splat_count: usize,
}

// ---------- glTF JSON shape (minimal subset) ----------

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GltfAsset {
    version: String,
    generator: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GltfBuffer {
    #[serde(rename = "byteLength")]
    byte_length: usize,
    uri: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GltfBufferView {
    buffer: usize,
    #[serde(rename = "byteOffset", default)]
    byte_offset: usize,
    #[serde(rename = "byteLength")]
    byte_length: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GltfAccessor {
    #[serde(rename = "bufferView")]
    buffer_view: usize,
    #[serde(rename = "componentType")]
    component_type: u32, // 5126 = FLOAT, 5121 = UBYTE, 5123 = USHORT
    count: usize,
    #[serde(rename = "type")]
    accessor_type: String,
    /// Whether the integer accessor should be interpreted as a normalized
    /// floating-point value in `[0, 1]` (signed integers map to `[-1, 1]`).
    /// Required for `KHR_mesh_quantization` integer accessors.
    #[serde(default, skip_serializing_if = "is_false")]
    normalized: bool,
    /// Per-component minima. Required on POSITION per glTF 2.0 §3.6.2.4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min: Option<Vec<f32>>,
    /// Per-component maxima.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max: Option<Vec<f32>>,
}

#[inline]
fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GltfRoot {
    asset: GltfAsset,
    #[serde(
        rename = "extensionsUsed",
        skip_serializing_if = "Vec::is_empty",
        default
    )]
    extensions_used: Vec<String>,
    #[serde(
        rename = "extensionsRequired",
        skip_serializing_if = "Vec::is_empty",
        default
    )]
    extensions_required: Vec<String>,
    #[serde(default)]
    buffers: Vec<GltfBuffer>,
    #[serde(rename = "bufferViews", default)]
    buffer_views: Vec<GltfBufferView>,
    #[serde(default)]
    accessors: Vec<GltfAccessor>,
    #[serde(default)]
    meshes: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    extensions: serde_json::Map<String, serde_json::Value>,
}

const KHR: &str = "KHR_gaussian_splatting";
const KHR_SPZ: &str = "KHR_gaussian_splatting_compression_spz";
const CT_INDEX: &str = "CT_spatial_streaming_index";
const KHR_QUANT: &str = "KHR_mesh_quantization";
/// Vendor extension carrying the brotli wrapper metadata for the GLB BIN
/// chunk. Lives at the root of the glTF document (not on a primitive) so a
/// single ext flag covers every bufferView in buffer 0. See
/// `apply_brotli_wrap` / `unwrap_brotli` for the wire format.
const CT_BROTLI: &str = "CT_brotli_buffer";
/// Vendor extension carrying the byte-plane-transposed zstd-19 wrapper.
/// Stored at the root level (single ext flag covers buffer 0). Manifest is
/// `{ buffer, uncompressedByteLength, compressedByteLength, views: [{offset,
/// length, stride}...] }`. The encoder transposes each listed view in-place
/// from interleaved layout (`stride` bytes per splat) to byte-plane layout
/// (one contiguous run per byte-within-stride) before zstd-19, and the
/// decoder reverses the transpose right after zstd decompression so all
/// downstream accessor reads see the original byte coordinates.
const CT_ZSTD_SPLIT: &str = "CT_zstd_split_buffer";
/// Vendor extension carrying a pointer to a `.shpal` sidecar containing a
/// 45-D k-means palette codebook + per-splat 16-bit palette indices that
/// reconstruct `SH_DEGREE_{l}_COEF_{n}` for l=1..=3. When this extension is
/// present, the GLB's mesh primitive will NOT carry SH-rest accessors at all
/// — decoders are expected to load `<sidecar_uri>` (resolved relative to the
/// GLB) and rebuild SH-rest from `(codebook, indices)`. Sidecar wire format
/// is in `catetus-optimize/src/vq_palette.rs::ShRestPaletteSidetable`
/// (magic `"SHPA"`, version `1`). The DC term continues to live in
/// `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` as a normal FP32 VEC3
/// accessor, so view-independent color is decoder-agnostic.
const CT_PALETTE: &str = "CT_gaussian_splatting_palette";
/// Vendor extension carrying a pointer to a `.glb.v5tail` sidecar containing
/// V5.2 joint-tail residuals (V5.1-F per-cell affine codec). See
/// [`v5_tail`] for the on-disk format. Listed in `extensionsUsed`; when
/// `V5TailRef::required` is true, also listed in `extensionsRequired`.
const CT_V5_TAIL: &str = "CT_v5_tail_residual";
/// Vendor extension marking ROTATION as SOG-style smallest-3 packed quaternion.
const CT_QUAT_SMALLEST3: &str = "CT_quat_smallest3";
/// Vendor extension marking SCALE and OPACITY accessors as log-/logit-space
/// quantized rather than linear. When present (with payload
/// `{"scale":"ln","opacity":"logit"}`):
///
/// * The `KHR_gaussian_splatting:SCALE` accessor's UBYTE/USHORT integers
///   dequantize to **log-space** values per the accessor's `min`/`max`
///   metadata. Decoder must apply `exp(v)` to recover linear scales.
/// * The `KHR_gaussian_splatting:OPACITY` accessor's UBYTE/USHORT integers
///   dequantize to **logit-space** values per the accessor's `min`/`max`
///   metadata (typically `min=-12, max=12`). Decoder must apply `sigmoid(v)`
///   to recover the `[0, 1]` opacity.
///
/// Background: 3DGS scales are heavy-tailed log-normal — uniform-in-linear
/// 8-bit quantization crushes the long tail (the majority of small splats)
/// to a single low bin. Opacity at the extremes (≈0 or ≈1) suffers similar
/// loss when round-tripped through `logit()` after a linear 8-bit quant.
/// Quantizing in log/logit space is information-preserving for both
/// distributions and lifts the visual quality back to SOG parity without
/// changing the byte budget.
///
/// Listed in `extensionsUsed` only — viewers that don't understand the
/// extension still load the asset, just with a distorted scale/opacity
/// distribution (the existing pre-fix behavior).
const CT_LOG_QUANT_ATTRS: &str = "CT_log_quant_attrs";

/// Phase 5 back-compat: GLB files written before the 2026-05-19 rename use
/// `SF_*` extension keys; the current encoder writes `CT_*`. For one minor
/// version cycle the decoder accepts both — at parse time we rewrite any
/// `SF_*` key on the document root or any primitive to its `CT_*` equivalent
/// (no-op if the modern key is already present) and emit a one-time
/// `tracing::warn`. Drop this shim plus `normalize_legacy_extensions` and the
/// associated test fixtures when the encoder has been at `CT_*` for at least
/// one minor version (target: 4-6 weeks from rename).
const LEGACY_SF_TO_CT: [(&str, &str); 7] = [
    ("SF_zstd_split_buffer", CT_ZSTD_SPLIT),
    ("SF_gaussian_splatting_palette", CT_PALETTE),
    ("SF_log_quant_attrs", CT_LOG_QUANT_ATTRS),
    ("SF_quat_smallest3", CT_QUAT_SMALLEST3),
    ("SF_v5_tail_residual", CT_V5_TAIL),
    ("SF_brotli_buffer", CT_BROTLI),
    ("SF_spatial_streaming_index", CT_INDEX),
];

fn normalize_legacy_extensions(extensions: &mut serde_json::Map<String, serde_json::Value>) {
    for (legacy, modern) in LEGACY_SF_TO_CT.iter() {
        if let Some(val) = extensions.remove(*legacy) {
            eprintln!(
                "[catetus-gltf] WARNING: reading deprecated {} extension. Re-encode with current encoder to use {}.",
                legacy, modern,
            );
            extensions.entry((*modern).to_string()).or_insert(val);
        }
    }
}

fn normalize_legacy_extensions_used(used: &mut Vec<String>) {
    for name in used.iter_mut() {
        for (legacy, modern) in LEGACY_SF_TO_CT.iter() {
            if name == legacy {
                *name = (*modern).to_string();
                break;
            }
        }
    }
}

const FLOAT: u32 = 5126;
const UBYTE: u32 = 5121;
const USHORT: u32 = 5123;
/// Unsigned 32-bit int (`5125`). SOG-style smallest-3 quaternion.
const UINT: u32 = 5125;
/// Signed SHORT (`5122`). Used for RC quaternion quantization
/// (`KHR_gaussian_splatting:ROTATION` with `normalized: true`).
const SHORT: u32 = 5122;
/// Signed BYTE (`5120`). Used for SH-rest quantization
/// (`KHR_gaussian_splatting:SH_DEGREE_l_COEF_n` with `normalized: true`).
const BYTE: u32 = 5120;

/// Per-spec attribute key for one of the five required per-splat attributes.
/// `POSITION` is core-glTF and never namespaced; the rest pick up the
/// `KHR_gaussian_splatting:` prefix under `SpecVersion::RcMay2026` and the
/// legacy underscore-prefix names under `SpecVersion::Pre2026`.
fn attr_name(spec: SpecVersion, base: AttrKind) -> &'static str {
    match (spec, base) {
        (_, AttrKind::Position) => "POSITION",
        (SpecVersion::RcMay2026, AttrKind::Rotation) => "KHR_gaussian_splatting:ROTATION",
        (SpecVersion::RcMay2026, AttrKind::Scale) => "KHR_gaussian_splatting:SCALE",
        (SpecVersion::RcMay2026, AttrKind::Opacity) => "KHR_gaussian_splatting:OPACITY",
        (SpecVersion::RcMay2026, AttrKind::ColorDc) => {
            // RC: DC color lives in the SH degree-0 slot, as a VEC3 FLOAT
            // accessor. There is no separate "_COLOR_DC".
            "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"
        }
        (SpecVersion::Pre2026, AttrKind::Rotation) => "_ROTATION",
        (SpecVersion::Pre2026, AttrKind::Scale) => "_SCALE",
        (SpecVersion::Pre2026, AttrKind::Opacity) => "_OPACITY",
        (SpecVersion::Pre2026, AttrKind::ColorDc) => "_COLOR_DC",
    }
}

#[derive(Debug, Clone, Copy)]
enum AttrKind {
    Position,
    Rotation,
    Scale,
    Opacity,
    /// DC term of the colour SH. In RC, this is `SH_DEGREE_0_COEF_0`;
    /// in pre-RC, the historical `_COLOR_DC`.
    ColorDc,
}

/// Number of SH coefficients per degree (1, 3, 5, 7) — matches the RC text
/// and `catetus-khr-conformance::sh_coef_count`.
const SH_COEFS_PER_DEGREE: [usize; 4] = [1, 3, 5, 7];

/// Number of *non-DC* SH coefficients packed in `Color::Sh::coeffs` after
/// the leading three DC entries. Our PLY pipeline always materialises a
/// 48-scalar coeffs vector (3 DC + 45 trailing), regardless of the
/// effective degree, so degrees 1..3 carry (3+5+7)*3 = 45 scalars total.
const NON_DC_SH_SCALARS: usize = 45;

/// RC name for `SH_DEGREE_l_COEF_n` (always namespaced).
fn sh_attr_name(l: u8, n: u8) -> String {
    format!("KHR_gaussian_splatting:SH_DEGREE_{l}_COEF_{n}")
}

fn add_view_acc(
    root: &mut GltfRoot,
    buffer_idx: usize,
    offset: &mut usize,
    count: usize,
    byte_len: usize,
    accessor_ty: &str,
) -> usize {
    add_view_acc_typed(
        root,
        buffer_idx,
        offset,
        count,
        byte_len,
        accessor_ty,
        FLOAT,
    )
}

fn add_view_acc_typed(
    root: &mut GltfRoot,
    buffer_idx: usize,
    offset: &mut usize,
    count: usize,
    byte_len: usize,
    accessor_ty: &str,
    component_type: u32,
) -> usize {
    let bv = root.buffer_views.len();
    root.buffer_views.push(GltfBufferView {
        buffer: buffer_idx,
        byte_offset: *offset,
        byte_length: byte_len,
    });
    let acc = root.accessors.len();
    root.accessors.push(GltfAccessor {
        buffer_view: bv,
        component_type,
        count,
        accessor_type: accessor_ty.to_string(),
        normalized: false,
        min: None,
        max: None,
    });
    *offset += byte_len;
    acc
}

fn set_accessor_minmax(root: &mut GltfRoot, acc: usize, mn: [f32; 3], mx: [f32; 3]) {
    if let Some(a) = root.accessors.get_mut(acc) {
        a.min = Some(mn.to_vec());
        a.max = Some(mx.to_vec());
    }
}

/// Holds every per-splat accessor index for a single chunk after the writer
/// has emitted them. `sh_coef_accs` is one entry per SH coefficient (degrees
/// 1..=top_sh_degree, in (l, n) order) under `SpecVersion::RcMay2026`, or a
/// single index pointing at the legacy SCALAR-of-45 buffer under
/// `SpecVersion::Pre2026`.
#[derive(Debug, Clone)]
struct ChunkAccessors {
    pos: usize,
    rot: usize,
    scale: usize,
    opacity: usize,
    /// DC color. Under RC, this is `SH_DEGREE_0_COEF_0`; under pre-RC, `_COLOR_DC`.
    color_dc: usize,
    /// SH coefficient accessors (excluding DC).
    /// * RC: one entry per coef, paired with `(l, n)`.
    /// * Pre-RC: at most one entry, `(l, n)` is `(0, 0)` and unused.
    sh_coefs: Vec<((u8, u8), usize)>,
    /// Highest SH degree carried by the chunk (`0` when no SH).
    sh_degree: u8,
}

/// Per-attribute byte sizes for a single splat, given a spec version and
/// quantization plan. Used by both the binary packer and accessor emitter so
/// they never drift.
#[derive(Debug, Clone)]
struct ChunkLayout<'a> {
    spec: SpecVersion,
    quantize: bool,
    /// When true and `quantize` is set, ROTATION is emitted as a normalized
    /// signed SHORT (`5122 / VEC4 / normalized`).
    quantize_rotation: bool,
    /// Highest SH degree the chunk contains (0 = none).
    sh_degree: u8,
    /// Per-channel SH-rest quantization side table. Borrowed from `WriteOpts`.
    sh_rest_quant: Option<&'a ShRestQuantTable>,
    /// SOG-style smallest-3 quaternion side table. When `Some`, ROTATION is
    /// emitted as a SCALAR UNSIGNED_INT accessor (4 B/splat).
    rotation_smallest3: Option<&'a RotationSmallest3Table>,
    rotation_quant: Option<&'a RotationQuantTable>,
    dc_quant: Option<&'a DcQuantTable>,
    /// When true, SH-rest accessors (and their bytes in the BIN chunk) are
    /// SKIPPED entirely — SH-rest lives in the `.shpal` sidecar referenced by
    /// the `CT_gaussian_splatting_palette` root extension. Mutually exclusive
    /// with `sh_rest_quant` (palette wins; the quant table is ignored when
    /// elision is on). DC color (`SH_DEGREE_0_COEF_0`) is still emitted.
    palette_elision: bool,
    /// When true (and `quantize` is set), SCALE is quantized in log space
    /// and OPACITY in logit space. Mirrors `WriteOpts::log_quant_attrs`.
    /// See [`CT_LOG_QUANT_ATTRS`].
    log_quant_attrs: bool,
}

impl<'a> ChunkLayout<'a> {
    fn position_bytes(&self) -> usize {
        if self.quantize {
            6
        } else {
            12
        }
    }
    fn rotation_bytes(&self) -> usize {
        if self.rotation_smallest3.is_some() {
            // SCALAR UNSIGNED_INT packed [q0|q1|q2|tag] u32.
            4
        } else if let Some(t) = self.rotation_quant {
            let bits = t.bits.clamp(2, 16);
            if bits <= 8 {
                4
            } else {
                8
            }
        } else if self.quantize && self.quantize_rotation {
            8 // VEC4 i16
        } else {
            16 // VEC4 f32
        }
    }
    fn scale_bytes(&self) -> usize {
        if self.quantize {
            3
        } else {
            12
        }
    }
    fn opacity_bytes(&self) -> usize {
        if self.quantize {
            1
        } else {
            4
        }
    }
    /// DC color size. Under RC, DC must remain VEC3 FLOAT to satisfy
    /// `ACC_SH_COEF`; pre-RC keeps the historical u8-normalized path.
    fn dc_bytes(&self) -> usize {
        if let Some(t) = self.dc_quant {
            let bits = t.bits.clamp(2, 16);
            return if bits <= 8 { 3 } else { 6 };
        }
        match self.spec {
            SpecVersion::RcMay2026 => 12, // VEC3 FLOAT — non-negotiable per RC.
            SpecVersion::Pre2026 => {
                if self.quantize {
                    3
                } else {
                    12
                }
            }
        }
    }
}

/// Emit every per-splat accessor for `chunk` into `root`, returning the
/// indices in `ChunkAccessors` order. The byte stride exactly matches the
/// output of `pack_chunk_with` for the same `layout`.
///
/// `scale_min`/`scale_max` and `opacity_min`/`opacity_max` are passed in the
/// **same space** the bytes were quantized in (linear for the default path,
/// log/logit when `layout.log_quant_attrs` is set). They land in the
/// accessor's `min`/`max` metadata verbatim.
#[allow(clippy::too_many_arguments)]
fn emit_chunk_accessors(
    root: &mut GltfRoot,
    buffer_idx: usize,
    n: usize,
    pos_min: &[f32; 3],
    pos_max: &[f32; 3],
    scale_min: &[f32; 3],
    scale_max: &[f32; 3],
    opacity_min: &[f32; 1],
    opacity_max: &[f32; 1],
    layout: &ChunkLayout,
) -> ChunkAccessors {
    let mut offset = 0usize;

    // POSITION.
    let pos = if layout.quantize {
        let acc = add_view_acc_typed(root, buffer_idx, &mut offset, n, n * 6, "VEC3", USHORT);
        if let Some(a) = root.accessors.get_mut(acc) {
            a.normalized = true;
            a.min = Some(pos_min.to_vec());
            a.max = Some(pos_max.to_vec());
        }
        offset = align_up(offset, 4);
        acc
    } else {
        let acc = add_view_acc(root, buffer_idx, &mut offset, n, n * 12, "VEC3");
        set_accessor_minmax(root, acc, *pos_min, *pos_max);
        acc
    };

    // ROTATION.
    let rot = if layout.rotation_smallest3.is_some() {
        // SOG-style smallest-3 packed quaternion. 4 B/splat, SCALAR
        // UNSIGNED_INT (5125). The `CT_quat_smallest3` root extension tells
        // the decoder how to extract (q0,q1,q2,tag) from the u32 and rebuild
        // the unit quaternion. See
        // `experiments/SOG_STUDY_RUN/SMALLEST3_QUAT_RESULT.md`.
        let acc = add_view_acc_typed(root, buffer_idx, &mut offset, n, n * 4, "SCALAR", UINT);
        offset = align_up(offset, 4);
        acc
    } else if let Some(table) = layout.rotation_quant {
        // 8-bit packed per-component rotation. VEC4 UBYTE/USHORT normalized.
        let bits = table.bits.clamp(2, 16);
        let (comp_ty, bpc) = if bits <= 8 {
            (UBYTE, 1usize)
        } else {
            (USHORT, 2usize)
        };
        let total = n * 4 * bpc;
        let acc = add_view_acc_typed(root, buffer_idx, &mut offset, n, total, "VEC4", comp_ty);
        if let Some(a) = root.accessors.get_mut(acc) {
            a.normalized = true;
            a.min = Some(table.mins.to_vec());
            a.max = Some(table.maxs.to_vec());
        }
        offset = align_up(offset, 4);
        acc
    } else if layout.quantize && layout.quantize_rotation {
        // Normalized signed SHORT per the RC quaternion-quantization table:
        // each component lives in [-1, 1] and is reconstructed as max(q/32767, -1).
        let acc = add_view_acc_typed(root, buffer_idx, &mut offset, n, n * 8, "VEC4", SHORT);
        if let Some(a) = root.accessors.get_mut(acc) {
            a.normalized = true;
        }
        offset = align_up(offset, 4);
        acc
    } else {
        add_view_acc(root, buffer_idx, &mut offset, n, n * 16, "VEC4")
    };

    // SCALE.
    let scale = if layout.quantize {
        let acc = add_view_acc_typed(root, buffer_idx, &mut offset, n, n * 3, "VEC3", UBYTE);
        if let Some(a) = root.accessors.get_mut(acc) {
            a.normalized = true;
            a.min = Some(scale_min.to_vec());
            a.max = Some(scale_max.to_vec());
        }
        offset = align_up(offset, 4);
        acc
    } else {
        add_view_acc(root, buffer_idx, &mut offset, n, n * 12, "VEC3")
    };

    // OPACITY.
    let opacity = if layout.quantize {
        let acc = add_view_acc_typed(root, buffer_idx, &mut offset, n, n, "SCALAR", UBYTE);
        if let Some(a) = root.accessors.get_mut(acc) {
            a.normalized = true;
            a.min = Some(opacity_min.to_vec());
            a.max = Some(opacity_max.to_vec());
        }
        offset = align_up(offset, 4);
        acc
    } else {
        add_view_acc(root, buffer_idx, &mut offset, n, n * 4, "SCALAR")
    };

    // DC color. Under RC, must be VEC3 FLOAT to satisfy ACC_SH_COEF — the
    // validator requires every SH coefficient accessor (DC included) to be
    // VEC3 FLOAT, so we ignore `quantize` for the DC under RC.
    let color_dc = if let Some(table) = layout.dc_quant {
        let bits = table.bits.clamp(2, 16);
        let (comp_ty, bpc) = if bits <= 8 {
            (UBYTE, 1usize)
        } else {
            (USHORT, 2usize)
        };
        let total = n * 3 * bpc;
        let acc = add_view_acc_typed(root, buffer_idx, &mut offset, n, total, "VEC3", comp_ty);
        if let Some(a) = root.accessors.get_mut(acc) {
            a.normalized = true;
            a.min = Some(table.mins.to_vec());
            a.max = Some(table.maxs.to_vec());
        }
        offset = align_up(offset, 4);
        acc
    } else {
        match layout.spec {
            SpecVersion::RcMay2026 => {
                add_view_acc(root, buffer_idx, &mut offset, n, n * 12, "VEC3")
            }
            SpecVersion::Pre2026 => {
                if layout.quantize {
                    let acc =
                        add_view_acc_typed(root, buffer_idx, &mut offset, n, n * 3, "VEC3", UBYTE);
                    if let Some(a) = root.accessors.get_mut(acc) {
                        a.normalized = true;
                        a.min = Some(vec![0.0, 0.0, 0.0]);
                        a.max = Some(vec![1.0, 1.0, 1.0]);
                    }
                    offset = align_up(offset, 4);
                    acc
                } else {
                    add_view_acc(root, buffer_idx, &mut offset, n, n * 12, "VEC3")
                }
            }
        }
    };

    // SH coefficient accessors. When palette elision is on, we deliberately
    // skip every l>=1 accessor; the decoder is expected to rebuild SH-rest
    // from the `.shpal` sidecar referenced via `CT_gaussian_splatting_palette`.
    // DC color stays in `color_dc` above, so view-independent rendering
    // remains decoder-agnostic.
    let mut sh_coefs: Vec<((u8, u8), usize)> = Vec::new();
    if layout.sh_degree > 0 && !layout.palette_elision {
        match layout.spec {
            SpecVersion::RcMay2026 => {
                // One VEC3 accessor per coefficient at degrees 1..=top. When
                // a `sh_rest_quant` table is supplied, each accessor is a
                // normalized signed BYTE/SHORT with per-channel min/max set
                // from the table (symmetric `[-r_ch, +r_ch]`); otherwise
                // FP32 is preserved.
                let mut coef_idx = 0usize;
                for l in 1..=layout.sh_degree {
                    for n_idx in 0..SH_COEFS_PER_DEGREE[l as usize] as u8 {
                        let acc = if let Some(table) = &layout.sh_rest_quant {
                            let bits = table.bits.clamp(2, 16);
                            let (comp_ty, bytes_per_comp) = if bits <= 8 {
                                (BYTE, 1usize)
                            } else {
                                (SHORT, 2usize)
                            };
                            let coef_bytes = n * 3 * bytes_per_comp;
                            let acc = add_view_acc_typed(
                                root,
                                buffer_idx,
                                &mut offset,
                                n,
                                coef_bytes,
                                "VEC3",
                                comp_ty,
                            );
                            let base = coef_idx * 3;
                            let r0 = table.ranges.get(base).copied().unwrap_or(1e-9);
                            let r1 = table.ranges.get(base + 1).copied().unwrap_or(1e-9);
                            let r2 = table.ranges.get(base + 2).copied().unwrap_or(1e-9);
                            if let Some(a) = root.accessors.get_mut(acc) {
                                a.normalized = true;
                                a.min = Some(vec![-r0, -r1, -r2]);
                                a.max = Some(vec![r0, r1, r2]);
                            }
                            // Pad to 4-byte alignment so subsequent
                            // bufferViews stay glTF-validator clean.
                            offset = align_up(offset, 4);
                            acc
                        } else {
                            add_view_acc(root, buffer_idx, &mut offset, n, n * 12, "VEC3")
                        };
                        sh_coefs.push(((l, n_idx), acc));
                        coef_idx += 1;
                    }
                }
            }
            SpecVersion::Pre2026 => {
                // Legacy: one SCALAR FLOAT accessor of 45 entries per splat.
                let acc = add_view_acc(
                    root,
                    buffer_idx,
                    &mut offset,
                    n,
                    n * NON_DC_SH_SCALARS * 4,
                    "SCALAR",
                );
                sh_coefs.push(((0, 0), acc));
            }
        }
    }

    ChunkAccessors {
        pos,
        rot,
        scale,
        opacity,
        color_dc,
        sh_coefs,
        sh_degree: layout.sh_degree,
    }
}

/// Build the per-primitive JSON object that goes into `meshes[…].primitives`.
/// Encodes the spec-version-specific structural choices in one place: under
/// `RcMay2026` the attributes live on the primitive (next to `mode`) and the
/// extension blob carries `kernel` + `colorSpace`; under `Pre2026` the
/// historical Catetus layout is preserved.
fn build_primitive(spec: SpecVersion, accs: &ChunkAccessors) -> serde_json::Value {
    let mut attrs = serde_json::Map::new();
    attrs.insert(
        attr_name(spec, AttrKind::Position).to_string(),
        serde_json::json!(accs.pos),
    );
    attrs.insert(
        attr_name(spec, AttrKind::Rotation).to_string(),
        serde_json::json!(accs.rot),
    );
    attrs.insert(
        attr_name(spec, AttrKind::Scale).to_string(),
        serde_json::json!(accs.scale),
    );
    attrs.insert(
        attr_name(spec, AttrKind::Opacity).to_string(),
        serde_json::json!(accs.opacity),
    );
    attrs.insert(
        attr_name(spec, AttrKind::ColorDc).to_string(),
        serde_json::json!(accs.color_dc),
    );
    match spec {
        SpecVersion::RcMay2026 => {
            for ((l, n), idx) in &accs.sh_coefs {
                attrs.insert(sh_attr_name(*l, *n), serde_json::json!(idx));
            }
            serde_json::json!({
                "mode": 0, // POINTS
                "attributes": serde_json::Value::Object(attrs),
                "extensions": {
                    KHR: {
                        "kernel": "ellipse",
                        "colorSpace": "srgb_rec709_display",
                        "projection": "perspective",
                        "sortingMethod": "cameraDistance",
                        "shDegree": accs.sh_degree,
                    }
                }
            })
        }
        SpecVersion::Pre2026 => {
            // Legacy: attributes live inside the extension and `_COLOR_SH`
            // is a single SCALAR-of-45 accessor.
            if let Some(((_l, _n), idx)) = accs.sh_coefs.first() {
                attrs.insert("_COLOR_SH".to_string(), serde_json::json!(idx));
            }
            serde_json::json!({
                "extensions": {
                    KHR: {
                        "attributes": serde_json::Value::Object(attrs),
                        "shDegree": accs.sh_degree,
                    }
                }
            })
        }
    }
}

/// Write a scene as `<dir>/scene.gltf` plus one or more `.bin` files under
/// `<dir>/buffers/`. `path` is the output `.gltf` path.
pub fn write_gltf(scene: &SplatScene, path: &Path, opts: &WriteOpts) -> Result<(), GltfError> {
    let dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let buffers_dir = dir.join("buffers");
    fs::create_dir_all(&buffers_dir)?;

    // Determine chunking.
    let chunks: Vec<&[Splat]> = if opts.chunked && opts.chunk_target_splats > 0 {
        scene.splats.chunks(opts.chunk_target_splats).collect()
    } else {
        vec![scene.splats.as_slice()]
    };

    let mut root = GltfRoot {
        asset: GltfAsset {
            version: "2.0".to_string(),
            generator: Some("catetus-gltf".to_string()),
        },
        extensions_used: vec![KHR.to_string()],
        extensions_required: vec![KHR.to_string()],
        buffers: Vec::new(),
        buffer_views: Vec::new(),
        accessors: Vec::new(),
        meshes: Vec::new(),
        extensions: serde_json::Map::new(),
    };
    if opts.chunked {
        root.extensions_used.push(CT_INDEX.to_string());
    }
    if opts.quantize {
        // Non-required: viewers that don't implement the extension still load
        // the asset, just with un-dequantized integer values. See SPEC-0013.
        root.extensions_used.push(KHR_QUANT.to_string());
    }
    if opts.quantize && opts.log_quant_attrs {
        // Signals to decoders that SCALE accessor values dequantize into
        // log-space (must apply `exp`) and OPACITY into logit-space (must
        // apply `sigmoid`). Listed in `extensionsUsed` only so legacy
        // decoders still load the asset, just with a distorted scale
        // distribution. See [`CT_LOG_QUANT_ATTRS`].
        root.extensions_used.push(CT_LOG_QUANT_ATTRS.to_string());
        root.extensions.insert(
            CT_LOG_QUANT_ATTRS.to_string(),
            serde_json::json!({ "scale": "ln", "opacity": "logit" }),
        );
    }

    let mut chunk_records: Vec<serde_json::Value> = Vec::new();
    let mut primitives: Vec<serde_json::Value> = Vec::new();
    // Scene-wide bbox spans all chunks. Initialized to ±infinity so the union
    // converges to the empty-scene fallback when no splats are present.
    let mut scene_min = [f32::INFINITY; 3];
    let mut scene_max = [f32::NEG_INFINITY; 3];
    let mut total_splat_count: usize = 0;

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let (chunk_min, chunk_max) = chunk_bbox(chunk);
        let log_quant = opts.quantize && opts.log_quant_attrs;
        let (scale_min, scale_max) = chunk_scale_bbox_in(chunk, log_quant);
        let (opacity_min, opacity_max) = chunk_opacity_bbox(chunk, log_quant);
        let chunk_sh_degree: u8 = chunk.iter().map(|s| s.color.degree()).max().unwrap_or(0);
        let layout = ChunkLayout {
            spec: opts.spec_version,
            quantize: opts.quantize,
            quantize_rotation: opts.quantize_rotation,
            sh_degree: chunk_sh_degree,
            sh_rest_quant: opts.sh_rest_quant.as_ref(),
            rotation_smallest3: opts.rotation_smallest3.as_ref(),
            rotation_quant: opts.rotation_quant.as_ref(),
            dc_quant: opts.dc_quant.as_ref(),
            palette_elision: opts.palette.is_some(),
            log_quant_attrs: opts.log_quant_attrs,
        };
        let buf_bytes = pack_chunk_with(chunk, &layout, &chunk_min, &chunk_max);
        let buf_name = format!("buffers/chunk_{chunk_idx:04}.bin");
        let buf_path = dir.join(&buf_name);
        fs::write(&buf_path, &buf_bytes)?;
        let buffer_idx = root.buffers.len();
        root.buffers.push(GltfBuffer {
            byte_length: buf_bytes.len(),
            uri: Some(buf_name.clone()),
        });

        let n = chunk.len();
        let accs = emit_chunk_accessors(
            &mut root,
            buffer_idx,
            n,
            &chunk_min,
            &chunk_max,
            &scale_min,
            &scale_max,
            &opacity_min,
            &opacity_max,
            &layout,
        );

        if !chunk.is_empty() {
            for i in 0..3 {
                if chunk_min[i] < scene_min[i] {
                    scene_min[i] = chunk_min[i];
                }
                if chunk_max[i] > scene_max[i] {
                    scene_max[i] = chunk_max[i];
                }
            }
            total_splat_count += n;
        }

        primitives.push(build_primitive(opts.spec_version, &accs));

        if opts.chunked {
            // Per-chunk record. `uri` mirrors `buffers[buffer_idx].uri` so JS
            // consumers that key off chunk.uri can resolve without indirection.
            let hash = blake3::hash(&buf_bytes).to_hex().to_string();
            chunk_records.push(serde_json::json!({
                "uri": buf_name,
                "buffer": buffer_idx,
                "byteOffset": 0,
                "byteLength": buf_bytes.len(),
                "splatCount": chunk.len(),
                "bbox": { "min": chunk_min, "max": chunk_max },
                "lod": 0,
                "checksum": format!("blake3:{hash}"),
                "loadPriority": chunk_idx,
            }));
        }
    }

    root.meshes
        .push(serde_json::json!({ "primitives": primitives }));

    // Top-level KHR extension carries scene-wide splatCount + bbox so viewers
    // can frame the asset without walking every primitive.
    let scene_bbox = if total_splat_count == 0 {
        ([0.0f32; 3], [0.0f32; 3])
    } else {
        (scene_min, scene_max)
    };
    let scene_sh_degree: u8 = scene
        .splats
        .iter()
        .map(|s| s.color.degree())
        .max()
        .unwrap_or(0);
    root.extensions.insert(
        KHR.to_string(),
        serde_json::json!({
            "splatCount": total_splat_count,
            "bbox": { "min": scene_bbox.0, "max": scene_bbox.1 },
            "shDegree": scene_sh_degree,
        }),
    );

    // Palette elision parity with `build_single_buffer_gltf` (GLB path) —
    // also expose `CT_gaussian_splatting_palette` on the gltf+bin layout so
    // a sidecar-aware viewer can rebuild SH-rest. Kept extensionsUsed-only
    // for backwards compat; legacy viewers fall through to DC-only.
    if let Some(pal) = &opts.palette {
        if !root.extensions_used.iter().any(|e| e == CT_PALETTE) {
            root.extensions_used.push(CT_PALETTE.to_string());
        }
        root.extensions.insert(
            CT_PALETTE.to_string(),
            serde_json::json!({
                "uri": pal.sidecar_uri,
                "paletteSize": pal.palette_size,
                "splatCount": pal.n_splats,
                "codebookBits": pal.codebook_bits,
                "shDegree": pal.sh_degree,
                "indexComponentType": USHORT,
                "format": "shpa-v1",
            }),
        );
    }

    // V5.2 joint-tail residual sidecar pointer. extensionsUsed by default;
    // also added to extensionsRequired when the caller marks the residual
    // as a hard requirement (`required = true`). See `crates/catetus-
    // gltf/src/v5_tail.rs` for the on-disk format.
    if let Some(tail) = &opts.v5_tail {
        if !root.extensions_used.iter().any(|e| e == CT_V5_TAIL) {
            root.extensions_used.push(CT_V5_TAIL.to_string());
        }
        if tail.required && !root.extensions_required.iter().any(|e| e == CT_V5_TAIL) {
            root.extensions_required.push(CT_V5_TAIL.to_string());
        }
        root.extensions.insert(
            CT_V5_TAIL.to_string(),
            serde_json::json!({
                "uri": tail.sidecar_uri,
                "splatCount": tail.n_splats,
                "kSelected": tail.k_selected,
                "shRestCoefs": tail.sh_rest_coefs,
                "nCells": tail.n_cells,
                "format": "sfv51tal-v2",
            }),
        );
    }

    // Smallest-3 quaternion marker. The ROTATION accessor is SCALAR
    // UNSIGNED_INT (5125) when this extension is present; the u32 payload is
    // `[q0:component_bits | q1:bits | q2:bits | tag:2]` (LE). Legacy viewers
    // see a SCALAR uint accessor — extensionsUsed-only keeps load graceful.
    if let Some(s3) = &opts.rotation_smallest3 {
        if !root.extensions_used.iter().any(|e| e == CT_QUAT_SMALLEST3) {
            root.extensions_used.push(CT_QUAT_SMALLEST3.to_string());
        }
        root.extensions.insert(
            CT_QUAT_SMALLEST3.to_string(),
            serde_json::json!({
                "componentBits": s3.component_bits,
                "componentType": UINT,
                "layout": "q0|q1|q2|tag",
                "tagBits": 2,
            }),
        );
    }

    if opts.chunked {
        let lods: Vec<serde_json::Value> = opts
            .lod_fractions
            .iter()
            .enumerate()
            .map(|(i, f)| serde_json::json!({ "level": i, "splatFraction": f }))
            .collect();
        root.extensions.insert(
            CT_INDEX.to_string(),
            serde_json::json!({
                "ordering": "morton",
                "chunkCount": chunk_records.len(),
                "chunks": chunk_records,
                "lods": lods,
            }),
        );
    }

    let json = serde_json::to_string_pretty(&root)?;
    fs::write(path, json)?;
    Ok(())
}

/// Round `x` up to the next multiple of `align`. `align` must be a power of two.
#[inline]
fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Encode one f32 component into a u16 normalized integer (`[min, max]` → `[0, 65535]`).
#[inline]
fn quantize_u16(v: f32, lo: f32, hi: f32) -> u16 {
    let span = (hi - lo).max(f32::EPSILON);
    let t = ((v - lo) / span).clamp(0.0, 1.0);
    (t * 65535.0 + 0.5) as u16
}

#[inline]
fn dequantize_u16(q: u16, lo: f32, hi: f32) -> f32 {
    let t = q as f32 / 65535.0;
    lo + t * (hi - lo)
}

/// Encode one f32 component into a u8 normalized integer (`[min, max]` → `[0, 255]`).
#[inline]
fn quantize_u8(v: f32, lo: f32, hi: f32) -> u8 {
    let span = (hi - lo).max(f32::EPSILON);
    let t = ((v - lo) / span).clamp(0.0, 1.0);
    (t * 255.0 + 0.5) as u8
}

#[inline]
fn dequantize_u8(q: u8, lo: f32, hi: f32) -> f32 {
    let t = q as f32 / 255.0;
    lo + t * (hi - lo)
}

/// Encode one f32 in `[-1, 1]` as a signed SHORT (`i16`) using the
/// `KHR_mesh_quantization` / glTF 2.0 normalized-signed reconstruction
/// `f = max(c / 32767.0, -1.0)`. The RC quaternion table specifies this
/// exact formula for `KHR_gaussian_splatting:ROTATION` accessors.
#[inline]
fn quantize_i16(v: f32) -> i16 {
    let t = v.clamp(-1.0, 1.0);
    (t * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

#[inline]
fn dequantize_i16(q: i16) -> f32 {
    (q as f32 / 32767.0).max(-1.0)
}

/// Per-axis bbox of `scale` over the chunk, optionally in **log space**.
/// When `log_space` is set, returns `(ln(min), ln(max))` so the GLB writer
/// can emit accessor `min`/`max` metadata in the same transformed space its
/// `pack_chunk_with` is quantizing in.
fn chunk_scale_bbox_in(chunk: &[Splat], log_space: bool) -> ([f32; 3], [f32; 3]) {
    if chunk.is_empty() {
        return ([0.0; 3], [1.0; 3]);
    }
    let fwd = |v: f32| -> f32 {
        if log_space {
            v.max(f32::MIN_POSITIVE).ln()
        } else {
            v
        }
    };
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for s in chunk {
        for i in 0..3 {
            let t = fwd(s.scale[i]);
            if t < mn[i] {
                mn[i] = t;
            }
            if t > mx[i] {
                mx[i] = t;
            }
        }
    }
    for i in 0..3 {
        if mx[i] <= mn[i] {
            mx[i] = mn[i] + f32::EPSILON;
        }
    }
    (mn, mx)
}

/// Per-component min/max of `opacity` over the chunk, optionally in
/// **logit space**. When `logit_space` is set, returns
/// `(logit(min_opacity), logit(max_opacity))`. The result is clamped to a
/// safe `±LOGIT_RANGE` band so values very close to 0 or 1 don't blow the
/// dequant span to infinity.
const OPACITY_LOGIT_RANGE: f32 = 12.0;

fn chunk_opacity_bbox(chunk: &[Splat], logit_space: bool) -> ([f32; 1], [f32; 1]) {
    if chunk.is_empty() {
        return if logit_space {
            ([-OPACITY_LOGIT_RANGE], [OPACITY_LOGIT_RANGE])
        } else {
            ([0.0], [1.0])
        };
    }
    let fwd = |p: f32| -> f32 {
        if logit_space {
            // Clamp inside (0, 1) so logit is finite, then clamp the logit
            // itself to the conservative ±LOGIT_RANGE band that survives an
            // 8-bit grid with ~0.09 logit step (sigmoid(±12) ≈ 1±6e-6).
            let p = p.clamp(
                1.0 / (1.0 + OPACITY_LOGIT_RANGE.exp()),
                1.0 - 1.0 / (1.0 + OPACITY_LOGIT_RANGE.exp()),
            );
            (p / (1.0 - p))
                .ln()
                .clamp(-OPACITY_LOGIT_RANGE, OPACITY_LOGIT_RANGE)
        } else {
            p
        }
    };
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for s in chunk {
        let t = fwd(s.opacity);
        if t < mn {
            mn = t;
        }
        if t > mx {
            mx = t;
        }
    }
    if mx <= mn {
        mx = mn + f32::EPSILON;
    }
    ([mn], [mx])
}

fn pack_chunk_with(
    chunk: &[Splat],
    layout: &ChunkLayout,
    pos_min: &[f32; 3],
    pos_max: &[f32; 3],
) -> Vec<u8> {
    let n = chunk.len();
    // Compute bboxes in the same space we'll be quantizing in. When
    // `log_quant_attrs` is set, SCALE rides log-space and OPACITY rides
    // logit-space — uniform-in-linear UBYTE quantization on a heavy-tailed
    // distribution otherwise crushes the long tail to a single low bin and
    // mangles the PLY round-trip (see CT_LOG_QUANT_ATTRS docs).
    let log_quant = layout.quantize && layout.log_quant_attrs;
    let (scale_min, scale_max) = chunk_scale_bbox_in(chunk, log_quant);
    let (opacity_min, opacity_max) = chunk_opacity_bbox(chunk, log_quant);
    let scale_fwd = |v: f32| -> f32 {
        if log_quant {
            v.max(f32::MIN_POSITIVE).ln()
        } else {
            v
        }
    };
    let opacity_fwd = |p: f32| -> f32 {
        if log_quant {
            let p = p.clamp(
                1.0 / (1.0 + OPACITY_LOGIT_RANGE.exp()),
                1.0 - 1.0 / (1.0 + OPACITY_LOGIT_RANGE.exp()),
            );
            (p / (1.0 - p))
                .ln()
                .clamp(-OPACITY_LOGIT_RANGE, OPACITY_LOGIT_RANGE)
        } else {
            p
        }
    };

    let mut out: Vec<u8> = Vec::with_capacity(
        n * (layout.position_bytes()
            + layout.rotation_bytes()
            + layout.scale_bytes()
            + layout.opacity_bytes()
            + layout.dc_bytes()),
    );

    // POSITION.
    if layout.quantize {
        for s in chunk {
            for i in 0..3 {
                let q = quantize_u16(s.position[i], pos_min[i], pos_max[i]);
                out.write_u16::<LittleEndian>(q).unwrap();
            }
        }
        pad_to(&mut out, 4);
    } else {
        for s in chunk {
            for v in s.position {
                out.write_f32::<LittleEndian>(v).unwrap();
            }
        }
    }

    // ROTATION.
    if let Some(s3) = layout.rotation_smallest3 {
        // SOG-style smallest-3: pack 3 components + 2-bit tag into u32. The
        // pass already round-trip-dequantized the scene so re-encoding here
        // produces bit-identical bytes to what the renderer will reconstruct.
        let bits = s3.component_bits.clamp(6, 10);
        let levels = ((1u32 << bits) - 1) as f32;
        let sqrt2 = std::f32::consts::SQRT_2;
        let mask = (1u32 << bits) - 1;
        for s in chunk {
            let r = s.rotation;
            let n2: f32 = r.iter().map(|v| v * v).sum::<f32>();
            let n = n2.sqrt().max(1e-12);
            let q = [r[0] / n, r[1] / n, r[2] / n, r[3] / n];
            let mut tag: u8 = 0;
            let mut best = q[0].abs();
            for i in 1..4 {
                let a = q[i].abs();
                if a > best {
                    best = a;
                    tag = i as u8;
                }
            }
            let sgn = if q[tag as usize] < 0.0 { -1.0 } else { 1.0 };
            let mut comps = [0u32; 3];
            let mut k = 0usize;
            for i in 0..4 {
                if i as u8 == tag {
                    continue;
                }
                let v = q[i] * sgn;
                let t = (v / sqrt2 + 0.5).clamp(0.0, 1.0);
                comps[k] = (t * levels).round() as u32 & mask;
                k += 1;
            }
            let packed = comps[0]
                | (comps[1] << bits)
                | (comps[2] << (2 * bits))
                | (((tag as u32) & 3) << 30);
            out.write_u32::<LittleEndian>(packed).unwrap();
        }
        pad_to(&mut out, 4);
    } else if let Some(table) = layout.rotation_quant {
        let bits = table.bits.clamp(2, 16);
        if bits <= 8 {
            for s in chunk {
                for i in 0..4 {
                    out.push(quantize_u8(s.rotation[i], table.mins[i], table.maxs[i]));
                }
            }
        } else {
            for s in chunk {
                for i in 0..4 {
                    out.write_u16::<LittleEndian>(quantize_u16(
                        s.rotation[i],
                        table.mins[i],
                        table.maxs[i],
                    ))
                    .unwrap();
                }
            }
        }
        pad_to(&mut out, 4);
    } else if layout.quantize && layout.quantize_rotation {
        for s in chunk {
            for v in s.rotation {
                out.write_i16::<LittleEndian>(quantize_i16(v)).unwrap();
            }
        }
        pad_to(&mut out, 4);
    } else {
        for s in chunk {
            for v in s.rotation {
                out.write_f32::<LittleEndian>(v).unwrap();
            }
        }
    }

    // SCALE.
    if layout.quantize {
        for s in chunk {
            for i in 0..3 {
                out.push(quantize_u8(
                    scale_fwd(s.scale[i]),
                    scale_min[i],
                    scale_max[i],
                ));
            }
        }
        pad_to(&mut out, 4);
    } else {
        for s in chunk {
            for v in s.scale {
                out.write_f32::<LittleEndian>(v).unwrap();
            }
        }
    }

    // OPACITY.
    if layout.quantize {
        for s in chunk {
            out.push(quantize_u8(
                opacity_fwd(s.opacity),
                opacity_min[0],
                opacity_max[0],
            ));
        }
        pad_to(&mut out, 4);
    } else {
        for s in chunk {
            out.write_f32::<LittleEndian>(s.opacity).unwrap();
        }
    }

    // DC color. With `dc_quant` set, per-channel min/max UBYTE/USHORT.
    let dc_of = |s: &Splat| -> [f32; 3] {
        match &s.color {
            Color::Rgb(c) => *c,
            Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
        }
    };
    if let Some(table) = layout.dc_quant {
        let bits = table.bits.clamp(2, 16);
        if bits <= 8 {
            for s in chunk {
                let dc = dc_of(s);
                for i in 0..3 {
                    out.push(quantize_u8(dc[i], table.mins[i], table.maxs[i]));
                }
            }
        } else {
            for s in chunk {
                let dc = dc_of(s);
                for i in 0..3 {
                    out.write_u16::<LittleEndian>(quantize_u16(
                        dc[i],
                        table.mins[i],
                        table.maxs[i],
                    ))
                    .unwrap();
                }
            }
        }
        pad_to(&mut out, 4);
    } else {
        let dc_as_float = match layout.spec {
            SpecVersion::RcMay2026 => true,
            SpecVersion::Pre2026 => !layout.quantize,
        };
        if dc_as_float {
            for s in chunk {
                for v in dc_of(s) {
                    out.write_f32::<LittleEndian>(v).unwrap();
                }
            }
        } else {
            for s in chunk {
                for v in dc_of(s) {
                    out.push(quantize_u8(v, 0.0, 1.0));
                }
            }
            pad_to(&mut out, 4);
        }
    }

    // SH coefficients (degrees 1..=top). Skipped entirely under palette
    // elision — `emit_chunk_accessors` produces zero SH-rest accessors in
    // that mode, so emitting the bytes would create orphan bufferView ranges
    // that the loader would never index.
    if layout.sh_degree > 0 && !layout.palette_elision {
        match layout.spec {
            SpecVersion::RcMay2026 => {
                // Per-coefficient VEC3 accessors. When a `sh_rest_quant`
                // table is supplied, each scalar is symmetric-quantized to
                // a normalized signed BYTE / SHORT using the per-channel
                // range; otherwise the legacy FP32 path runs.
                let coef_count: usize = SH_COEFS_PER_DEGREE[1..=layout.sh_degree as usize]
                    .iter()
                    .sum();
                let table = layout.sh_rest_quant;
                for coef_idx in 0..coef_count {
                    let base_ch = coef_idx * 3;
                    let (bits, ranges) = match table {
                        Some(t) => (t.bits.clamp(2, 16), Some(&t.ranges)),
                        None => (0u8, None),
                    };
                    let levels = if bits == 0 {
                        0.0
                    } else {
                        ((1u32 << (bits as u32 - 1)) - 1) as f32
                    };
                    let use_short = bits > 8;
                    for s in chunk {
                        let vals: [f32; 3] = match &s.color {
                            Color::Sh { coeffs, .. } => {
                                let base = 3 + coef_idx * 3;
                                [
                                    coeffs.get(base).copied().unwrap_or(0.0),
                                    coeffs.get(base + 1).copied().unwrap_or(0.0),
                                    coeffs.get(base + 2).copied().unwrap_or(0.0),
                                ]
                            }
                            Color::Rgb(_) => [0.0, 0.0, 0.0],
                        };
                        if let Some(rs) = ranges {
                            for (k, &val) in vals.iter().enumerate() {
                                let r = rs.get(base_ch + k).copied().unwrap_or(1e-9).max(1e-9);
                                let t = (val / r).clamp(-1.0, 1.0);
                                let q = (t * levels).round();
                                if use_short {
                                    let qi = q.clamp(-32767.0, 32767.0) as i16;
                                    out.write_i16::<LittleEndian>(qi).unwrap();
                                } else {
                                    let qi = q.clamp(-127.0, 127.0) as i8;
                                    out.push(qi as u8);
                                }
                            }
                        } else {
                            for v in vals {
                                out.write_f32::<LittleEndian>(v).unwrap();
                            }
                        }
                    }
                    if table.is_some() {
                        pad_to(&mut out, 4);
                    }
                }
            }
            SpecVersion::Pre2026 => {
                // Legacy: single SCALAR-of-45 buffer.
                for s in chunk {
                    match &s.color {
                        Color::Sh { coeffs, .. } => {
                            for i in 0..NON_DC_SH_SCALARS {
                                let v = coeffs.get(i + 3).copied().unwrap_or(0.0);
                                out.write_f32::<LittleEndian>(v).unwrap();
                            }
                        }
                        Color::Rgb(_) => {
                            for _ in 0..NON_DC_SH_SCALARS {
                                out.write_f32::<LittleEndian>(0.0).unwrap();
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[inline]
fn pad_to(buf: &mut Vec<u8>, align: usize) {
    let target = align_up(buf.len(), align);
    while buf.len() < target {
        buf.push(0);
    }
}

/// Decode an accessor's raw byte range into a flat `Vec<f32>` of length
/// `accessor.count * comps`. Handles both FLOAT and the `KHR_mesh_quantization`
/// integer accessor variants (UNSIGNED_BYTE / UNSIGNED_SHORT, normalized).
fn decode_accessor(bytes: &[u8], acc: &GltfAccessor, comps: usize) -> Result<Vec<f32>, GltfError> {
    let total = acc.count * comps;
    match acc.component_type {
        FLOAT => {
            if bytes.len() < total * 4 {
                return Err(GltfError::Malformed("accessor under-sized".to_string()));
            }
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let c = &bytes[i * 4..i * 4 + 4];
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        USHORT => {
            if bytes.len() < total * 2 {
                return Err(GltfError::Malformed("u16 accessor under-sized".to_string()));
            }
            let lo = acc.min.clone();
            let hi = acc.max.clone();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let c = &bytes[i * 2..i * 2 + 2];
                let q = u16::from_le_bytes([c[0], c[1]]);
                let comp = i % comps;
                let v = if acc.normalized {
                    match (&lo, &hi) {
                        (Some(lo), Some(hi)) if lo.len() == comps && hi.len() == comps => {
                            dequantize_u16(q, lo[comp], hi[comp])
                        }
                        _ => q as f32 / 65535.0,
                    }
                } else {
                    q as f32
                };
                out.push(v);
            }
            Ok(out)
        }
        UBYTE => {
            if bytes.len() < total {
                return Err(GltfError::Malformed("u8 accessor under-sized".to_string()));
            }
            let lo = acc.min.clone();
            let hi = acc.max.clone();
            let mut out = Vec::with_capacity(total);
            for (i, &q) in bytes.iter().take(total).enumerate() {
                let comp = i % comps;
                let v = if acc.normalized {
                    match (&lo, &hi) {
                        (Some(lo), Some(hi)) if lo.len() == comps && hi.len() == comps => {
                            dequantize_u8(q, lo[comp], hi[comp])
                        }
                        _ => q as f32 / 255.0,
                    }
                } else {
                    q as f32
                };
                out.push(v);
            }
            Ok(out)
        }
        SHORT => {
            // Signed SHORT — used for the RC normalized quaternion encoding
            // and for SH-rest @ bits > 8 (with per-channel min/max ranges).
            if bytes.len() < total * 2 {
                return Err(GltfError::Malformed("i16 accessor under-sized".to_string()));
            }
            let lo = acc.min.clone();
            let hi = acc.max.clone();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let c = &bytes[i * 2..i * 2 + 2];
                let q = i16::from_le_bytes([c[0], c[1]]);
                let v = if acc.normalized {
                    // If per-component min/max are present we treat them as a
                    // symmetric range [-r, r] (SH-rest path); otherwise we
                    // fall back to the gltf-spec [-1, 1] reconstruction.
                    let comp = i % comps;
                    match (&lo, &hi) {
                        (Some(lo), Some(hi)) if lo.len() == comps && hi.len() == comps => {
                            let r = hi[comp].max(-lo[comp]).max(1e-9);
                            dequantize_i16(q) * r
                        }
                        _ => dequantize_i16(q),
                    }
                } else {
                    q as f32
                };
                out.push(v);
            }
            Ok(out)
        }
        BYTE => {
            // Signed BYTE — used for SH-rest quantization @ bits <= 8.
            // glTF normalized reconstruction is `f = max(c/127, -1)`; we
            // scale by the per-channel range pulled from min/max.
            if bytes.len() < total {
                return Err(GltfError::Malformed("i8 accessor under-sized".to_string()));
            }
            let lo = acc.min.clone();
            let hi = acc.max.clone();
            let mut out = Vec::with_capacity(total);
            for (i, &raw) in bytes.iter().enumerate().take(total) {
                let q = raw as i8;
                let v = if acc.normalized {
                    let comp = i % comps;
                    let base = (q as f32 / 127.0).max(-1.0);
                    match (&lo, &hi) {
                        (Some(lo), Some(hi)) if lo.len() == comps && hi.len() == comps => {
                            let r = hi[comp].max(-lo[comp]).max(1e-9);
                            base * r
                        }
                        _ => base,
                    }
                } else {
                    q as f32
                };
                out.push(v);
            }
            Ok(out)
        }
        other => Err(GltfError::Malformed(format!(
            "unsupported componentType {other}"
        ))),
    }
}

/// Indices of the per-splat accessors discovered on read, regardless of
/// whether the source asset used the pre-RC or RC attribute schema. SH
/// coefficients are flattened back into the legacy 45-scalar layout under
/// `non_dc_sh`: 15 coefficients × 3 channels, in `(degree, n, channel)`
/// order, matching `Color::Sh::coeffs[3..]`.
#[derive(Debug, Clone)]
struct ResolvedAttrs {
    spec: SpecVersion,
    pos: usize,
    rot: usize,
    scale: usize,
    opacity: usize,
    dc: usize,
    /// One entry per non-DC SH coefficient (RC) or one entry pointing at
    /// the legacy SCALAR-of-45 accessor (pre-RC); empty when no SH.
    non_dc_sh: Vec<usize>,
}

fn resolve_attrs(primitive: &serde_json::Value) -> Result<ResolvedAttrs, GltfError> {
    // RC: attributes live on the primitive directly (next to `mode` /
    // `indices`). Pre-RC: attributes are inside the KHR extension blob.
    let prim_attrs = primitive.get("attributes").and_then(|a| a.as_object());
    let ext_attrs = primitive
        .get("extensions")
        .and_then(|e| e.get(KHR))
        .and_then(|k| k.get("attributes"))
        .and_then(|a| a.as_object());

    // Sniff for RC by looking at where ROTATION lives. Either object can
    // contain `POSITION`; we discriminate on the namespaced ROTATION key.
    let in_prim = prim_attrs
        .map(|m| m.contains_key("KHR_gaussian_splatting:ROTATION"))
        .unwrap_or(false);
    let in_ext = ext_attrs
        .map(|m| m.contains_key("_ROTATION"))
        .unwrap_or(false);

    let (spec, attrs) = if in_prim {
        (SpecVersion::RcMay2026, prim_attrs.unwrap())
    } else if in_ext {
        (SpecVersion::Pre2026, ext_attrs.unwrap())
    } else if let Some(p) = prim_attrs {
        // Spec hint missing — assume RC since it's the default forward.
        (SpecVersion::RcMay2026, p)
    } else if let Some(p) = ext_attrs {
        (SpecVersion::Pre2026, p)
    } else {
        return Err(GltfError::Malformed("no attributes".to_string()));
    };

    let get_idx = |name: &str| -> Option<usize> {
        attrs.get(name).and_then(|v| v.as_u64()).map(|n| n as usize)
    };

    let pos = get_idx(attr_name(spec, AttrKind::Position)).ok_or(GltfError::MissingExtension)?;
    let rot = get_idx(attr_name(spec, AttrKind::Rotation)).ok_or(GltfError::MissingExtension)?;
    let scale = get_idx(attr_name(spec, AttrKind::Scale)).ok_or(GltfError::MissingExtension)?;
    let opacity = get_idx(attr_name(spec, AttrKind::Opacity)).ok_or(GltfError::MissingExtension)?;
    let dc = get_idx(attr_name(spec, AttrKind::ColorDc)).ok_or(GltfError::MissingExtension)?;

    let mut non_dc_sh = Vec::new();
    match spec {
        SpecVersion::RcMay2026 => {
            // Walk degrees 1..=3; stop at the first missing degree-l 0th
            // coefficient (matches `SH_DEGREES_FULL`).
            for l in 1u8..=3 {
                let coef_count = SH_COEFS_PER_DEGREE[l as usize];
                if get_idx(&sh_attr_name(l, 0)).is_none() {
                    break;
                }
                for n in 0..coef_count as u8 {
                    if let Some(idx) = get_idx(&sh_attr_name(l, n)) {
                        non_dc_sh.push(idx);
                    }
                }
            }
        }
        SpecVersion::Pre2026 => {
            if let Some(idx) = get_idx("_COLOR_SH") {
                non_dc_sh.push(idx);
            }
        }
    }

    Ok(ResolvedAttrs {
        spec,
        pos,
        rot,
        scale,
        opacity,
        dc,
        non_dc_sh,
    })
}

fn chunk_bbox(chunk: &[Splat]) -> ([f32; 3], [f32; 3]) {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for s in chunk {
        for i in 0..3 {
            if s.position[i] < mn[i] {
                mn[i] = s.position[i];
            }
            if s.position[i] > mx[i] {
                mx[i] = s.position[i];
            }
        }
    }
    if chunk.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (mn, mx)
    }
}

/// Read a glTF file produced by `write_gltf` back into an IR scene.
///
/// Strict palette-sidecar semantics mirror [`read_glb`]: a missing
/// `.shpal` sidecar referenced by `CT_gaussian_splatting_palette` is a
/// hard error. Use [`read_gltf_with_opts`] (or set
/// `CATETUS_ALLOW_MISSING_PALETTE=1`) to opt back into legacy
/// DC-only degradation.
pub fn read_gltf(path: &Path) -> Result<SplatScene, GltfError> {
    read_gltf_with_opts(path, &ReadOpts::default())
}

/// Variant of [`read_gltf`] that takes a [`ReadOpts`] to opt into permissive
/// behaviour (e.g. allowing a missing `.shpal` sidecar).
pub fn read_gltf_with_opts(path: &Path, opts: &ReadOpts) -> Result<SplatScene, GltfError> {
    let raw = fs::read_to_string(path)?;
    let dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    read_gltf_str_with_opts(&raw, &dir, opts)
}

fn read_gltf_str_with_opts(
    raw: &str,
    base_dir: &Path,
    opts: &ReadOpts,
) -> Result<SplatScene, GltfError> {
    let mut root: GltfRoot = serde_json::from_str(raw)?;
    normalize_legacy_extensions(&mut root.extensions);
    normalize_legacy_extensions_used(&mut root.extensions_used);
    if !root.extensions_used.iter().any(|e| e == KHR) {
        return Err(GltfError::MissingExtension);
    }
    let prim_val = root
        .meshes
        .first()
        .and_then(|m| m.get("primitives"))
        .and_then(|p| p.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| GltfError::Malformed("no primitives".to_string()))?;
    let attrs = resolve_attrs(prim_val)?;

    // Load buffer bytes.
    let mut buffers_bytes: Vec<Vec<u8>> = Vec::with_capacity(root.buffers.len());
    for b in &root.buffers {
        let uri = b
            .uri
            .as_ref()
            .ok_or_else(|| GltfError::BufferNotFound("no uri".to_string()))?;
        let bp = base_dir.join(uri);
        let data =
            fs::read(&bp).map_err(|_| GltfError::BufferNotFound(bp.display().to_string()))?;
        buffers_bytes.push(data);
    }

    let read_attr = |acc_idx: usize, comps: usize| -> Result<Vec<f32>, GltfError> {
        let acc = &root.accessors[acc_idx];
        let bv = &root.buffer_views[acc.buffer_view];
        let data = &buffers_bytes[bv.buffer];
        let bytes = &data[bv.byte_offset..bv.byte_offset + bv.byte_length];
        decode_accessor(bytes, acc, comps)
    };
    let read_acc_raw = |acc_idx: usize| -> Result<Vec<u8>, GltfError> {
        let acc = &root.accessors[acc_idx];
        let bv = &root.buffer_views[acc.buffer_view];
        let data = &buffers_bytes[bv.buffer];
        Ok(data[bv.byte_offset..bv.byte_offset + bv.byte_length].to_vec())
    };

    // CT_gaussian_splatting_palette: mirror the GLB path — load the
    // `.shpal` sidecar (resolved relative to `base_dir`) so SH-rest
    // round-trips through `.gltf+.bin` the same way it does through `.glb`.
    // Strict by default; opt out via `ReadOpts.allow_missing_palette` or
    // `CATETUS_ALLOW_MISSING_PALETTE=1`.
    let palette: Option<ShPalette> = if let Some(pal_ext) = root.extensions.get(CT_PALETTE) {
        let uri = pal_ext.get("uri").and_then(|v| v.as_str()).ok_or_else(|| {
            GltfError::Malformed("CT_gaussian_splatting_palette: missing uri".into())
        })?;
        let p = base_dir.join(uri);
        match fs::read(&p) {
            Ok(bytes) => {
                let exp_k = pal_ext
                    .get("paletteSize")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let exp_n = pal_ext
                    .get("splatCount")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let exp_b = pal_ext
                    .get("codebookBits")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u8;
                let sh_deg = pal_ext
                    .get("shDegree")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as u8;
                Some(decode_shpal_bytes(
                    &bytes,
                    Some((exp_k, exp_n, exp_b)),
                    sh_deg,
                )?)
            }
            Err(_) if opts.allow_missing_palette_effective() => None,
            Err(_) => {
                return Err(GltfError::MissingPaletteSidecar {
                    uri: uri.to_string(),
                    tried: p.display().to_string(),
                });
            }
        }
    } else {
        None
    };

    let smallest3_bits: Option<u8> = root
        .extensions
        .get(CT_QUAT_SMALLEST3)
        .and_then(|e| e.get("componentBits"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u8);

    let log_quant_attrs = root.extensions.contains_key(CT_LOG_QUANT_ATTRS);

    let ext = DecodeExtensions {
        palette: palette.as_ref(),
        smallest3_bits,
        log_quant_attrs,
    };
    let mut scene = decode_resolved_full(&attrs, &read_attr, &ext, &read_acc_raw)?;

    // CT_v5_tail_residual: parity with the GLB reader. Hard-fail when the
    // extension is in `extensionsRequired` and the sidecar is missing
    // (unless `allow_missing_tail`); otherwise warn + render baseline.
    if let Some(tail_ext) = root.extensions.get(CT_V5_TAIL) {
        let uri = tail_ext
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GltfError::Malformed("CT_v5_tail_residual: missing uri".into()))?;
        let required = root.extensions_required.iter().any(|e| e == CT_V5_TAIL);
        let p = base_dir.join(uri);
        match fs::read(&p) {
            Ok(sidecar_bytes) => {
                let decoded = v5_tail::decode_v5tail_bytes(&sidecar_bytes).map_err(|e| {
                    GltfError::Malformed(format!("CT_v5_tail_residual decode failed: {e:#}"))
                })?;
                apply_v5tail_to_scene(&mut scene, &decoded)?;
            }
            Err(_) => {
                if required && !opts.allow_missing_tail_effective() {
                    return Err(GltfError::MissingTailSidecar {
                        uri: uri.to_string(),
                        tried: p.display().to_string(),
                    });
                }
                eprintln!(
                    "warning: CT_v5_tail_residual sidecar not found at {} \
                     (declared uri={}); rendering baseline reconstruction",
                    p.display(),
                    uri,
                );
            }
        }
    }
    Ok(scene)
}

/// Optional sidecar / extension state plumbed into [`decode_resolved`] so the
/// reader can reconstruct SOG-style palette SH-rest and smallest-3 packed
/// quaternions on top of the plain accessor decode.
#[derive(Default)]
struct DecodeExtensions<'a> {
    /// Decoded `.shpal` sidecar contents — when present, SH-rest is rebuilt
    /// from `(codebook, indices)` (overriding any `attrs.non_dc_sh`).
    palette: Option<&'a ShPalette>,
    /// When present, ROTATION is decoded by unpacking the SCALAR UINT
    /// accessor with this many bits per stored component (3 + 2-bit tag).
    smallest3_bits: Option<u8>,
    /// When `true`, the writer stored SCALE in log-space and OPACITY in
    /// logit-space. After the plain accessor decode, the reader must apply
    /// `exp` to each scale component and `sigmoid` to each opacity to
    /// recover the IR-space linear / [0,1] values. See
    /// [`CT_LOG_QUANT_ATTRS`].
    log_quant_attrs: bool,
}

/// Materialise a `SplatScene` from a `ResolvedAttrs` plus a closure that
/// decodes an accessor index into a flat `Vec<f32>`. Shared by both the
/// `.gltf` and `.glb` readers so the spec-version reconciliation stays in
/// one place.
///
/// Understands `CT_gaussian_splatting_palette` (SH-rest rebuild from a
/// decoded `.shpal` sidecar) and `CT_quat_smallest3` (UINT ROTATION
/// accessor unpacked to a 4-vector quaternion). `read_acc_raw` returns
/// a copy of an accessor's bufferView bytes — used by the smallest3 path
/// which needs the underlying u32s rather than the `decode_accessor`
/// floats.
fn decode_resolved_full<F, R>(
    attrs: &ResolvedAttrs,
    read_attr: &F,
    ext: &DecodeExtensions,
    read_acc_raw: &R,
) -> Result<SplatScene, GltfError>
where
    F: Fn(usize, usize) -> Result<Vec<f32>, GltfError>,
    R: Fn(usize) -> Result<Vec<u8>, GltfError>,
{
    let positions = read_attr(attrs.pos, 3)?;
    let mut scales = read_attr(attrs.scale, 3)?;
    let mut opacities = read_attr(attrs.opacity, 1)?;
    let dc = read_attr(attrs.dc, 3)?;
    let n = opacities.len();
    // CT_log_quant_attrs: the writer stored SCALE in log-space and
    // OPACITY in logit-space. The accessor min/max metadata reflects
    // that space, so `decode_accessor` already dequantized into the
    // transformed space — we just need to apply the inverse here to
    // recover the IR's linear scale + sigmoid opacity.
    if ext.log_quant_attrs {
        for v in &mut scales {
            *v = v.exp();
        }
        for p in &mut opacities {
            *p = 1.0 / (1.0 + (-*p).exp());
        }
    }

    // ROTATION: smallest3 wins, otherwise fall through to the accessor path
    // (FLOAT / UBYTE-normalized / SHORT-normalized are all handled by
    // `decode_accessor`).
    let rotations: Vec<f32> = if let Some(bits) = ext.smallest3_bits {
        let raw = read_acc_raw(attrs.rot)?;
        unpack_smallest3_rotations(&raw, n, bits)?
    } else {
        read_attr(attrs.rot, 4)?
    };

    // Flatten SH back into the 45-scalar interleaved layout that `Color::Sh`
    // expects (3 DC + 45 trailing = 48 floats). Palette wins over accessor
    // SH-rest — the writer skips per-coefficient accessors entirely when
    // palette elision is on, so the only way to recover SH-rest is via the
    // `.shpal` codebook + indices.
    let mut sh_flat: Option<Vec<f32>> = None;
    if let Some(pal) = ext.palette {
        let cov_scalars = shpal_non_dc_scalars_for_degree(pal.sh_degree.max(1));
        if cov_scalars == 0 {
            // Palette declared sh_degree=0 — leave SH-rest absent; the
            // downstream consumer falls back to `Color::Rgb`.
        } else {
            if pal.indices.len() < n {
                return Err(GltfError::Malformed(format!(
                    "CT_gaussian_splatting_palette: indices {} < splat count {}",
                    pal.indices.len(),
                    n
                )));
            }
            // Codebook always stores 45 scalars per centroid (degrees 1+2+3).
            // We copy a contiguous prefix of `cov_scalars` so a sh<3 palette
            // still works without buffer over-runs.
            let mut flat = vec![0.0f32; n * NON_DC_SH_SCALARS];
            for i in 0..n {
                let idx = pal.indices[i] as usize;
                if idx >= pal.k {
                    return Err(GltfError::Malformed(format!(
                        "CT_gaussian_splatting_palette: index {idx} >= K {}",
                        pal.k
                    )));
                }
                let src = &pal.codebook[idx * SHPAL_VQ_DIM..idx * SHPAL_VQ_DIM + cov_scalars];
                let dst_base = i * NON_DC_SH_SCALARS;
                flat[dst_base..dst_base + cov_scalars].copy_from_slice(src);
            }
            sh_flat = Some(flat);
        }
    } else if !attrs.non_dc_sh.is_empty() {
        let mut flat = vec![0.0f32; n * NON_DC_SH_SCALARS];
        match attrs.spec {
            SpecVersion::RcMay2026 => {
                // attrs.non_dc_sh: one accessor per (l, n) coefficient.
                for (coef_idx, &acc_idx) in attrs.non_dc_sh.iter().enumerate() {
                    let v = read_attr(acc_idx, 3)?;
                    for i in 0..n {
                        let dst = i * NON_DC_SH_SCALARS + coef_idx * 3;
                        let src = i * 3;
                        if dst + 3 <= flat.len() && src + 3 <= v.len() {
                            flat[dst..dst + 3].copy_from_slice(&v[src..src + 3]);
                        }
                    }
                }
            }
            SpecVersion::Pre2026 => {
                // Single SCALAR-of-45 accessor.
                let v = read_attr(attrs.non_dc_sh[0], NON_DC_SH_SCALARS)?;
                let len = flat.len();
                if v.len() >= len {
                    flat.copy_from_slice(&v[..len]);
                }
            }
        }
        sh_flat = Some(flat);
    }

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let color = if let Some(ref sh) = sh_flat {
            let mut coeffs = Vec::with_capacity(48);
            coeffs.extend_from_slice(&dc[i * 3..i * 3 + 3]);
            coeffs.extend_from_slice(&sh[i * NON_DC_SH_SCALARS..(i + 1) * NON_DC_SH_SCALARS]);
            Color::Sh { degree: 3, coeffs }
        } else {
            Color::Rgb([dc[i * 3], dc[i * 3 + 1], dc[i * 3 + 2]])
        };
        splats.push(Splat {
            position: [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]],
            rotation: [
                rotations[i * 4],
                rotations[i * 4 + 1],
                rotations[i * 4 + 2],
                rotations[i * 4 + 3],
            ],
            scale: [scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]],
            opacity: opacities[i],
            color,
        });
    }

    Ok(SplatScene {
        splats,
        coordinate_system: CoordinateSystem::default(),
        semantic_labels: None,
        temporal_mode: TemporalMode::Static,
        lods: None,
        codecgs: None,
    })
}

/// Inspect a glTF file: verify extension presence, splat count, and chunk
/// checksums when the spatial-streaming-index extension is present.
pub fn inspect_gltf(path: &Path) -> Result<InspectReport, GltfError> {
    let raw = fs::read_to_string(path)?;
    let dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let used: Vec<String> = value
        .get("extensionsUsed")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let has_khr = used.iter().any(|e| e == KHR);
    if !has_khr {
        return Err(GltfError::MissingExtension);
    }

    // Optional KHR version check on extension blob — anything other than absent or matching version 1.
    if let Some(ext) = value
        .get("meshes")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .and_then(|m| m.get("primitives"))
        .and_then(|p| p.as_array())
        .and_then(|p| p.first())
        .and_then(|p| p.get("extensions"))
        .and_then(|e| e.get(KHR))
    {
        if let Some(ver) = ext.get("version").and_then(|v| v.as_str()) {
            if ver != "1" {
                return Err(GltfError::UnsupportedVersion(ver.to_string()));
            }
        }
    }

    let has_spatial_index = used.iter().any(|e| e == CT_INDEX);
    let mut chunk_count = 0usize;
    // Checksum mismatches bail out with an early `Err` rather than flagging
    // here, so the report just records the success state.
    let checksum_ok = true;
    let mut splat_count = 0usize;
    if has_spatial_index {
        let chunks = value
            .get("extensions")
            .and_then(|e| e.get(CT_INDEX))
            .and_then(|e| e.get("chunks"))
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        chunk_count = chunks.len();
        // Load buffers list for chunk byte resolution.
        let buffers = value
            .get("buffers")
            .and_then(|b| b.as_array())
            .cloned()
            .unwrap_or_default();
        for (i, chunk) in chunks.iter().enumerate() {
            let buffer_idx = chunk
                .get("buffer")
                .and_then(|b| b.as_u64())
                .ok_or_else(|| GltfError::Malformed("chunk missing buffer".to_string()))?
                as usize;
            let byte_offset = chunk
                .get("byteOffset")
                .and_then(|b| b.as_u64())
                .unwrap_or(0) as usize;
            let byte_length = chunk
                .get("byteLength")
                .and_then(|b| b.as_u64())
                .unwrap_or(0) as usize;
            let expected = chunk.get("checksum").and_then(|c| c.as_str()).unwrap_or("");
            let buf = buffers.get(buffer_idx).ok_or_else(|| {
                GltfError::Malformed(format!("chunk {i} references missing buffer"))
            })?;
            let uri = buf
                .get("uri")
                .and_then(|u| u.as_str())
                .ok_or_else(|| GltfError::Malformed("buffer has no uri".to_string()))?;
            let data =
                fs::read(dir.join(uri)).map_err(|_| GltfError::BufferNotFound(uri.to_string()))?;
            let slice = &data[byte_offset..byte_offset + byte_length];
            let actual = format!("blake3:{}", blake3::hash(slice).to_hex());
            if actual != expected {
                // Function returns before `checksum_ok` could ever be read,
                // so just propagate the error.
                return Err(GltfError::ChecksumMismatch(i));
            }
            splat_count += chunk
                .get("splatCount")
                .and_then(|s| s.as_u64())
                .unwrap_or(0) as usize;
        }
    } else {
        // Walk accessors to find OPACITY count as splat count. Honour both
        // the RC (namespaced, attributes-on-primitive) and pre-RC (underscore,
        // attributes-in-extension) layouts.
        if let Some(prim) = value
            .get("meshes")
            .and_then(|m| m.as_array())
            .and_then(|m| m.first())
            .and_then(|m| m.get("primitives"))
            .and_then(|p| p.as_array())
            .and_then(|p| p.first())
        {
            let prim_attrs = prim.get("attributes");
            let ext_attrs = prim
                .get("extensions")
                .and_then(|e| e.get(KHR))
                .and_then(|e| e.get("attributes"));
            let candidates: [(Option<&serde_json::Value>, &str); 2] = [
                (prim_attrs, "KHR_gaussian_splatting:OPACITY"),
                (ext_attrs, "_OPACITY"),
            ];
            for (attrs, key) in candidates {
                let Some(attrs) = attrs else { continue };
                if let Some(idx) = attrs.get(key).and_then(|v| v.as_u64()) {
                    if let Some(acc) = value
                        .get("accessors")
                        .and_then(|a| a.as_array())
                        .and_then(|a| a.get(idx as usize))
                    {
                        splat_count =
                            acc.get("count").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
                        break;
                    }
                }
            }
        }
    }

    Ok(InspectReport {
        has_khr,
        has_spatial_index,
        chunk_count,
        checksum_ok,
        splat_count,
    })
}

/// Convenience: resolve `scene.gltf` path inside a directory.
pub fn default_gltf_path(dir: &Path) -> PathBuf {
    dir.join("scene.gltf")
}

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const GLB_JSON: u32 = 0x4E4F_534A; // "JSON"
const GLB_BIN: u32 = 0x004E_4942; // "BIN\0"

/// Build the JSON document plus a single in-memory binary blob for GLB output.
/// All buffer URIs are dropped — the resulting glTF references buffer 0 with no
/// URI, which is the GLB-embedded buffer.
fn build_single_buffer_gltf(
    scene: &SplatScene,
    opts: &WriteOpts,
) -> Result<(GltfRoot, Vec<u8>), GltfError> {
    let chunk: &[Splat] = scene.splats.as_slice();
    let (chunk_min, chunk_max) = chunk_bbox(chunk);
    let log_quant = opts.quantize && opts.log_quant_attrs;
    let (scale_min, scale_max) = chunk_scale_bbox_in(chunk, log_quant);
    let (opacity_min, opacity_max) = chunk_opacity_bbox(chunk, log_quant);
    let sh_degree: u8 = chunk.iter().map(|s| s.color.degree()).max().unwrap_or(0);
    let layout = ChunkLayout {
        spec: opts.spec_version,
        quantize: opts.quantize,
        quantize_rotation: opts.quantize_rotation,
        sh_degree,
        sh_rest_quant: opts.sh_rest_quant.as_ref(),
        rotation_smallest3: opts.rotation_smallest3.as_ref(),
        rotation_quant: opts.rotation_quant.as_ref(),
        dc_quant: opts.dc_quant.as_ref(),
        palette_elision: opts.palette.is_some(),
        log_quant_attrs: opts.log_quant_attrs,
    };
    let buf_bytes = pack_chunk_with(chunk, &layout, &chunk_min, &chunk_max);

    let mut extensions_used = vec![KHR.to_string()];
    if opts.quantize {
        extensions_used.push(KHR_QUANT.to_string());
    }
    if log_quant {
        extensions_used.push(CT_LOG_QUANT_ATTRS.to_string());
    }
    let mut root = GltfRoot {
        asset: GltfAsset {
            version: "2.0".to_string(),
            generator: Some("catetus-gltf".to_string()),
        },
        extensions_used,
        extensions_required: vec![KHR.to_string()],
        buffers: vec![GltfBuffer {
            byte_length: buf_bytes.len(),
            uri: None,
        }],
        buffer_views: Vec::new(),
        accessors: Vec::new(),
        meshes: Vec::new(),
        extensions: serde_json::Map::new(),
    };

    let n = chunk.len();
    let accs = emit_chunk_accessors(
        &mut root,
        0,
        n,
        &chunk_min,
        &chunk_max,
        &scale_min,
        &scale_max,
        &opacity_min,
        &opacity_max,
        &layout,
    );

    root.meshes.push(serde_json::json!({
        "primitives": [build_primitive(opts.spec_version, &accs)]
    }));

    let scene_bbox = if chunk.is_empty() {
        ([0.0f32; 3], [0.0f32; 3])
    } else {
        (chunk_min, chunk_max)
    };
    root.extensions.insert(
        KHR.to_string(),
        serde_json::json!({
            "splatCount": n,
            "bbox": { "min": scene_bbox.0, "max": scene_bbox.1 },
            "shDegree": sh_degree,
        }),
    );
    if log_quant {
        root.extensions.insert(
            CT_LOG_QUANT_ATTRS.to_string(),
            serde_json::json!({ "scale": "ln", "opacity": "logit" }),
        );
    }

    // Palette elision: advertise the `.shpal` sidecar so palette-aware
    // decoders can rebuild SH-rest from `(codebook, indices)` even though
    // the GLB itself carries no `SH_DEGREE_l_COEF_n` accessors. We mark
    // the extension as `extensionsUsed` only (NOT `extensionsRequired`) so
    // legacy decoders ignore it and simply render DC-only — the asset still
    // loads, just without view-dependent SH. This is the same opt-in shape
    // SOG uses for its `meta.json` sidecar pointer.
    if let Some(pal) = &opts.palette {
        if !root.extensions_used.iter().any(|e| e == CT_PALETTE) {
            root.extensions_used.push(CT_PALETTE.to_string());
        }
        root.extensions.insert(
            CT_PALETTE.to_string(),
            serde_json::json!({
                "uri": pal.sidecar_uri,
                "paletteSize": pal.palette_size,
                "splatCount": pal.n_splats,
                "codebookBits": pal.codebook_bits,
                "shDegree": pal.sh_degree,
                "indexComponentType": USHORT,
                "format": "shpa-v1",
            }),
        );
    }

    // V5.2 joint-tail residual sidecar pointer (single-buffer GLB path;
    // same shape as the chunked path above).
    if let Some(tail) = &opts.v5_tail {
        if !root.extensions_used.iter().any(|e| e == CT_V5_TAIL) {
            root.extensions_used.push(CT_V5_TAIL.to_string());
        }
        if tail.required && !root.extensions_required.iter().any(|e| e == CT_V5_TAIL) {
            root.extensions_required.push(CT_V5_TAIL.to_string());
        }
        root.extensions.insert(
            CT_V5_TAIL.to_string(),
            serde_json::json!({
                "uri": tail.sidecar_uri,
                "splatCount": tail.n_splats,
                "kSelected": tail.k_selected,
                "shRestCoefs": tail.sh_rest_coefs,
                "nCells": tail.n_cells,
                "format": "sfv51tal-v2",
            }),
        );
    }

    // Smallest-3 quaternion marker (single-buffer GLB path; same shape as
    // the chunked path above).
    if let Some(s3) = &opts.rotation_smallest3 {
        if !root.extensions_used.iter().any(|e| e == CT_QUAT_SMALLEST3) {
            root.extensions_used.push(CT_QUAT_SMALLEST3.to_string());
        }
        root.extensions.insert(
            CT_QUAT_SMALLEST3.to_string(),
            serde_json::json!({
                "componentBits": s3.component_bits,
                "componentType": UINT,
                "layout": "q0|q1|q2|tag",
                "tagBits": 2,
            }),
        );
    }

    Ok((root, buf_bytes))
}

/// Brotli-compress `input` at the given quality level. Uses a 16 MiB window
/// (`lgwin = 24`) which beats the default 22 by ~1% on the large FP32 SH
/// payloads we care about — the BIN chunk is the only payload here so there's
/// no shared-context concern.
fn brotli_compress(input: &[u8], quality: i32) -> Result<Vec<u8>, GltfError> {
    let mut out: Vec<u8> = Vec::with_capacity(input.len() / 2 + 64);
    let mut params = brotli::enc::BrotliEncoderParams::default();
    params.quality = quality;
    params.lgwin = 24;
    let mut reader = std::io::Cursor::new(input);
    brotli::BrotliCompress(&mut reader, &mut out, &params)
        .map_err(|e| GltfError::Brotli(format!("encode failed: {e:?}")))?;
    Ok(out)
}

/// Brotli-decompress `input` into a fresh `Vec<u8>`.
fn brotli_decompress(input: &[u8], expected_len: Option<usize>) -> Result<Vec<u8>, GltfError> {
    let cap = expected_len.unwrap_or_else(|| input.len() * 4);
    let mut out: Vec<u8> = Vec::with_capacity(cap);
    let mut reader = std::io::Cursor::new(input);
    brotli::BrotliDecompress(&mut reader, &mut out)
        .map_err(|e| GltfError::Brotli(format!("decode failed: {e:?}")))?;
    if let Some(exp) = expected_len {
        if out.len() != exp {
            return Err(GltfError::Brotli(format!(
                "decoded length {} != expected {}",
                out.len(),
                exp
            )));
        }
    }
    Ok(out)
}

/// Compress `bin` with brotli at `quality` and rewrite `root` so loaders that
/// understand `CT_brotli_buffer` can decompress it. The wrapper metadata
/// records the uncompressed byte length so the decoder can size its output
/// buffer once.
///
/// `buffers[0].byteLength` is left pointing at the *uncompressed* length and
/// every accessor / bufferView keeps its original offsets — this matches the
/// validator's "byteLength is the logical size" contract and the reader only
/// has to consult the wrapper to translate between the two coordinate spaces.
fn apply_brotli_wrap(
    root: &mut GltfRoot,
    bin: Vec<u8>,
    wrap: LosslessWrap,
) -> Result<Vec<u8>, GltfError> {
    let quality = match wrap {
        LosslessWrap::Brotli11 => 11,
        // `apply_brotli_wrap` is only ever dispatched from the `Brotli11`
        // arm in `write_glb`; this is a defensive default in case a future
        // caller passes a non-brotli variant in.
        LosslessWrap::Zstd19Split => {
            return Err(GltfError::Brotli(
                "Zstd19Split is not a brotli wrap; route through apply_zstd_split_wrap".to_string(),
            ))
        }
    };
    let uncompressed_len = bin.len();
    let compressed = brotli_compress(&bin, quality)?;
    if !root.extensions_used.iter().any(|e| e == CT_BROTLI) {
        root.extensions_used.push(CT_BROTLI.to_string());
    }
    // `CT_brotli_buffer` is REQUIRED — readers that don't understand it would
    // see a compressed BIN and silently miss-read every accessor. Better to
    // hard-fail in those readers than corrupt the scene.
    if !root.extensions_required.iter().any(|e| e == CT_BROTLI) {
        root.extensions_required.push(CT_BROTLI.to_string());
    }
    root.extensions.insert(
        CT_BROTLI.to_string(),
        serde_json::json!({
            "buffer": 0usize,
            "uncompressedByteLength": uncompressed_len,
            "compressedByteLength": compressed.len(),
            "quality": quality,
        }),
    );
    Ok(compressed)
}

/// Compute the per-splat byte stride for each bufferView in `root`, derived
/// from any accessor pointing at it (stride = bv.byte_length / accessor.count).
/// Views with no accessor — or with multiple disagreeing accessors — fall
/// back to a stride of 1 (which makes the transpose a no-op for that view).
/// Only views with `buffer == 0` and non-zero length are considered eligible
/// for the split wrap; others are returned with `stride = 1` so the wrapper
/// degrades to plain whole-BIN zstd for them.
fn derive_view_strides(root: &GltfRoot) -> Vec<usize> {
    let mut strides = vec![0usize; root.buffer_views.len()];
    for acc in &root.accessors {
        if acc.buffer_view >= strides.len() || acc.count == 0 {
            continue;
        }
        let bv = &root.buffer_views[acc.buffer_view];
        if bv.byte_length == 0 || bv.byte_length % acc.count != 0 {
            continue;
        }
        let s = bv.byte_length / acc.count;
        // First accessor wins; if a later accessor disagrees, fall back to 1.
        if strides[acc.buffer_view] == 0 {
            strides[acc.buffer_view] = s;
        } else if strides[acc.buffer_view] != s {
            strides[acc.buffer_view] = 1;
        }
    }
    for s in strides.iter_mut() {
        if *s == 0 {
            *s = 1;
        }
    }
    strides
}

/// Transpose `src` from interleaved `[count][stride]` to byte-plane
/// `[stride][count]` layout. `src.len()` must equal `count * stride`. When
/// `stride <= 1` this is a no-op copy.
fn transpose_split(src: &[u8], stride: usize) -> Vec<u8> {
    let n = src.len();
    if stride <= 1 || n == 0 {
        return src.to_vec();
    }
    debug_assert_eq!(n % stride, 0, "transpose_split: len not multiple of stride");
    let count = n / stride;
    let mut out = vec![0u8; n];
    // out[b * count + i] = src[i * stride + b]
    for b in 0..stride {
        let dst_base = b * count;
        for i in 0..count {
            out[dst_base + i] = src[i * stride + b];
        }
    }
    out
}

/// Inverse of `transpose_split`. Reads `[stride][count]` and writes
/// `[count][stride]`. `src.len()` must equal `count * stride`.
fn transpose_unsplit(src: &[u8], stride: usize) -> Vec<u8> {
    let n = src.len();
    if stride <= 1 || n == 0 {
        return src.to_vec();
    }
    debug_assert_eq!(
        n % stride,
        0,
        "transpose_unsplit: len not multiple of stride"
    );
    let count = n / stride;
    let mut out = vec![0u8; n];
    // out[i * stride + b] = src[b * count + i]
    for b in 0..stride {
        let src_base = b * count;
        for i in 0..count {
            out[i * stride + b] = src[src_base + i];
        }
    }
    out
}

/// Apply the `Zstd19Split` lossless wrap. Each bufferView in `root` that
/// targets buffer 0 is compressed as its own zstd-19 frame; the encoder picks
/// the smaller of the byte-plane-transposed (`stride` from the accessor) and
/// the interleaved variants. Per-view frames are concatenated and the
/// manifest records `{origOffset, origLength, stride, splitApplied,
/// compOffset, compLength}` so the decoder can reverse exactly. Bytes
/// outside any view (alignment padding) are left zero on decode — they don't
/// participate in any accessor read.
fn apply_zstd_split_wrap(root: &mut GltfRoot, bin: Vec<u8>) -> Result<Vec<u8>, GltfError> {
    let uncompressed_len = bin.len();
    let strides = derive_view_strides(root);
    let mut compressed_concat: Vec<u8> = Vec::with_capacity(bin.len() / 2);
    let mut manifest: Vec<serde_json::Value> = Vec::with_capacity(root.buffer_views.len());
    for (idx, bv) in root.buffer_views.iter().enumerate() {
        if bv.buffer != 0 || bv.byte_length == 0 {
            manifest.push(serde_json::json!({
                "origOffset": bv.byte_offset,
                "origLength": bv.byte_length,
                "stride": 1,
                "splitApplied": false,
                "compOffset": compressed_concat.len(),
                "compLength": 0,
            }));
            continue;
        }
        let stride = strides[idx];
        let end = bv.byte_offset.saturating_add(bv.byte_length);
        if end > bin.len() {
            return Err(GltfError::Brotli(format!(
                "bufferView {idx} exceeds BIN length"
            )));
        }
        let src = &bin[bv.byte_offset..end];
        // Always compress the plain stream. Also try the split-transpose
        // when stride > 1 and the length divides cleanly; pick the smaller.
        let plain_comp = zstd::bulk::compress(src, 19)
            .map_err(|e| GltfError::Brotli(format!("zstd encode failed (plain): {e:?}")))?;
        let (chosen, applied) = if stride > 1 && bv.byte_length % stride == 0 {
            let split = transpose_split(src, stride);
            let split_comp = zstd::bulk::compress(&split, 19)
                .map_err(|e| GltfError::Brotli(format!("zstd encode failed (split): {e:?}")))?;
            if split_comp.len() < plain_comp.len() {
                (split_comp, true)
            } else {
                (plain_comp, false)
            }
        } else {
            (plain_comp, false)
        };
        let comp_offset = compressed_concat.len();
        let comp_length = chosen.len();
        compressed_concat.extend_from_slice(&chosen);
        manifest.push(serde_json::json!({
            "origOffset": bv.byte_offset,
            "origLength": bv.byte_length,
            "stride": stride,
            "splitApplied": applied,
            "compOffset": comp_offset,
            "compLength": comp_length,
        }));
    }
    if !root.extensions_used.iter().any(|e| e == CT_ZSTD_SPLIT) {
        root.extensions_used.push(CT_ZSTD_SPLIT.to_string());
    }
    if !root.extensions_required.iter().any(|e| e == CT_ZSTD_SPLIT) {
        root.extensions_required.push(CT_ZSTD_SPLIT.to_string());
    }
    root.extensions.insert(
        CT_ZSTD_SPLIT.to_string(),
        serde_json::json!({
            "buffer": 0usize,
            "uncompressedByteLength": uncompressed_len,
            "compressedByteLength": compressed_concat.len(),
            "level": 19,
            "perViewFrames": true,
            "views": manifest,
        }),
    );
    Ok(compressed_concat)
}

/// Inverse of `apply_zstd_split_wrap`. Returns `Some(bin)` when
/// `CT_zstd_split_buffer` is present on `root`, `None` otherwise.
fn unwrap_zstd_split(root: &GltfRoot, bin: &[u8]) -> Result<Option<Vec<u8>>, GltfError> {
    let Some(ext) = root.extensions.get(CT_ZSTD_SPLIT) else {
        return Ok(None);
    };
    let expected = ext
        .get("uncompressedByteLength")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            GltfError::Malformed("CT_zstd_split_buffer missing uncompressedByteLength".into())
        })? as usize;
    let views = ext
        .get("views")
        .and_then(|v| v.as_array())
        .ok_or_else(|| GltfError::Malformed("CT_zstd_split_buffer missing views".into()))?;
    let mut out = vec![0u8; expected];
    for view in views {
        let orig_offset = view.get("origOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let orig_length = view.get("origLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let stride = view.get("stride").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let split_applied = view
            .get("splitApplied")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let comp_offset = view.get("compOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let comp_length = view.get("compLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if orig_length == 0 {
            continue;
        }
        let comp_end = comp_offset.saturating_add(comp_length);
        if comp_end > bin.len() {
            return Err(GltfError::Brotli(format!(
                "CT_zstd_split_buffer view comp range {comp_offset}..{comp_end} exceeds compressed BIN"
            )));
        }
        let frame = &bin[comp_offset..comp_end];
        let decoded = zstd::bulk::decompress(frame, orig_length)
            .map_err(|e| GltfError::Brotli(format!("zstd decode failed: {e:?}")))?;
        if decoded.len() != orig_length {
            return Err(GltfError::Brotli(format!(
                "decoded length {} != expected {}",
                decoded.len(),
                orig_length
            )));
        }
        let final_bytes = if split_applied && stride > 1 {
            transpose_unsplit(&decoded, stride)
        } else {
            decoded
        };
        let dst_end = orig_offset.saturating_add(orig_length);
        if dst_end > out.len() {
            return Err(GltfError::Brotli(format!(
                "view orig range {orig_offset}..{dst_end} exceeds uncompressed BIN"
            )));
        }
        out[orig_offset..dst_end].copy_from_slice(&final_bytes);
    }
    Ok(Some(out))
}

/// Inverse of `apply_brotli_wrap`. Returns the decompressed BIN bytes when
/// `CT_brotli_buffer` is present on `root`, or `None` otherwise.
fn unwrap_brotli(root: &GltfRoot, bin: &[u8]) -> Result<Option<Vec<u8>>, GltfError> {
    let Some(ext) = root.extensions.get(CT_BROTLI) else {
        return Ok(None);
    };
    let expected = ext
        .get("uncompressedByteLength")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let decoded = brotli_decompress(bin, expected)?;
    Ok(Some(decoded))
}

/// Decoded `.shpal` sidecar: 45-D k-means SH-rest palette codebook plus per-
/// splat 16-bit palette indices. Mirrors the JS reference decoder in
/// `experiments/w3-fidelity-harness/code/cpu-fidelity.mjs::loadShPaletteSidecar`
/// and the writer in
/// `catetus-optimize/src/vq_palette.rs::ShRestPaletteSidetable`.
#[derive(Debug, Clone)]
pub struct ShPalette {
    /// Codebook entry count (palette size).
    pub k: usize,
    /// Splat count (matches the GLB's POSITION accessor.count).
    pub n: usize,
    /// Quantization bit-width: `<=8` ⇒ codebook stored as i8, otherwise i16.
    pub codebook_bits: u8,
    /// Per-coefficient dequant ranges (length 45).
    pub ranges: Vec<f32>,
    /// Codebook entries, row-major: `codebook[c * 45 + d]` (length `k * 45`).
    pub codebook: Vec<f32>,
    /// Per-splat palette indices (length `n`).
    pub indices: Vec<u16>,
    /// SH degree the palette covers (1, 2, or 3); 0 when unknown.
    pub sh_degree: u8,
}

/// Number of non-DC SH coefficients (`d = 3 + 5 + 7 = 15`), times 3 channels.
const SHPAL_VQ_DIM: usize = 45;

/// Decode a `.shpal` sidecar (zstd-19 compressed) into an [`ShPalette`].
///
/// Wire format (matches the JS reference and the Rust writer side):
/// `magic "SHPA" (u32 LE = 0x53485041) | version u32 LE | k u32 LE | n u32 LE |
/// codebookBits u8 | _pad[3] | ranges f32×45 | codebook i8[k*45] (when
/// codebookBits<=8) or i16×k*45 (otherwise) | indices u16×n`.
///
/// `expected` (when provided) lets callers cross-check against the
/// `CT_gaussian_splatting_palette` metadata on the GLB.
pub fn decode_shpal_bytes(
    compressed: &[u8],
    expected: Option<(usize, usize, u8)>, // (palette_size, splat_count, codebook_bits)
    sh_degree: u8,
) -> Result<ShPalette, GltfError> {
    // The sidecar is a single zstd frame. We don't know the uncompressed size
    // up front; pass a generous capacity (the writer caps at <100 MB).
    let decoded = zstd::bulk::decompress(compressed, 256 * 1024 * 1024)
        .map_err(|e| GltfError::Brotli(format!(".shpal zstd decode failed: {e:?}")))?;
    if decoded.len() < 20 {
        return Err(GltfError::Malformed(format!(
            ".shpal too small: {} bytes",
            decoded.len()
        )));
    }
    let magic = u32::from_le_bytes([decoded[0], decoded[1], decoded[2], decoded[3]]);
    if magic != 0x5348_5041 {
        return Err(GltfError::Malformed(format!(
            ".shpal magic mismatch: 0x{magic:08x}"
        )));
    }
    let version = u32::from_le_bytes([decoded[4], decoded[5], decoded[6], decoded[7]]);
    if version != 1 {
        return Err(GltfError::Malformed(format!(
            "unsupported .shpal version: {version}"
        )));
    }
    let k = u32::from_le_bytes([decoded[8], decoded[9], decoded[10], decoded[11]]) as usize;
    let n = u32::from_le_bytes([decoded[12], decoded[13], decoded[14], decoded[15]]) as usize;
    let codebook_bits = decoded[16];
    if let Some((exp_k, exp_n, exp_b)) = expected {
        if exp_k != 0 && exp_k != k {
            return Err(GltfError::Malformed(format!(
                ".shpal paletteSize mismatch: ext={exp_k} sidecar={k}"
            )));
        }
        if exp_n != 0 && exp_n != n {
            return Err(GltfError::Malformed(format!(
                ".shpal splatCount mismatch: ext={exp_n} sidecar={n}"
            )));
        }
        if exp_b != 0 && exp_b != codebook_bits {
            return Err(GltfError::Malformed(format!(
                ".shpal codebookBits mismatch: ext={exp_b} sidecar={codebook_bits}"
            )));
        }
    }
    // 16 header + 3 pad bytes (after the u8) + 1 leading-byte padding handled
    // by the writer's struct alignment. The JS reference reads from offset 20.
    let mut off: usize = 20;
    if decoded.len() < off + SHPAL_VQ_DIM * 4 {
        return Err(GltfError::Malformed(".shpal truncated in ranges".into()));
    }
    let mut ranges = vec![0.0f32; SHPAL_VQ_DIM];
    for r in ranges.iter_mut() {
        *r = f32::from_le_bytes([
            decoded[off],
            decoded[off + 1],
            decoded[off + 2],
            decoded[off + 3],
        ]);
        off += 4;
    }
    let mut codebook = vec![0.0f32; k * SHPAL_VQ_DIM];
    if codebook_bits <= 8 {
        let levels = 127.0f32;
        let needed = k * SHPAL_VQ_DIM;
        if decoded.len() < off + needed {
            return Err(GltfError::Malformed(
                ".shpal truncated in i8 codebook".into(),
            ));
        }
        for c in 0..k {
            for d in 0..SHPAL_VQ_DIM {
                let q = decoded[off] as i8;
                off += 1;
                codebook[c * SHPAL_VQ_DIM + d] = (q as f32 / levels) * ranges[d];
            }
        }
    } else {
        let levels = 32767.0f32;
        let needed = k * SHPAL_VQ_DIM * 2;
        if decoded.len() < off + needed {
            return Err(GltfError::Malformed(
                ".shpal truncated in i16 codebook".into(),
            ));
        }
        for c in 0..k {
            for d in 0..SHPAL_VQ_DIM {
                let q = i16::from_le_bytes([decoded[off], decoded[off + 1]]);
                off += 2;
                codebook[c * SHPAL_VQ_DIM + d] = (q as f32 / levels) * ranges[d];
            }
        }
    }
    if decoded.len() < off + n * 2 {
        return Err(GltfError::Malformed(".shpal truncated in indices".into()));
    }
    let mut indices = vec![0u16; n];
    for idx in indices.iter_mut() {
        *idx = u16::from_le_bytes([decoded[off], decoded[off + 1]]);
        off += 2;
    }
    Ok(ShPalette {
        k,
        n,
        codebook_bits,
        ranges,
        codebook,
        indices,
        sh_degree,
    })
}

/// Apply a decoded V5.2 joint-tail sidecar to a baseline `SplatScene` by
/// adding the per-attribute residuals to the selected splats. Mutates
/// `scene.splats[sel_idx[k]]` for every `k` in `0..K`.
///
/// **Coordinate space.** The V5.2 codec stores residuals in **raw 3DGS-PLY
/// space**: `scale` field is `ln(linear_scale)`, `opacity` field is
/// `logit(probability)`, `rotation` is the raw (un-normalised) quat,
/// `position` / `dc` / `sh_rest` are linear / raw. So the apply path must
/// (a) convert the in-memory IR splat back to raw PLY space, (b) add the
/// residual, (c) re-apply `exp` / `sigmoid` to land back in IR space.
/// Without this round-trip the IR-space addition silently collapses the
/// scale and opacity residuals to near-zero (because IR `scale = exp(raw)`
/// is on the order of 1e-5 for typical splats — quant-12 of that range
/// gives a residual ~1e-7, far below the bench's perceptual budget).
///
/// SH-rest layout: the Python prototype's `sh_rest[n, coef, channel]` is
/// row-major (coef outer, channel inner). `Color::Sh::coeffs[3..3+45]` is
/// laid out as a flat 45 floats — the PLY reader copies `f_rest_0..44`
/// in declaration order, so the flat indices line up. We treat the
/// residual buffer as a flat slice of length `sh_rest_coefs * 3` per
/// splat and add component-wise.
///
/// Returns the number of splats actually modified (== K when no `sel_idx`
/// entry overruns the scene).
pub fn apply_v5tail_to_scene(
    scene: &mut SplatScene,
    decoded: &v5_tail::DecodedSidecar,
) -> Result<usize, GltfError> {
    let n = scene.splats.len();
    if decoded.header.n_splats as usize != n {
        return Err(GltfError::Malformed(format!(
            "CT_v5_tail_residual: sidecar n_splats={} != scene splat count={}",
            decoded.header.n_splats, n,
        )));
    }
    let sh_rest_coefs = decoded.header.sh_rest_coefs as usize;
    let shr_chan = sh_rest_coefs * 3;
    let mut modified = 0usize;
    for (k, &splat_idx) in decoded.sel_idx.iter().enumerate() {
        let i = splat_idx as usize;
        if i >= n {
            return Err(GltfError::Malformed(format!(
                "CT_v5_tail_residual: sel_idx[{}] = {} out of range [0, {})",
                k, i, n,
            )));
        }
        let s = &mut scene.splats[i];
        // pos (3) — already raw / linear in both IR and PLY space.
        for c in 0..3 {
            s.position[c] += decoded.pos[k * 3 + c];
        }
        // rot (4) — raw additive (already same in IR and raw PLY space;
        // `read_ply` normalises the quat but the V5.2 prototype adds the
        // un-normalised PLY-space residual, so we add as-is here too).
        for c in 0..4 {
            s.rotation[c] += decoded.rot[k * 4 + c];
        }
        // opa (1) — logit-space residual. Round-trip through raw PLY space.
        s.opacity = sigmoid_clamped(logit_clamped(s.opacity) + decoded.opa[k]);
        // sca (3) — log-space residual. Round-trip through raw PLY space.
        for c in 0..3 {
            let raw = ln_clamped(s.scale[c]);
            s.scale[c] = (raw + decoded.sca[k * 3 + c]).exp();
        }
        // dc (3) + shr (sh_rest_coefs * 3) — both linear / raw additive.
        match &mut s.color {
            Color::Rgb(rgb) => {
                for c in 0..3 {
                    rgb[c] += decoded.dc[k * 3 + c];
                }
                if shr_chan > 0 {
                    eprintln!(
                        "warning: CT_v5_tail_residual skipping SH-rest residual for \
                         RGB-only splat {} (color has no SH coefficients)",
                        i
                    );
                }
            }
            Color::Sh { coeffs, .. } => {
                for c in 0..3 {
                    coeffs[c] += decoded.dc[k * 3 + c];
                }
                let avail = coeffs.len().saturating_sub(3);
                let take = avail.min(shr_chan);
                for c in 0..take {
                    coeffs[3 + c] += decoded.shr[k * shr_chan + c];
                }
            }
        }
        modified += 1;
    }
    Ok(modified)
}

/// Inverse of `read_ply`'s `sigmoid` on opacity. Clamped to (1e-7, 1-1e-7).
#[inline]
fn logit_clamped(p: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln()
}

/// Forward sigmoid used to re-IR-ify opacity after the residual is applied.
#[inline]
fn sigmoid_clamped(x: f32) -> f32 {
    // Avoid overflow in `exp(-x)` for very large positive `x`.
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Inverse of `read_ply`'s `exp` on scale. Clamped to a positive minimum so
/// genuinely-tiny 3DGS scales (e.g. `exp(-19)`) survive the round trip.
#[inline]
fn ln_clamped(s: f32) -> f32 {
    s.max(f32::MIN_POSITIVE).ln()
}

/// Number of non-DC SH scalars covered at a given degree: `Σ (2l+1) * 3` for
/// `l = 1..=degree`. Mirrors `SH_COEFS_PER_DEGREE`.
fn shpal_non_dc_scalars_for_degree(degree: u8) -> usize {
    let d = degree.min(3) as usize;
    let mut total = 0usize;
    for l in 1..=d {
        total += SH_COEFS_PER_DEGREE[l] * 3;
    }
    total
}

/// Unpack `n` SOG-style smallest-3 packed quaternions from `bytes` (which must
/// be exactly `n * 4` bytes — one u32 LE per splat). Returns a flat `Vec<f32>`
/// of length `n * 4` in (x, y, z, w) order.
///
/// Wire format (matches `pack_chunk_with`, ROTATION branch):
/// `u32 = [q0:component_bits | q1:component_bits | q2:component_bits | tag:2]`
/// where the three stored components are `((val*sgn)/sqrt(2) + 0.5) * levels`
/// (rounded), `sgn` is the sign of the dropped (largest) component, and `tag`
/// (0..=3) records which original index was dropped.
fn unpack_smallest3_rotations(
    bytes: &[u8],
    n: usize,
    component_bits: u8,
) -> Result<Vec<f32>, GltfError> {
    if bytes.len() < n * 4 {
        return Err(GltfError::Malformed(format!(
            "CT_quat_smallest3: ROTATION needs {} bytes, have {}",
            n * 4,
            bytes.len()
        )));
    }
    let bits = component_bits.clamp(6, 10) as u32;
    let levels = ((1u32 << bits) - 1) as f32;
    let mask = (1u32 << bits) - 1;
    let sqrt2 = std::f32::consts::SQRT_2;
    let mut out = vec![0.0f32; n * 4];
    for i in 0..n {
        let off = i * 4;
        let packed =
            u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let tag = ((packed >> 30) & 3) as usize;
        // Stored components: t in [0, 1], reverse to val/sqrt(2) in [-0.5, 0.5]
        // so val = (t - 0.5) * sqrt(2), giving val in [-1/sqrt(2), 1/sqrt(2)].
        let mut stored = [0.0f32; 3];
        let mut sum_sq = 0.0f32;
        for k in 0..3 {
            let q = (packed >> (k as u32 * bits)) & mask;
            let t = q as f32 / levels;
            let v = (t - 0.5) * sqrt2;
            stored[k] = v;
            sum_sq += v * v;
        }
        // Largest (dropped) component is positive after the encoder's sign-
        // normalisation: |q[tag]| = sqrt(1 - Σ stored²). Numerical safety:
        // clamp to >=0 before sqrt.
        let largest = (1.0f32 - sum_sq).max(0.0).sqrt();
        // Reinsert into a 4-vector at position `tag`.
        let mut quat = [0.0f32; 4];
        let mut k = 0usize;
        for j in 0..4 {
            if j == tag {
                quat[j] = largest;
            } else {
                quat[j] = stored[k];
                k += 1;
            }
        }
        out[i * 4..i * 4 + 4].copy_from_slice(&quat);
    }
    Ok(out)
}

fn pad_to_4(buf: &mut Vec<u8>, pad_byte: u8) {
    while buf.len() % 4 != 0 {
        buf.push(pad_byte);
    }
}

/// Build the JSON + single binary buffer for an SPZ-compressed GLB.
///
/// The BIN chunk holds the raw SPZ blob (header magic 0x5053_4E47 "GNSP").
/// Per-attribute accessors are still emitted so loaders that don't understand
/// SPZ can locate the splat count, even though they point at zero-length
/// bufferViews — clients are expected to detect the SPZ extension on the
/// primitive and decode the blob via catetus-spz.
fn build_single_buffer_gltf_spz(
    scene: &SplatScene,
    variant: SpzVariant,
    opts: &WriteOpts,
) -> Result<(GltfRoot, Vec<u8>), GltfError> {
    let spz_blob = encode_spz(scene).map_err(|e| GltfError::Spz(e.to_string()))?;
    let bin: Vec<u8> = spz_blob.clone();

    let n = scene.splats.len();
    let (chunk_min, chunk_max) = chunk_bbox(scene.splats.as_slice());

    let extensions_used = vec![KHR.to_string(), KHR_SPZ.to_string()];
    let extensions_required = vec![KHR.to_string(), KHR_SPZ.to_string()];

    let mut root = GltfRoot {
        asset: GltfAsset {
            version: "2.0".to_string(),
            generator: Some("catetus-gltf".to_string()),
        },
        extensions_used,
        extensions_required,
        buffers: vec![GltfBuffer {
            byte_length: spz_blob.len(),
            uri: None,
        }],
        buffer_views: vec![GltfBufferView {
            buffer: 0,
            byte_offset: 0,
            byte_length: spz_blob.len(),
        }],
        accessors: Vec::new(),
        meshes: Vec::new(),
        extensions: serde_json::Map::new(),
    };
    let spz_view_idx = 0usize;

    // Build primitive JSON with KHR_gaussian_splatting + SPZ extension blocks.
    // We emit only the RC-shaped attribute layout (RcMay2026) here; SPZ is a
    // post-RC compression extension so this is the only sensible target.
    let attrs = serde_json::json!({
        "POSITION": serde_json::Value::Null,
    });
    // We can't actually use null accessor indices in valid glTF; instead emit
    // a single placeholder VEC3 FLOAT accessor with count=n pointing at the
    // SPZ bufferView. Decoders MUST consult the SPZ extension and decode the
    // blob; the accessor is purely a hint for the splat count.
    let placeholder_acc = root.accessors.len();
    root.accessors.push(GltfAccessor {
        buffer_view: spz_view_idx,
        component_type: FLOAT,
        normalized: false,
        count: n,
        accessor_type: "VEC3".to_string(),
        min: Some(vec![chunk_min[0], chunk_min[1], chunk_min[2]]),
        max: Some(vec![chunk_max[0], chunk_max[1], chunk_max[2]]),
    });
    let _ = attrs;

    let primitive = serde_json::json!({
        "mode": 0,
        "attributes": {
            "POSITION": placeholder_acc
        },
        "extensions": {
            KHR: {
                "kernel": "ellipse",
                "colorSpace": "srgb_rec709_display"
            },
            KHR_SPZ: {
                "version": variant.version(),
                "bufferView": spz_view_idx,
                "splatCount": n
            }
        }
    });

    root.meshes.push(serde_json::json!({
        "primitives": [primitive]
    }));

    let _ = opts; // currently no SPZ-specific options
    Ok((root, bin))
}

/// Write a `SplatScene` as a binary glTF (`.glb`) container with the JSON and
/// the splat data embedded as a single chunk. Chunked output is not supported
/// for GLB; pass `opts.chunked == false`.
pub fn write_glb(scene: &SplatScene, path: &Path, opts: &WriteOptions) -> Result<(), GltfError> {
    if opts.chunked {
        return Err(GltfError::GlbChunkedUnsupported);
    }
    let (mut root, mut bin) = match opts.compress {
        Some(variant) => build_single_buffer_gltf_spz(scene, variant, opts)?,
        None => build_single_buffer_gltf(scene, opts)?,
    };
    // Apply the optional brotli wrapper AFTER the per-attribute packing so it
    // wraps the final BIN payload exactly once. The SPZ path is already
    // content-compressed so we reject the combination — double-wrapping is a
    // net loss in our measurements.
    if let Some(wrap) = opts.lossless {
        if opts.compress.is_some() {
            return Err(GltfError::Brotli(
                "lossless brotli wrap is incompatible with --compress spz                  (the SPZ blob is already compressed)"
                    .to_string(),
            ));
        }
        bin = match wrap {
            LosslessWrap::Brotli11 => apply_brotli_wrap(&mut root, bin, wrap)?,
            LosslessWrap::Zstd19Split => apply_zstd_split_wrap(&mut root, bin)?,
        };
    }
    let json_str = serde_json::to_string(&root)?;

    // Chunk payloads padded to 4-byte alignment.
    let mut json_chunk: Vec<u8> = json_str.into_bytes();
    pad_to_4(&mut json_chunk, b' ');
    let mut bin_chunk: Vec<u8> = bin;
    pad_to_4(&mut bin_chunk, 0);

    let total: u32 = (12 + 8 + json_chunk.len() + 8 + bin_chunk.len()) as u32;
    let mut out: Vec<u8> = Vec::with_capacity(total as usize);
    out.write_u32::<LittleEndian>(GLB_MAGIC)?;
    out.write_u32::<LittleEndian>(2)?;
    out.write_u32::<LittleEndian>(total)?;
    // JSON chunk
    out.write_u32::<LittleEndian>(json_chunk.len() as u32)?;
    out.write_u32::<LittleEndian>(GLB_JSON)?;
    out.write_all(&json_chunk)?;
    // BIN chunk
    out.write_u32::<LittleEndian>(bin_chunk.len() as u32)?;
    out.write_u32::<LittleEndian>(GLB_BIN)?;
    out.write_all(&bin_chunk)?;

    fs::write(path, out)?;
    Ok(())
}

/// Reader options for [`read_glb_with_opts`].
///
/// All fields default to the strictest, most loss-detecting behavior — call
/// sites that need permissive decoding (viewers that gracefully degrade,
/// caches that don't co-locate sidecars) must opt in explicitly.
#[derive(Debug, Clone, Default)]
pub struct ReadOpts {
    /// When `true`, a missing `.shpal` sidecar referenced by
    /// `CT_gaussian_splatting_palette` silently degrades the scene to
    /// DC-only colour. When `false` (default), the decoder returns
    /// [`GltfError::MissingPaletteSidecar`] so callers do not accidentally
    /// produce zero-SH-rest scenes (the root cause of the ~9 dB PSNR
    /// regression in the canonical-11 bench, see
    /// `experiments/convert-shrest-fix/RESULT.md`). The environment
    /// variable `CATETUS_ALLOW_MISSING_PALETTE=1` flips this to `true`
    /// at runtime without touching call sites.
    pub allow_missing_palette: bool,
    /// When `true`, a missing `.glb.v5tail` sidecar referenced by
    /// `CT_v5_tail_residual` silently degrades to the baseline VQ45
    /// reconstruction (no residual applied). When `false` (default), the
    /// decoder returns [`GltfError::MissingTailSidecar`] iff the GLB
    /// declared the extension in `extensionsRequired`; merely listed in
    /// `extensionsUsed` + sidecar missing always produces a warning +
    /// baseline regardless of this flag. The environment variable
    /// `CATETUS_ALLOW_MISSING_TAIL=1` flips this to `true` at runtime.
    pub allow_missing_tail: bool,
}

impl ReadOpts {
    /// Resolve `allow_missing_palette`, taking the env var override into
    /// account. Set via `CATETUS_ALLOW_MISSING_PALETTE=1` (or `true`).
    fn allow_missing_palette_effective(&self) -> bool {
        if self.allow_missing_palette {
            return true;
        }
        match std::env::var("CATETUS_ALLOW_MISSING_PALETTE") {
            Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
            Err(_) => false,
        }
    }

    /// Resolve `allow_missing_tail`, taking the env var override into
    /// account. Set via `CATETUS_ALLOW_MISSING_TAIL=1` (or `true`).
    fn allow_missing_tail_effective(&self) -> bool {
        if self.allow_missing_tail {
            return true;
        }
        match std::env::var("CATETUS_ALLOW_MISSING_TAIL") {
            Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
            Err(_) => false,
        }
    }
}

/// Read a `.glb` file produced by `write_glb` back into a `SplatScene`. When
/// the GLB advertises `CT_gaussian_splatting_palette`, the `.shpal` sidecar
/// is resolved relative to the GLB's parent directory and decoded so the
/// returned scene carries full `Color::Sh` coefficients (not just DC).
///
/// **Strict mode (default):** if the GLB declares the palette extension but
/// the sidecar is missing, the call returns
/// [`GltfError::MissingPaletteSidecar`] rather than silently emitting
/// all-zero SH-rest coefficients. Use [`read_glb_with_opts`] (or set
/// `CATETUS_ALLOW_MISSING_PALETTE=1`) to opt back into the legacy
/// permissive behaviour.
pub fn read_glb(path: &Path) -> Result<SplatScene, GltfError> {
    read_glb_with_opts(path, &ReadOpts::default())
}

/// Variant of [`read_glb`] that takes a [`ReadOpts`] to opt into permissive
/// behaviour (e.g. allowing a missing `.shpal` sidecar).
pub fn read_glb_with_opts(path: &Path, opts: &ReadOpts) -> Result<SplatScene, GltfError> {
    let bytes = fs::read(path)?;
    let base = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let allow_missing = opts.allow_missing_palette_effective();
    let loader = |uri: &str| -> Result<Vec<u8>, GltfError> {
        let p = base.join(uri);
        match fs::read(&p) {
            Ok(bytes) => Ok(bytes),
            Err(_) if allow_missing => Err(GltfError::BufferNotFound(p.display().to_string())),
            Err(_) => Err(GltfError::MissingPaletteSidecar {
                uri: uri.to_string(),
                tried: p.display().to_string(),
            }),
        }
    };
    let mut scene = read_glb_bytes_with_sidecars(&bytes, Some(&loader))?;

    // V5.2 tail residual sidecar (post-pass). We have to re-parse the JSON
    // chunk here because `read_glb_bytes_with_sidecars` doesn't surface
    // extension JSON to callers — the API was designed for the palette
    // case where the loader callback fully resolves the sidecar. For
    // v5_tail we need access to the file system to locate the
    // `.glb.v5tail` next to the GLB, plus the `extensionsRequired`
    // strict-mode signal. Both are path-aware so they live here.
    let (json_str, _bin) = parse_glb_chunks(&bytes)?;
    let mut root: GltfRoot = serde_json::from_str(json_str)?;
    normalize_legacy_extensions(&mut root.extensions);
    normalize_legacy_extensions_used(&mut root.extensions_used);
    if let Some(tail_ext) = root.extensions.get(CT_V5_TAIL) {
        let uri = tail_ext
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GltfError::Malformed("CT_v5_tail_residual: missing uri".into()))?;
        let required = root.extensions_required.iter().any(|e| e == CT_V5_TAIL);
        let p = base.join(uri);
        match fs::read(&p) {
            Ok(sidecar_bytes) => {
                let decoded = v5_tail::decode_v5tail_bytes(&sidecar_bytes).map_err(|e| {
                    GltfError::Malformed(format!("CT_v5_tail_residual decode failed: {e:#}"))
                })?;
                let modified = apply_v5tail_to_scene(&mut scene, &decoded)?;
                eprintln!(
                    "applied CT_v5_tail_residual sidecar {} ({} bytes, K={}/{}, modified={})",
                    p.display(),
                    sidecar_bytes.len(),
                    decoded.header.k_selected,
                    decoded.header.n_splats,
                    modified,
                );
            }
            Err(_) => {
                // Hard-fail iff the sidecar is in extensionsRequired and the
                // user hasn't opted into the permissive mode.
                if required && !opts.allow_missing_tail_effective() {
                    return Err(GltfError::MissingTailSidecar {
                        uri: uri.to_string(),
                        tried: p.display().to_string(),
                    });
                }
                eprintln!(
                    "warning: CT_v5_tail_residual sidecar not found at {} \
                     (declared uri={}); rendering baseline reconstruction",
                    p.display(),
                    uri,
                );
            }
        }
    }
    Ok(scene)
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, GltfError> {
    if bytes.len() < offset + 4 {
        return Err(GltfError::Malformed("truncated GLB header".to_string()));
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

/// Parse a GLB byte stream into an IR scene, with an optional callback for
/// loading external sidecars referenced by extension URIs (currently only
/// `CT_gaussian_splatting_palette` → `.shpal`). Pass `None` (or use
/// [`read_glb_bytes`]) when sidecars don't exist or aren't available — the
/// reader degrades to `Color::Rgb` (DC only) in that case.
pub fn read_glb_bytes_with_sidecars<L>(
    bytes: &[u8],
    sidecar_loader: Option<&L>,
) -> Result<SplatScene, GltfError>
where
    L: Fn(&str) -> Result<Vec<u8>, GltfError>,
{
    let (json_str, bin_bytes) = parse_glb_chunks(bytes)?;
    read_glb_json_with_sidecars(json_str, bin_bytes, sidecar_loader)
}

/// Backwards-compatible no-sidecar variant. Equivalent to
/// `read_glb_bytes_with_sidecars(bytes, None::<&fn(_)->_>)`.
pub fn read_glb_bytes(bytes: &[u8]) -> Result<SplatScene, GltfError> {
    let (json_str, bin_bytes) = parse_glb_chunks(bytes)?;
    read_glb_json_with_sidecars::<fn(&str) -> Result<Vec<u8>, GltfError>>(json_str, bin_bytes, None)
}

/// Internal: split a GLB byte stream into its JSON-chunk string and BIN-chunk
/// slice. Bubbles up `GltfError::Malformed` for header / chunk shape issues.
fn parse_glb_chunks(bytes: &[u8]) -> Result<(&str, &[u8]), GltfError> {
    if bytes.len() < 12 {
        return Err(GltfError::Malformed("file too small for GLB".to_string()));
    }
    let magic = read_u32_le(bytes, 0)?;
    if magic != GLB_MAGIC {
        return Err(GltfError::Malformed("bad GLB magic".to_string()));
    }
    let version = read_u32_le(bytes, 4)?;
    if version != 2 {
        return Err(GltfError::Malformed(format!(
            "unsupported GLB version {version}"
        )));
    }
    let total = read_u32_le(bytes, 8)? as usize;
    if bytes.len() < total {
        return Err(GltfError::Malformed(
            "GLB length exceeds buffer".to_string(),
        ));
    }

    let mut offset = 12usize;
    let mut json_bytes: Option<&[u8]> = None;
    let mut bin_bytes: Option<&[u8]> = None;
    while offset + 8 <= total {
        let chunk_len = read_u32_le(bytes, offset)? as usize;
        let chunk_ty = read_u32_le(bytes, offset + 4)?;
        let data_start = offset + 8;
        let data_end = data_start
            .checked_add(chunk_len)
            .ok_or_else(|| GltfError::Malformed("chunk length overflow".to_string()))?;
        if data_end > total {
            return Err(GltfError::Malformed("chunk exceeds GLB length".to_string()));
        }
        match chunk_ty {
            GLB_JSON => json_bytes = Some(&bytes[data_start..data_end]),
            GLB_BIN => bin_bytes = Some(&bytes[data_start..data_end]),
            _ => {} // ignore unknown chunks
        }
        offset = data_end;
    }
    let json_bytes =
        json_bytes.ok_or_else(|| GltfError::Malformed("missing JSON chunk".to_string()))?;
    let bin_bytes =
        bin_bytes.ok_or_else(|| GltfError::Malformed("missing BIN chunk".to_string()))?;

    // Trim trailing whitespace padding from the JSON chunk before parsing.
    let json_trimmed = {
        let mut end = json_bytes.len();
        while end > 0 && (json_bytes[end - 1] == b' ' || json_bytes[end - 1] == 0) {
            end -= 1;
        }
        &json_bytes[..end]
    };
    let json_str = std::str::from_utf8(json_trimmed)
        .map_err(|_| GltfError::Malformed("JSON chunk not UTF-8".to_string()))?;
    Ok((json_str, bin_bytes))
}

fn read_glb_json_with_sidecars<L>(
    raw: &str,
    bin: &[u8],
    sidecar_loader: Option<&L>,
) -> Result<SplatScene, GltfError>
where
    L: Fn(&str) -> Result<Vec<u8>, GltfError>,
{
    let mut root: GltfRoot = serde_json::from_str(raw)?;
    normalize_legacy_extensions(&mut root.extensions);
    normalize_legacy_extensions_used(&mut root.extensions_used);
    if !root.extensions_used.iter().any(|e| e == KHR) {
        return Err(GltfError::MissingExtension);
    }
    // If the BIN chunk is brotli- or zstd-split-wrapped, decompress (and
    // reverse the byte-plane transpose, in the zstd-split case) once up
    // front so every bufferView offset below resolves against the
    // uncompressed bytes — the same coordinate space the writer used when
    // it laid out accessors.
    let decoded_bin: Option<Vec<u8>> = if let Some(v) = unwrap_zstd_split(&root, bin)? {
        Some(v)
    } else {
        unwrap_brotli(&root, bin)?
    };
    let bin: &[u8] = match decoded_bin.as_deref() {
        Some(decoded) => decoded,
        None => bin,
    };
    let prim_val = root
        .meshes
        .first()
        .and_then(|m| m.get("primitives"))
        .and_then(|p| p.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| GltfError::Malformed("no primitives".to_string()))?;

    // SPZ-compressed branch: splat data lives in the SPZ blob; non-SPZ
    // accessors are placeholders. Decode via catetus-spz and return early.
    if let Some(spz_ext) = prim_val.get("extensions").and_then(|e| e.get(KHR_SPZ)) {
        let bv_idx = spz_ext
            .get("bufferView")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| GltfError::Malformed("SPZ extension missing bufferView".to_string()))?
            as usize;
        let bv = root
            .buffer_views
            .get(bv_idx)
            .ok_or_else(|| GltfError::Malformed(format!("SPZ bufferView {bv_idx} out of range")))?;
        if bv.buffer != 0 {
            return Err(GltfError::Malformed(
                "SPZ GLB only supports buffer 0".to_string(),
            ));
        }
        if bin.len() < bv.byte_offset + bv.byte_length {
            return Err(GltfError::Malformed(
                "SPZ bufferView exceeds BIN chunk".to_string(),
            ));
        }
        let blob = &bin[bv.byte_offset..bv.byte_offset + bv.byte_length];
        return read_spz_bytes(blob).map_err(|e| GltfError::Spz(e.to_string()));
    }

    let attrs = resolve_attrs(prim_val)?;

    // CT_quat_smallest3: ROTATION is a SCALAR UNSIGNED_INT (5125) accessor;
    // `decode_accessor` doesn't recognise UINT, so we sniff the extension
    // here and pass the per-component bit width down to `decode_resolved_full`.
    let smallest3_bits: Option<u8> = root
        .extensions
        .get(CT_QUAT_SMALLEST3)
        .and_then(|e| e.get("componentBits"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u8);

    // CT_gaussian_splatting_palette: SH-rest lives in a `.shpal` sidecar that
    // a path-aware caller (e.g. `read_glb` from disk) can hand back via
    // `sidecar_loader`. We sanity-check the GLB-side metadata against the
    // sidecar header inside `decode_shpal_bytes`.
    let palette: Option<ShPalette> = match (root.extensions.get(CT_PALETTE), sidecar_loader) {
        (Some(pal_ext), Some(loader)) => {
            let uri = pal_ext.get("uri").and_then(|v| v.as_str()).ok_or_else(|| {
                GltfError::Malformed("CT_gaussian_splatting_palette: missing uri".into())
            })?;
            // Missing sidecar is not fatal — the GLB still loads with DC-only
            // colour, matching legacy behaviour and the JS bench harness
            // when the .shpal file isn't co-located. Other decode failures
            // (corrupt magic, length mismatch) still propagate so the user
            // gets a clear error rather than silently-wrong SH-rest.
            match loader(uri) {
                Ok(bytes) => {
                    let exp_k = pal_ext
                        .get("paletteSize")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let exp_n = pal_ext
                        .get("splatCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let exp_b = pal_ext
                        .get("codebookBits")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u8;
                    let sh_deg = pal_ext
                        .get("shDegree")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(3) as u8;
                    Some(decode_shpal_bytes(
                        &bytes,
                        Some((exp_k, exp_n, exp_b)),
                        sh_deg,
                    )?)
                }
                // `BufferNotFound` is the permissive sentinel — the loader
                // chose to degrade to DC-only rather than hard-fail (e.g.
                // the byte-stream API where the caller can't pre-locate
                // sidecars). `MissingPaletteSidecar` is the strict signal
                // from the disk-path loader and must propagate so the user
                // sees the actionable error rather than a silently zeroed
                // SH-rest field.
                Err(GltfError::BufferNotFound(_)) => None,
                Err(e) => return Err(e),
            }
        }
        _ => None,
    };

    let read_attr = |acc_idx: usize, comps: usize| -> Result<Vec<f32>, GltfError> {
        let acc = &root.accessors[acc_idx];
        let bv = &root.buffer_views[acc.buffer_view];
        if bv.buffer != 0 {
            return Err(GltfError::Malformed(
                "GLB only supports buffer 0".to_string(),
            ));
        }
        if bin.len() < bv.byte_offset + bv.byte_length {
            return Err(GltfError::Malformed("accessor out of range".to_string()));
        }
        let bytes = &bin[bv.byte_offset..bv.byte_offset + bv.byte_length];
        decode_accessor(bytes, acc, comps)
    };
    let read_acc_raw = |acc_idx: usize| -> Result<Vec<u8>, GltfError> {
        let acc = &root.accessors[acc_idx];
        let bv = &root.buffer_views[acc.buffer_view];
        if bv.buffer != 0 {
            return Err(GltfError::Malformed(
                "GLB only supports buffer 0".to_string(),
            ));
        }
        if bin.len() < bv.byte_offset + bv.byte_length {
            return Err(GltfError::Malformed("accessor out of range".to_string()));
        }
        Ok(bin[bv.byte_offset..bv.byte_offset + bv.byte_length].to_vec())
    };

    let log_quant_attrs = root.extensions.contains_key(CT_LOG_QUANT_ATTRS);

    let ext = DecodeExtensions {
        palette: palette.as_ref(),
        smallest3_bits,
        log_quant_attrs,
    };
    decode_resolved_full(&attrs, &read_attr, &ext, &read_acc_raw)
}
