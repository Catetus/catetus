#![deny(clippy::all)]
//! glTF 2.0 + `KHR_gaussian_splatting` writer/reader, with the optional
//! `SF_spatial_streaming_index` vendor extension defined in SPEC-0007.
//!
//! We hand-roll the JSON to stay in control of the wire format (the
//! `gltf` crate doesn't know about KHR_gaussian_splatting yet).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, WriteBytesExt};
use serde::{Deserialize, Serialize};
use splatforge_core::{Color, CoordinateSystem, Splat, SplatScene, TemporalMode};
use splatforge_spz::{encode_spz, read_spz_bytes};
use thiserror::Error;

/// Variants of the `KHR_gaussian_splatting_compression_spz` extension that the
/// writer can emit. The wire-version integer flows straight into the
/// extension's `version` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpzVariant {
    /// SPZ v2 — the current wire format produced by `splatforge-spz`.
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
/// `RcMay2026` default so it passes `splatforge-khr-conformance` and the
/// upstream Khronos validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpecVersion {
    /// Khronos RC text at commit `63770cc7` (2026-04-15) — the default.
    #[default]
    RcMay2026,
    /// Pre-RC layout used by SplatForge v0.x and the legacy web viewer.
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
    /// The `splatforge optimize` CLI flips this on for the web-targeted presets.
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
    /// When set, route the splat payload through `splatforge-spz` and emit
    /// the `KHR_gaussian_splatting_compression_spz` extension on the output.
    /// Only meaningful for GLB output.
    pub compress: Option<SpzVariant>,
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
        }
    }
}

/// Result of `inspect_gltf`.
#[derive(Debug, Clone)]
pub struct InspectReport {
    /// Whether `KHR_gaussian_splatting` is declared.
    pub has_khr: bool,
    /// Whether the `SF_spatial_streaming_index` extension is present.
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
const SF_INDEX: &str = "SF_spatial_streaming_index";
const KHR_QUANT: &str = "KHR_mesh_quantization";
const FLOAT: u32 = 5126;
const UBYTE: u32 = 5121;
const USHORT: u32 = 5123;
/// Signed SHORT (`5122`). Used for RC quaternion quantization
/// (`KHR_gaussian_splatting:ROTATION` with `normalized: true`).
const SHORT: u32 = 5122;

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
/// and `splatforge-khr-conformance::sh_coef_count`.
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
#[derive(Debug, Clone, Copy)]
struct ChunkLayout {
    spec: SpecVersion,
    quantize: bool,
    /// When true and `quantize` is set, ROTATION is emitted as a normalized
    /// signed SHORT (`5122 / VEC4 / normalized`).
    quantize_rotation: bool,
    /// Highest SH degree the chunk contains (0 = none).
    sh_degree: u8,
}

impl ChunkLayout {
    fn position_bytes(&self) -> usize {
        if self.quantize {
            6
        } else {
            12
        }
    }
    fn rotation_bytes(&self) -> usize {
        if self.quantize && self.quantize_rotation {
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
#[allow(clippy::too_many_arguments)]
fn emit_chunk_accessors(
    root: &mut GltfRoot,
    buffer_idx: usize,
    n: usize,
    pos_min: &[f32; 3],
    pos_max: &[f32; 3],
    scale_min: &[f32; 3],
    scale_max: &[f32; 3],
    layout: ChunkLayout,
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
    let rot = if layout.quantize && layout.quantize_rotation {
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
            a.min = Some(vec![0.0]);
            a.max = Some(vec![1.0]);
        }
        offset = align_up(offset, 4);
        acc
    } else {
        add_view_acc(root, buffer_idx, &mut offset, n, n * 4, "SCALAR")
    };

    // DC color. Under RC, must be VEC3 FLOAT to satisfy ACC_SH_COEF — the
    // validator requires every SH coefficient accessor (DC included) to be
    // VEC3 FLOAT, so we ignore `quantize` for the DC under RC.
    let color_dc = match layout.spec {
        SpecVersion::RcMay2026 => add_view_acc(root, buffer_idx, &mut offset, n, n * 12, "VEC3"),
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
    };

    // SH coefficient accessors.
    let mut sh_coefs: Vec<((u8, u8), usize)> = Vec::new();
    if layout.sh_degree > 0 {
        match layout.spec {
            SpecVersion::RcMay2026 => {
                // One VEC3 FLOAT accessor per coefficient at degrees 1..=top.
                for l in 1..=layout.sh_degree {
                    for n_idx in 0..SH_COEFS_PER_DEGREE[l as usize] as u8 {
                        let acc = add_view_acc(root, buffer_idx, &mut offset, n, n * 12, "VEC3");
                        sh_coefs.push(((l, n_idx), acc));
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
/// historical SplatForge layout is preserved.
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
            generator: Some("splatforge-gltf".to_string()),
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
        root.extensions_used.push(SF_INDEX.to_string());
    }
    if opts.quantize {
        // Non-required: viewers that don't implement the extension still load
        // the asset, just with un-dequantized integer values. See SPEC-0013.
        root.extensions_used.push(KHR_QUANT.to_string());
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
        let (scale_min, scale_max) = chunk_scale_bbox(chunk);
        let chunk_sh_degree: u8 = chunk.iter().map(|s| s.color.degree()).max().unwrap_or(0);
        let layout = ChunkLayout {
            spec: opts.spec_version,
            quantize: opts.quantize,
            quantize_rotation: opts.quantize_rotation,
            sh_degree: chunk_sh_degree,
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
            &mut root, buffer_idx, n, &chunk_min, &chunk_max, &scale_min, &scale_max, layout,
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

    if opts.chunked {
        let lods: Vec<serde_json::Value> = opts
            .lod_fractions
            .iter()
            .enumerate()
            .map(|(i, f)| serde_json::json!({ "level": i, "splatFraction": f }))
            .collect();
        root.extensions.insert(
            SF_INDEX.to_string(),
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

/// Compute per-axis min/max of `scale` over the chunk. Returns `[0,0,0]` /
/// `[1,1,1]` for an empty chunk so the dequantized range remains well defined.
fn chunk_scale_bbox(chunk: &[Splat]) -> ([f32; 3], [f32; 3]) {
    if chunk.is_empty() {
        return ([0.0; 3], [1.0; 3]);
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for s in chunk {
        for i in 0..3 {
            if s.scale[i] < mn[i] {
                mn[i] = s.scale[i];
            }
            if s.scale[i] > mx[i] {
                mx[i] = s.scale[i];
            }
        }
    }
    // Guarantee mx > mn so the dequant span is non-zero.
    for i in 0..3 {
        if mx[i] <= mn[i] {
            mx[i] = mn[i] + f32::EPSILON;
        }
    }
    (mn, mx)
}

fn pack_chunk_with(
    chunk: &[Splat],
    layout: &ChunkLayout,
    pos_min: &[f32; 3],
    pos_max: &[f32; 3],
) -> Vec<u8> {
    let n = chunk.len();
    let (scale_min, scale_max) = chunk_scale_bbox(chunk);

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
    if layout.quantize && layout.quantize_rotation {
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
                out.push(quantize_u8(s.scale[i], scale_min[i], scale_max[i]));
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
            out.push(quantize_u8(s.opacity, 0.0, 1.0));
        }
        pad_to(&mut out, 4);
    } else {
        for s in chunk {
            out.write_f32::<LittleEndian>(s.opacity).unwrap();
        }
    }

    // DC color. Under RC, always VEC3 FLOAT; under pre-RC, u8 normalized
    // when `quantize` is set, else VEC3 FLOAT.
    let dc_as_float = match layout.spec {
        SpecVersion::RcMay2026 => true,
        SpecVersion::Pre2026 => !layout.quantize,
    };
    if dc_as_float {
        for s in chunk {
            let dc = match &s.color {
                Color::Rgb(c) => *c,
                Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
            };
            for v in dc {
                out.write_f32::<LittleEndian>(v).unwrap();
            }
        }
    } else {
        for s in chunk {
            let dc = match &s.color {
                Color::Rgb(c) => *c,
                Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
            };
            for v in dc {
                out.push(quantize_u8(v, 0.0, 1.0));
            }
        }
        pad_to(&mut out, 4);
    }

    // SH coefficients (degrees 1..=top).
    if layout.sh_degree > 0 {
        match layout.spec {
            SpecVersion::RcMay2026 => {
                // Per-coefficient VEC3 FLOAT accessors. Walk (l, n) in the
                // same order as `emit_chunk_accessors`, and for each pull
                // the 3-scalar slice from the splat's coeffs vector. Our
                // pipeline lays coeffs as `[DC.r, DC.g, DC.b, c1_0.r,
                // c1_0.g, c1_0.b, c1_1.r, ...]` — i.e. interleaved per
                // coefficient, RGB-major. That matches the RC's
                // per-coefficient VEC3 layout, just sliced.
                let coef_count: usize = SH_COEFS_PER_DEGREE[1..=layout.sh_degree as usize]
                    .iter()
                    .sum();
                for (coef_idx, _) in (0..coef_count).enumerate() {
                    for s in chunk {
                        match &s.color {
                            Color::Sh { coeffs, .. } => {
                                // DC occupies coeffs[0..3]; coefficient
                                // `coef_idx` (0-based among non-DC) lives at
                                // `3 + coef_idx*3 .. 3 + coef_idx*3 + 3`.
                                let base = 3 + coef_idx * 3;
                                for k in 0..3 {
                                    let v = coeffs.get(base + k).copied().unwrap_or(0.0);
                                    out.write_f32::<LittleEndian>(v).unwrap();
                                }
                            }
                            Color::Rgb(_) => {
                                for _ in 0..3 {
                                    out.write_f32::<LittleEndian>(0.0).unwrap();
                                }
                            }
                        }
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
            // Signed SHORT — used for the RC normalized quaternion encoding.
            if bytes.len() < total * 2 {
                return Err(GltfError::Malformed("i16 accessor under-sized".to_string()));
            }
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let c = &bytes[i * 2..i * 2 + 2];
                let q = i16::from_le_bytes([c[0], c[1]]);
                let v = if acc.normalized {
                    dequantize_i16(q)
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
pub fn read_gltf(path: &Path) -> Result<SplatScene, GltfError> {
    let raw = fs::read_to_string(path)?;
    let dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    read_gltf_str(&raw, &dir)
}

fn read_gltf_str(raw: &str, base_dir: &Path) -> Result<SplatScene, GltfError> {
    let root: GltfRoot = serde_json::from_str(raw)?;
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

    decode_resolved(&attrs, &read_attr)
}

/// Materialise a `SplatScene` from a `ResolvedAttrs` plus a closure that
/// decodes an accessor index into a flat `Vec<f32>`. Shared by both the
/// `.gltf` and `.glb` readers so the spec-version reconciliation stays in
/// one place.
fn decode_resolved<F>(attrs: &ResolvedAttrs, read_attr: &F) -> Result<SplatScene, GltfError>
where
    F: Fn(usize, usize) -> Result<Vec<f32>, GltfError>,
{
    let positions = read_attr(attrs.pos, 3)?;
    let rotations = read_attr(attrs.rot, 4)?;
    let scales = read_attr(attrs.scale, 3)?;
    let opacities = read_attr(attrs.opacity, 1)?;
    let dc = read_attr(attrs.dc, 3)?;
    let n = opacities.len();

    // Flatten SH back into the 45-scalar interleaved layout that `Color::Sh`
    // expects (3 DC + 45 trailing = 48 floats).
    let mut sh_flat: Option<Vec<f32>> = None;
    if !attrs.non_dc_sh.is_empty() {
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

    let has_spatial_index = used.iter().any(|e| e == SF_INDEX);
    let mut chunk_count = 0usize;
    // Checksum mismatches bail out with an early `Err` rather than flagging
    // here, so the report just records the success state.
    let checksum_ok = true;
    let mut splat_count = 0usize;
    if has_spatial_index {
        let chunks = value
            .get("extensions")
            .and_then(|e| e.get(SF_INDEX))
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
    let (scale_min, scale_max) = chunk_scale_bbox(chunk);
    let sh_degree: u8 = chunk.iter().map(|s| s.color.degree()).max().unwrap_or(0);
    let layout = ChunkLayout {
        spec: opts.spec_version,
        quantize: opts.quantize,
        quantize_rotation: opts.quantize_rotation,
        sh_degree,
    };
    let buf_bytes = pack_chunk_with(chunk, &layout, &chunk_min, &chunk_max);

    let mut extensions_used = vec![KHR.to_string()];
    if opts.quantize {
        extensions_used.push(KHR_QUANT.to_string());
    }
    let mut root = GltfRoot {
        asset: GltfAsset {
            version: "2.0".to_string(),
            generator: Some("splatforge-gltf".to_string()),
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
        &mut root, 0, n, &chunk_min, &chunk_max, &scale_min, &scale_max, layout,
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

    Ok((root, buf_bytes))
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
/// primitive and decode the blob via splatforge-spz.
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
            generator: Some("splatforge-gltf".to_string()),
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
    let (root, bin) = match opts.compress {
        Some(variant) => build_single_buffer_gltf_spz(scene, variant, opts)?,
        None => build_single_buffer_gltf(scene, opts)?,
    };
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

/// Read a `.glb` file produced by `write_glb` back into a `SplatScene`.
pub fn read_glb(path: &Path) -> Result<SplatScene, GltfError> {
    let bytes = fs::read(path)?;
    read_glb_bytes(&bytes)
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

/// Parse a GLB byte stream into an IR scene.
pub fn read_glb_bytes(bytes: &[u8]) -> Result<SplatScene, GltfError> {
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
    read_glb_json(json_str, bin_bytes)
}

fn read_glb_json(raw: &str, bin: &[u8]) -> Result<SplatScene, GltfError> {
    let root: GltfRoot = serde_json::from_str(raw)?;
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

    // SPZ-compressed branch: splat data lives in the SPZ blob; non-SPZ
    // accessors are placeholders. Decode via splatforge-spz and return early.
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

    decode_resolved(&attrs, &read_attr)
}
