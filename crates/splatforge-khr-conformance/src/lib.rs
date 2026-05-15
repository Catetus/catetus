#![deny(clippy::all)]
//! Conformance test suite for the Khronos `KHR_gaussian_splatting` glTF
//! extension.
//!
//! Each public [`Clause`] corresponds to one normative requirement in the
//! KHR_gaussian_splatting Release Candidate text (SHA
//! `63770cc70a3709cf101a42cece0bdf602b37e2e7`, dated 2026-04-15 — the
//! "Editorial review" RC merge). The validator loads a glTF 2.0 JSON
//! document — either as an external `.gltf` or extracted from the JSON
//! chunk of a `.glb` container — and returns a [`Report`] that says
//! whether every clause passed, failed, or was skipped (not applicable).
//!
//! The report is JSON-serialisable so the same code can drive both Rust
//! integration tests and the `splatforge-khr-validate` CLI binary.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifier for every spec clause the validator can check.
///
/// The string forms (`"EXT_USED"`, `"PRIM_EXT"`, …) are stable and become
/// part of the public JSON report contract.  Removing or renaming one is a
/// breaking change to the conformance protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Clause {
    ExtUsed,
    AssetVersion,
    PrimitiveExtensionPresent,
    PrimitiveModePoints,
    ExtKernelRequired,
    ExtColorSpaceRequired,
    ExtProjectionValid,
    ExtSortingValid,
    PositionPresent,
    RotationPresent,
    ScalePresent,
    OpacityPresent,
    ShDcPresent,
    PositionAccessor,
    RotationAccessor,
    ScaleAccessor,
    OpacityAccessor,
    ShCoefAccessor,
    PositionMinMax,
    ShDegreesFullyDefined,
    AccessorCountsAgree,
    BufferViewBounds,
    NoUnknownNamespacedAttributes,
    SpzExtensionDeclared,
    SpzExtensionConsistent,
    SpzExtPresent,
    SpzVersion,
    SpzBufferView,
    SpzBlobMagic,
    SpzDecodedCount,
}

impl Clause {
    /// All clauses in spec order. Used by the CLI to render the report.
    pub fn all() -> &'static [Clause] {
        &[
            Clause::ExtUsed,
            Clause::AssetVersion,
            Clause::PrimitiveExtensionPresent,
            Clause::PrimitiveModePoints,
            Clause::ExtKernelRequired,
            Clause::ExtColorSpaceRequired,
            Clause::ExtProjectionValid,
            Clause::ExtSortingValid,
            Clause::PositionPresent,
            Clause::RotationPresent,
            Clause::ScalePresent,
            Clause::OpacityPresent,
            Clause::ShDcPresent,
            Clause::PositionAccessor,
            Clause::RotationAccessor,
            Clause::ScaleAccessor,
            Clause::OpacityAccessor,
            Clause::ShCoefAccessor,
            Clause::PositionMinMax,
            Clause::ShDegreesFullyDefined,
            Clause::AccessorCountsAgree,
            Clause::BufferViewBounds,
            Clause::NoUnknownNamespacedAttributes,
            Clause::SpzExtensionDeclared,
            Clause::SpzExtensionConsistent,
            Clause::SpzExtPresent,
            Clause::SpzVersion,
            Clause::SpzBufferView,
            Clause::SpzBlobMagic,
            Clause::SpzDecodedCount,
        ]
    }

    /// Short, stable identifier used in JSON output and CLI tables.
    pub fn id(self) -> &'static str {
        match self {
            Clause::ExtUsed => "EXT_USED",
            Clause::AssetVersion => "ASSET_VERSION",
            Clause::PrimitiveExtensionPresent => "PRIM_EXT",
            Clause::PrimitiveModePoints => "PRIM_MODE_POINTS",
            Clause::ExtKernelRequired => "EXT_KERNEL",
            Clause::ExtColorSpaceRequired => "EXT_COLOR_SPACE",
            Clause::ExtProjectionValid => "EXT_PROJECTION",
            Clause::ExtSortingValid => "EXT_SORTING",
            Clause::PositionPresent => "ATTR_POSITION",
            Clause::RotationPresent => "ATTR_ROTATION",
            Clause::ScalePresent => "ATTR_SCALE",
            Clause::OpacityPresent => "ATTR_OPACITY",
            Clause::ShDcPresent => "ATTR_SH_DC",
            Clause::PositionAccessor => "ACC_POSITION",
            Clause::RotationAccessor => "ACC_ROTATION",
            Clause::ScaleAccessor => "ACC_SCALE",
            Clause::OpacityAccessor => "ACC_OPACITY",
            Clause::ShCoefAccessor => "ACC_SH_COEF",
            Clause::PositionMinMax => "ACC_POSITION_MINMAX",
            Clause::ShDegreesFullyDefined => "SH_DEGREES_FULL",
            Clause::AccessorCountsAgree => "ACC_COUNTS_AGREE",
            Clause::BufferViewBounds => "BUFFERVIEW_BOUNDS",
            Clause::NoUnknownNamespacedAttributes => "ATTRS_KNOWN_ONLY",
            Clause::SpzExtensionDeclared => "SPZ_DECLARED",
            Clause::SpzExtensionConsistent => "SPZ_CONSISTENT",
            Clause::SpzExtPresent => "SPZ_EXT_PRESENT",
            Clause::SpzVersion => "SPZ_VERSION",
            Clause::SpzBufferView => "SPZ_BUFFERVIEW",
            Clause::SpzBlobMagic => "SPZ_BLOB_MAGIC",
            Clause::SpzDecodedCount => "SPZ_DECODED_COUNT",
        }
    }

    /// Human-readable description, suitable for the conformance.md table.
    pub fn description(self) -> &'static str {
        match self {
            Clause::ExtUsed => {
                "Root extensionsUsed array MUST list \"KHR_gaussian_splatting\"."
            }
            Clause::AssetVersion => "asset.version MUST be \"2.0\" per glTF 2.0.",
            Clause::PrimitiveExtensionPresent => {
                "At least one mesh.primitives[i].extensions[\"KHR_gaussian_splatting\"] block MUST be present."
            }
            Clause::PrimitiveModePoints => {
                "A primitive carrying KHR_gaussian_splatting MUST set mode to POINTS (0)."
            }
            Clause::ExtKernelRequired => {
                "The KHR_gaussian_splatting extension object MUST declare a string `kernel`."
            }
            Clause::ExtColorSpaceRequired => {
                "The KHR_gaussian_splatting extension object MUST declare a string `colorSpace`."
            }
            Clause::ExtProjectionValid => {
                "If `projection` is present it MUST be a string (default \"perspective\")."
            }
            Clause::ExtSortingValid => {
                "If `sortingMethod` is present it MUST be a string (default \"cameraDistance\")."
            }
            Clause::PositionPresent => {
                "The primitive's attributes object MUST declare a POSITION accessor."
            }
            Clause::RotationPresent => {
                "The attributes object MUST declare KHR_gaussian_splatting:ROTATION."
            }
            Clause::ScalePresent => {
                "The attributes object MUST declare KHR_gaussian_splatting:SCALE."
            }
            Clause::OpacityPresent => {
                "The attributes object MUST declare KHR_gaussian_splatting:OPACITY."
            }
            Clause::ShDcPresent => {
                "The attributes object MUST declare KHR_gaussian_splatting:SH_DEGREE_0_COEF_0."
            }
            Clause::PositionAccessor => {
                "POSITION accessor MUST be VEC3 (componentType FLOAT, or normalized UNSIGNED_SHORT/UNSIGNED_BYTE under KHR_mesh_quantization)."
            }
            Clause::RotationAccessor => {
                "KHR_gaussian_splatting:ROTATION accessor MUST be VEC4 FLOAT, or normalized BYTE / SHORT (unit quaternion xyzw)."
            }
            Clause::ScaleAccessor => {
                "KHR_gaussian_splatting:SCALE accessor MUST be VEC3 (FLOAT or unsigned-integer, with normalized variants allowed)."
            }
            Clause::OpacityAccessor => {
                "KHR_gaussian_splatting:OPACITY accessor MUST be SCALAR (FLOAT or normalized UNSIGNED_BYTE / UNSIGNED_SHORT)."
            }
            Clause::ShCoefAccessor => {
                "Every KHR_gaussian_splatting:SH_DEGREE_l_COEF_n accessor MUST be VEC3 FLOAT."
            }
            Clause::PositionMinMax => {
                "POSITION accessor MUST provide both min and max arrays (glTF 2.0 §3.6.2.4)."
            }
            Clause::ShDegreesFullyDefined => {
                "SH degrees MUST be fully defined — each degree l requires its full (2l+1) coefficient set, and using degree l requires degrees 0..l-1."
            }
            Clause::AccessorCountsAgree => {
                "All per-splat accessors (POSITION, ROTATION, SCALE, OPACITY, and every SH coefficient accessor) MUST share the same count."
            }
            Clause::BufferViewBounds => {
                "Every referenced accessor's bufferView MUST be in range and its byte footprint MUST fit inside the parent buffer."
            }
            Clause::NoUnknownNamespacedAttributes => {
                "Any `KHR_gaussian_splatting:*` attribute key MUST match one of the names defined by the spec (ROTATION, SCALE, OPACITY, SH_DEGREE_{0..3}_COEF_{0..6})."
            }
            Clause::SpzExtensionDeclared => {
                "If a primitive declares KHR_gaussian_splatting_compression_spz, the root extensionsUsed MUST list it."
            }
            Clause::SpzExtensionConsistent => {
                "A primitive declaring KHR_gaussian_splatting_compression_spz MUST also declare KHR_gaussian_splatting."
            }
            Clause::SpzExtPresent => {
                "The primitive carrying KHR_gaussian_splatting_compression_spz MUST attach the extension object."
            }
            Clause::SpzVersion => {
                "The SPZ extension MUST declare version == 2 (current SPZ wire format)."
            }
            Clause::SpzBufferView => {
                "The SPZ extension MUST reference a bufferView that fits inside its buffer and is at least 16 bytes long."
            }
            Clause::SpzBlobMagic => {
                "The first four bytes of the SPZ blob MUST be the SPZ magic 0x5053_4E47 (little-endian: \"GNSP\")."
            }
            Clause::SpzDecodedCount => {
                "The SPZ blob's header splat_count MUST match the primitive's declared splat count."
            }
        }
    }

    /// Whether this clause is a MUST (true) or SHOULD/optional (false).
    pub fn is_mandatory(self) -> bool {
        true
    }
}

/// Outcome of evaluating a single clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Clause checked and satisfied.
    Pass,
    /// Clause checked and violated.
    Fail,
    /// Clause did not apply to this asset.
    Skip,
}

/// Per-clause result emitted by [`validate_path`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseResult {
    /// Short stable identifier (`Clause::id`).
    pub id: String,
    /// Outcome bucket.
    pub status: Status,
    /// Optional human-readable detail; required on Fail.
    pub detail: Option<String>,
}

/// Top-level validator report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// File the report was produced for, normalised to its display string.
    pub source: String,
    /// glTF container variant: `"gltf"` or `"glb"`.
    pub container: String,
    /// Number of clauses with status [`Status::Pass`].
    pub pass: usize,
    /// Number of clauses with status [`Status::Fail`].
    pub fail: usize,
    /// Number of clauses with status [`Status::Skip`].
    pub skip: usize,
    /// Full per-clause breakdown in spec order.
    pub clauses: Vec<ClauseResult>,
}

impl Report {
    /// True when no clause failed (skips are tolerated).
    pub fn is_pass(&self) -> bool {
        self.fail == 0
    }
}

/// Validator-level errors. These never become clause failures because they
/// short-circuit the whole report.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum ValidateError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("malformed glb: {0}")]
    Glb(String),
    #[error("unsupported file extension: {0}")]
    UnsupportedExt(String),
}

const GLB_MAGIC: [u8; 4] = *b"glTF";
const KHR: &str = "KHR_gaussian_splatting";
const NS: &str = "KHR_gaussian_splatting:";
const SPZ: &str = "KHR_gaussian_splatting_compression_spz";

/// Validate a `.gltf` or `.glb` file and return the conformance report.
pub fn validate_path(path: &Path) -> Result<Report, ValidateError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let bytes = fs::read(path)?;
    let (json_str, container, bin) = match ext.as_str() {
        "gltf" => (
            String::from_utf8_lossy(&bytes).to_string(),
            "gltf",
            None,
        ),
        "glb" => {
            let (j, b) = extract_glb_chunks(&bytes)?;
            (j, "glb", b)
        }
        other => return Err(ValidateError::UnsupportedExt(other.to_string())),
    };
    let value: serde_json::Value = serde_json::from_str(&json_str)?;
    let clauses = run_clauses(&value, bin.as_deref());
    Ok(summarize(
        path.display().to_string(),
        container.to_string(),
        clauses,
    ))
}

/// Validate a glTF JSON document already in memory.
pub fn validate_json(json: &serde_json::Value, source: &str) -> Report {
    let clauses = run_clauses(json, None);
    summarize(source.to_string(), "gltf".to_string(), clauses)
}

/// Validate a glTF JSON document with an associated BIN chunk in memory.
pub fn validate_json_with_bin(
    json: &serde_json::Value,
    bin: Option<&[u8]>,
    source: &str,
) -> Report {
    let clauses = run_clauses(json, bin);
    summarize(source.to_string(), "gltf".to_string(), clauses)
}

fn summarize(source: String, container: String, clauses: Vec<ClauseResult>) -> Report {
    let mut pass = 0;
    let mut fail = 0;
    let mut skip = 0;
    for c in &clauses {
        match c.status {
            Status::Pass => pass += 1,
            Status::Fail => fail += 1,
            Status::Skip => skip += 1,
        }
    }
    Report {
        source,
        container,
        pass,
        fail,
        skip,
        clauses,
    }
}

fn extract_glb_chunks(bytes: &[u8]) -> Result<(String, Option<Vec<u8>>), ValidateError> {
    if bytes.len() < 12 || bytes[..4] != GLB_MAGIC {
        return Err(ValidateError::Glb("bad magic".to_string()));
    }
    let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if bytes.len() < total {
        return Err(ValidateError::Glb("truncated".to_string()));
    }
    let mut offset = 12usize;
    let mut json: Option<String> = None;
    let mut bin: Option<Vec<u8>> = None;
    while offset + 8 <= total {
        let chunk_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let chunk_ty = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        let data_start = offset + 8;
        let data_end = data_start + chunk_len;
        if data_end > total {
            return Err(ValidateError::Glb("chunk exceeds total".to_string()));
        }
        match chunk_ty {
            0x4E4F_534A => {
                // "JSON"
                let mut end = data_end;
                while end > data_start && (bytes[end - 1] == b' ' || bytes[end - 1] == 0) {
                    end -= 1;
                }
                json = Some(
                    std::str::from_utf8(&bytes[data_start..end])
                        .map_err(|_| ValidateError::Glb("JSON chunk not UTF-8".to_string()))?
                        .to_string(),
                );
            }
            0x004E_4942 => {
                // "BIN\0"
                bin = Some(bytes[data_start..data_end].to_vec());
            }
            _ => {}
        }
        offset = data_end;
    }
    let j = json.ok_or_else(|| ValidateError::Glb("missing JSON chunk".to_string()))?;
    Ok((j, bin))
}

// ---------- clause evaluation ----------

fn pass(id: Clause) -> ClauseResult {
    ClauseResult {
        id: id.id().to_string(),
        status: Status::Pass,
        detail: None,
    }
}

fn fail(id: Clause, detail: impl Into<String>) -> ClauseResult {
    ClauseResult {
        id: id.id().to_string(),
        status: Status::Fail,
        detail: Some(detail.into()),
    }
}

fn skip(id: Clause, detail: impl Into<String>) -> ClauseResult {
    ClauseResult {
        id: id.id().to_string(),
        status: Status::Skip,
        detail: Some(detail.into()),
    }
}

/// Names of valid `KHR_gaussian_splatting:*` namespaced attributes
/// per the RC text.
fn known_namespaced_attrs() -> Vec<String> {
    let mut v = vec![
        "ROTATION".to_string(),
        "SCALE".to_string(),
        "OPACITY".to_string(),
    ];
    // SH degree 0: 1 coef; 1: 3; 2: 5; 3: 7.
    let coefs_per_degree = [1usize, 3, 5, 7];
    for (l, n) in coefs_per_degree.iter().enumerate() {
        for i in 0..*n {
            v.push(format!("SH_DEGREE_{l}_COEF_{i}"));
        }
    }
    v
}

fn sh_attr_name(l: usize, n: usize) -> String {
    format!("{NS}SH_DEGREE_{l}_COEF_{n}")
}

fn sh_coef_count(l: usize) -> usize {
    2 * l + 1
}

fn run_clauses(root: &serde_json::Value, bin: Option<&[u8]>) -> Vec<ClauseResult> {
    let mut out = Vec::with_capacity(Clause::all().len());

    // ----- root-level extension declaration -----
    let used: Vec<&str> = root
        .get("extensionsUsed")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    let has_khr_used = used.contains(&KHR);
    let has_spz_used = used.contains(&SPZ);

    out.push(if has_khr_used {
        pass(Clause::ExtUsed)
    } else {
        fail(
            Clause::ExtUsed,
            "KHR_gaussian_splatting not in extensionsUsed",
        )
    });
    out.push(
        match root
            .get("asset")
            .and_then(|a| a.get("version"))
            .and_then(|v| v.as_str())
        {
            Some("2.0") => pass(Clause::AssetVersion),
            Some(other) => fail(
                Clause::AssetVersion,
                format!("asset.version is {other:?}, want \"2.0\""),
            ),
            None => fail(Clause::AssetVersion, "asset.version missing"),
        },
    );

    // ----- locate the first KHR-bearing primitive -----
    let primitive = root
        .get("meshes")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .and_then(|m| m.get("primitives"))
        .and_then(|p| p.as_array())
        .and_then(|p| p.first());

    let prim_ext_blob = primitive
        .and_then(|p| p.get("extensions"))
        .and_then(|e| e.get(KHR));

    let attrs_obj = primitive
        .and_then(|p| p.get("attributes"))
        .and_then(|a| a.as_object());

    out.push(match prim_ext_blob {
        Some(_) => pass(Clause::PrimitiveExtensionPresent),
        None => fail(
            Clause::PrimitiveExtensionPresent,
            "no primitive declares KHR_gaussian_splatting",
        ),
    });

    // PRIM_MODE_POINTS — primitive.mode must be 0 (POINTS).
    out.push(match primitive {
        None => fail(Clause::PrimitiveModePoints, "no primitive in mesh"),
        Some(p) => match p.get("mode").and_then(|v| v.as_u64()) {
            Some(0) => pass(Clause::PrimitiveModePoints),
            Some(other) => fail(
                Clause::PrimitiveModePoints,
                format!("primitive.mode={other}, MUST be 0 (POINTS)"),
            ),
            None => fail(
                Clause::PrimitiveModePoints,
                "primitive.mode missing (glTF default is 4 / TRIANGLES; spec requires POINTS)",
            ),
        },
    });

    // EXT_KERNEL, EXT_COLOR_SPACE, EXT_PROJECTION, EXT_SORTING.
    let kernel = prim_ext_blob.and_then(|e| e.get("kernel"));
    out.push(match kernel.and_then(|v| v.as_str()) {
        Some(_) => pass(Clause::ExtKernelRequired),
        None if kernel.is_some() => fail(
            Clause::ExtKernelRequired,
            "kernel present but not a string",
        ),
        None => fail(Clause::ExtKernelRequired, "kernel property missing"),
    });

    let color_space = prim_ext_blob.and_then(|e| e.get("colorSpace"));
    out.push(match color_space.and_then(|v| v.as_str()) {
        Some(_) => pass(Clause::ExtColorSpaceRequired),
        None if color_space.is_some() => fail(
            Clause::ExtColorSpaceRequired,
            "colorSpace present but not a string",
        ),
        None => fail(
            Clause::ExtColorSpaceRequired,
            "colorSpace property missing",
        ),
    });

    let projection = prim_ext_blob.and_then(|e| e.get("projection"));
    out.push(match projection {
        None => skip(
            Clause::ExtProjectionValid,
            "projection not declared (defaults to perspective)",
        ),
        Some(v) => match v.as_str() {
            Some(_) => pass(Clause::ExtProjectionValid),
            None => fail(
                Clause::ExtProjectionValid,
                "projection present but not a string",
            ),
        },
    });

    let sorting = prim_ext_blob.and_then(|e| e.get("sortingMethod"));
    out.push(match sorting {
        None => skip(
            Clause::ExtSortingValid,
            "sortingMethod not declared (defaults to cameraDistance)",
        ),
        Some(v) => match v.as_str() {
            Some(_) => pass(Clause::ExtSortingValid),
            None => fail(
                Clause::ExtSortingValid,
                "sortingMethod present but not a string",
            ),
        },
    });

    // ----- attribute presence -----
    let attr_idx = |name: &str| -> Option<usize> {
        attrs_obj
            .and_then(|a| a.get(name))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    };

    let names_required: &[(Clause, &str, bool)] = &[
        (Clause::PositionPresent, "POSITION", false),
        (Clause::RotationPresent, "ROTATION", true),
        (Clause::ScalePresent, "SCALE", true),
        (Clause::OpacityPresent, "OPACITY", true),
        (Clause::ShDcPresent, "SH_DEGREE_0_COEF_0", true),
    ];
    for (clause, name, namespaced) in names_required {
        let key = if *namespaced {
            format!("{NS}{name}")
        } else {
            (*name).to_string()
        };
        out.push(if attr_idx(&key).is_some() {
            pass(*clause)
        } else {
            fail(*clause, format!("attribute {key} missing"))
        });
    }

    // ----- accessor-level shape checks -----
    let accessors = root
        .get("accessors")
        .and_then(|a| a.as_array())
        .cloned()
        .unwrap_or_default();
    let buffer_views = root
        .get("bufferViews")
        .and_then(|b| b.as_array())
        .cloned()
        .unwrap_or_default();
    let buffers = root
        .get("buffers")
        .and_then(|b| b.as_array())
        .cloned()
        .unwrap_or_default();

    let check_acc =
        |clause: Clause, name: &str, want_type: &[&str], want_ct: &[u32]| -> ClauseResult {
            let Some(idx) = attr_idx(name) else {
                return skip(clause, format!("{name} accessor absent"));
            };
            let Some(acc) = accessors.get(idx) else {
                return fail(clause, format!("accessor index {idx} out of range"));
            };
            let ty = acc.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let ct = acc
                .get("componentType")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            if !want_type.contains(&ty) {
                return fail(
                    clause,
                    format!("{name}.type={ty:?}, want one of {want_type:?}"),
                );
            }
            if !want_ct.contains(&ct) {
                return fail(
                    clause,
                    format!("{name}.componentType={ct}, want one of {want_ct:?}"),
                );
            }
            pass(clause)
        };

    // glTF componentType values:
    // 5120 BYTE, 5121 UNSIGNED_BYTE, 5122 SHORT, 5123 UNSIGNED_SHORT,
    // 5125 UNSIGNED_INT, 5126 FLOAT.
    out.push(check_acc(
        Clause::PositionAccessor,
        "POSITION",
        &["VEC3"],
        &[5126, 5121, 5123],
    ));
    out.push(check_acc(
        Clause::RotationAccessor,
        &format!("{NS}ROTATION"),
        &["VEC4"],
        &[5126, 5120, 5122],
    ));
    out.push(check_acc(
        Clause::ScaleAccessor,
        &format!("{NS}SCALE"),
        &["VEC3"],
        &[5126, 5121, 5123],
    ));
    out.push(check_acc(
        Clause::OpacityAccessor,
        &format!("{NS}OPACITY"),
        &["SCALAR"],
        &[5126, 5121, 5123],
    ));

    // SH coefficient accessors — every SH_DEGREE_l_COEF_n declared MUST be VEC3 FLOAT.
    out.push({
        let mut problems: Vec<String> = Vec::new();
        let mut checked = 0usize;
        for l in 0..=3 {
            for n in 0..sh_coef_count(l) {
                let key = sh_attr_name(l, n);
                let Some(idx) = attr_idx(&key) else {
                    continue;
                };
                checked += 1;
                let Some(acc) = accessors.get(idx) else {
                    problems.push(format!("{key}: accessor {idx} out of range"));
                    continue;
                };
                let ty = acc.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let ct = acc
                    .get("componentType")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                if ty != "VEC3" {
                    problems.push(format!("{key}.type={ty:?}, want VEC3"));
                }
                if ct != 5126 {
                    problems.push(format!("{key}.componentType={ct}, want 5126 (FLOAT)"));
                }
            }
        }
        if checked == 0 {
            skip(Clause::ShCoefAccessor, "no SH coefficient accessors declared")
        } else if problems.is_empty() {
            pass(Clause::ShCoefAccessor)
        } else {
            fail(Clause::ShCoefAccessor, problems.join("; "))
        }
    });

    // POSITION min/max.
    out.push(match attr_idx("POSITION").and_then(|i| accessors.get(i)) {
        Some(acc) => {
            let has_min = acc
                .get("min")
                .and_then(|v| v.as_array())
                .map(|a| a.len() == 3)
                .unwrap_or(false);
            let has_max = acc
                .get("max")
                .and_then(|v| v.as_array())
                .map(|a| a.len() == 3)
                .unwrap_or(false);
            if has_min && has_max {
                pass(Clause::PositionMinMax)
            } else {
                fail(
                    Clause::PositionMinMax,
                    "POSITION accessor missing min and/or max",
                )
            }
        }
        None => skip(Clause::PositionMinMax, "POSITION accessor absent"),
    });

    // SH degrees fully defined.
    out.push({
        let mut highest_used: Option<usize> = None;
        let mut partial_problems: Vec<String> = Vec::new();
        for l in 0..=3 {
            let n_required = sh_coef_count(l);
            let n_present = (0..n_required)
                .filter(|n| attr_idx(&sh_attr_name(l, *n)).is_some())
                .count();
            if n_present == 0 {
                continue;
            }
            if n_present != n_required {
                partial_problems.push(format!(
                    "SH degree {l} partial: {n_present}/{n_required} coefficients present"
                ));
            } else {
                highest_used = Some(l);
            }
        }
        if let Some(top) = highest_used {
            for l in 0..top {
                let n_required = sh_coef_count(l);
                let missing: Vec<usize> = (0..n_required)
                    .filter(|n| attr_idx(&sh_attr_name(l, *n)).is_none())
                    .collect();
                if !missing.is_empty() {
                    partial_problems.push(format!(
                        "SH degree {l} missing coefs {missing:?} but degree {top} is in use"
                    ));
                }
            }
        }
        if partial_problems.is_empty() {
            pass(Clause::ShDegreesFullyDefined)
        } else {
            fail(Clause::ShDegreesFullyDefined, partial_problems.join("; "))
        }
    });

    // All per-splat accessors share count.
    out.push({
        let mut names: Vec<String> = vec![
            "POSITION".to_string(),
            format!("{NS}ROTATION"),
            format!("{NS}SCALE"),
            format!("{NS}OPACITY"),
        ];
        for l in 0..=3 {
            for n in 0..sh_coef_count(l) {
                names.push(sh_attr_name(l, n));
            }
        }
        let mut counts: Vec<(String, usize)> = Vec::new();
        for n in &names {
            if let Some(i) = attr_idx(n) {
                if let Some(acc) = accessors.get(i) {
                    let c = acc.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    counts.push((n.clone(), c));
                }
            }
        }
        if counts.is_empty() {
            skip(Clause::AccessorCountsAgree, "no per-splat accessors found")
        } else {
            let first = counts[0].1;
            if counts.iter().all(|(_, c)| *c == first) {
                pass(Clause::AccessorCountsAgree)
            } else {
                fail(
                    Clause::AccessorCountsAgree,
                    format!("counts disagree: {counts:?}"),
                )
            }
        }
    });

    // BufferView bounds.
    out.push({
        let mut problems: Vec<String> = Vec::new();
        for (i, acc) in accessors.iter().enumerate() {
            let bv_idx = acc.get("bufferView").and_then(|v| v.as_u64());
            let count = acc.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let ty = acc.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let ct = acc.get("componentType").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let comps = match ty {
                "SCALAR" => 1,
                "VEC2" => 2,
                "VEC3" => 3,
                "VEC4" => 4,
                _ => 0,
            };
            let bytes_per = match ct {
                5121 | 5120 => 1,
                5123 | 5122 => 2,
                5125 | 5126 => 4,
                _ => 0,
            };
            if comps == 0 || bytes_per == 0 || bv_idx.is_none() {
                continue;
            }
            let bv_idx = bv_idx.unwrap() as usize;
            let need = count * comps * bytes_per;
            let Some(bv) = buffer_views.get(bv_idx) else {
                problems.push(format!("accessor {i} bufferView {bv_idx} out of range"));
                continue;
            };
            let bv_len = bv.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let bv_off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let buf_idx = bv.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            if need > bv_len {
                problems.push(format!(
                    "accessor {i} needs {need} bytes but bufferView {bv_idx} has {bv_len}"
                ));
            }
            if let Some(buf) = buffers.get(buf_idx) {
                let buf_len = buf.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if bv_off + bv_len > buf_len {
                    problems.push(format!(
                        "bufferView {bv_idx} (off={bv_off}, len={bv_len}) overflows buffer {buf_idx} (len={buf_len})"
                    ));
                }
            } else {
                problems.push(format!("bufferView {bv_idx} refs missing buffer {buf_idx}"));
            }
        }
        if problems.is_empty() {
            pass(Clause::BufferViewBounds)
        } else {
            fail(Clause::BufferViewBounds, problems.join("; "))
        }
    });

    // Unknown KHR-namespaced attribute keys (the spec reserves the namespace).
    out.push(match attrs_obj {
        None => skip(Clause::NoUnknownNamespacedAttributes, "no attributes object"),
        Some(map) => {
            let known = known_namespaced_attrs();
            let unknown: Vec<&String> = map
                .keys()
                .filter(|k| k.starts_with(NS))
                .filter(|k| {
                    let suffix = &k[NS.len()..];
                    !known.iter().any(|kk| kk == suffix)
                })
                .collect();
            if unknown.is_empty() {
                pass(Clause::NoUnknownNamespacedAttributes)
            } else {
                fail(
                    Clause::NoUnknownNamespacedAttributes,
                    format!("unknown namespaced attributes: {unknown:?}"),
                )
            }
        }
    });

    // ----- KHR_gaussian_splatting_compression_spz clauses (7) -----
    let spz_ext_blob = primitive
        .and_then(|p| p.get("extensions"))
        .and_then(|e| e.get(SPZ));
    let primitive_declares_spz = spz_ext_blob.is_some();

    out.push(if !has_spz_used && !primitive_declares_spz {
        skip(Clause::SpzExtensionDeclared, "SPZ not present in asset")
    } else if primitive_declares_spz && !has_spz_used {
        fail(
            Clause::SpzExtensionDeclared,
            "primitive declares SPZ but extensionsUsed omits it",
        )
    } else {
        pass(Clause::SpzExtensionDeclared)
    });

    out.push(if !has_spz_used && !primitive_declares_spz {
        skip(Clause::SpzExtensionConsistent, "SPZ not present in asset")
    } else if primitive_declares_spz && prim_ext_blob.is_none() {
        fail(
            Clause::SpzExtensionConsistent,
            "primitive declares SPZ but not KHR_gaussian_splatting",
        )
    } else {
        pass(Clause::SpzExtensionConsistent)
    });

    let spz_in_use = has_spz_used || primitive_declares_spz;

    out.push(if !spz_in_use {
        skip(Clause::SpzExtPresent, "SPZ extension not declared")
    } else if spz_ext_blob.is_none() {
        fail(
            Clause::SpzExtPresent,
            "extensionsUsed contains SPZ but no primitive declares it",
        )
    } else {
        pass(Clause::SpzExtPresent)
    });

    out.push(if !spz_in_use {
        skip(Clause::SpzVersion, "SPZ extension not declared")
    } else {
        match spz_ext_blob
            .and_then(|e| e.get("version"))
            .and_then(|v| v.as_u64())
        {
            None => fail(Clause::SpzVersion, "SPZ extension missing version field"),
            Some(2) => pass(Clause::SpzVersion),
            Some(other) => fail(
                Clause::SpzVersion,
                format!("SPZ version={other}, want 2 (current SPZ wire format)"),
            ),
        }
    });

    // SPZ bufferView resolution, reused by magic + count clauses.
    let bv_resolved: Option<(usize, usize, usize)> = spz_ext_blob
        .and_then(|e| e.get("bufferView"))
        .and_then(|v| v.as_u64())
        .map(|idx| idx as usize)
        .and_then(|idx| {
            buffer_views.get(idx).map(|bv| {
                let off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let len = bv.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let buf = bv.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                (buf, off, len)
            })
        });

    out.push(if !spz_in_use {
        skip(Clause::SpzBufferView, "SPZ extension not declared")
    } else {
        match spz_ext_blob
            .and_then(|e| e.get("bufferView"))
            .and_then(|v| v.as_u64())
        {
            None => fail(Clause::SpzBufferView, "SPZ extension missing bufferView"),
            Some(idx) => {
                let idx = idx as usize;
                match buffer_views.get(idx) {
                    None => fail(
                        Clause::SpzBufferView,
                        format!("SPZ bufferView {idx} out of range"),
                    ),
                    Some(bv) => {
                        let off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0)
                            as usize;
                        let len = bv.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0)
                            as usize;
                        let buf_idx =
                            bv.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let buf_len = buffers
                            .get(buf_idx)
                            .and_then(|b| b.get("byteLength"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        if off + len > buf_len {
                            fail(
                                Clause::SpzBufferView,
                                format!(
                                    "SPZ bufferView (off={off}, len={len}) overflows buffer {buf_idx} (len={buf_len})"
                                ),
                            )
                        } else if len < 16 {
                            fail(
                                Clause::SpzBufferView,
                                format!("SPZ bufferView byteLength={len} < 16 (SPZ header size)"),
                            )
                        } else {
                            pass(Clause::SpzBufferView)
                        }
                    }
                }
            }
        }
    });

    out.push(if !spz_in_use {
        skip(Clause::SpzBlobMagic, "SPZ extension not declared")
    } else {
        match (bv_resolved, bin) {
            (Some((0, off, len)), Some(bin_bytes)) => {
                if off + len > bin_bytes.len() || len < 4 {
                    fail(
                        Clause::SpzBlobMagic,
                        format!("SPZ bufferView (off={off}, len={len}) does not fit in BIN chunk"),
                    )
                } else {
                    let m = &bin_bytes[off..off + 4];
                    // SPZ_MAGIC = 0x5053_4E47 LE => bytes [0x47, 0x4e, 0x53, 0x50] = "GNSP".
                    let want = [0x47u8, 0x4e, 0x53, 0x50];
                    if m == want {
                        pass(Clause::SpzBlobMagic)
                    } else {
                        fail(
                            Clause::SpzBlobMagic,
                            format!(
                                "SPZ blob magic={:02x?}, want {:02x?} (0x5053_4e47 LE)",
                                m, want
                            ),
                        )
                    }
                }
            }
            (Some(_), None) => skip(
                Clause::SpzBlobMagic,
                "SPZ blob magic not checkable on .gltf (no BIN chunk)",
            ),
            (Some((buf, _, _)), Some(_)) => skip(
                Clause::SpzBlobMagic,
                format!("SPZ bufferView refers to buffer {buf}, not the GLB BIN buffer"),
            ),
            (None, _) => skip(
                Clause::SpzBlobMagic,
                "SPZ bufferView did not resolve to a buffer range",
            ),
        }
    });

    out.push(if !spz_in_use {
        skip(Clause::SpzDecodedCount, "SPZ extension not declared")
    } else {
        let declared = spz_ext_blob
            .and_then(|e| e.get("splatCount"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| {
                attr_idx(&format!("{NS}OPACITY"))
                    .and_then(|i| accessors.get(i))
                    .and_then(|a| a.get("count"))
                    .and_then(|c| c.as_u64())
                    .map(|n| n as usize)
                    .filter(|n| *n > 0)
            });
        match (bv_resolved, bin, declared) {
            (Some((0, off, len)), Some(bin_bytes), Some(want))
                if off + 12 <= bin_bytes.len() && len >= 12 =>
            {
                let header = &bin_bytes[off..off + 12];
                let count = u32::from_le_bytes([
                    header[8],
                    header[9],
                    header[10],
                    header[11],
                ]) as usize;
                if count == want {
                    pass(Clause::SpzDecodedCount)
                } else {
                    fail(
                        Clause::SpzDecodedCount,
                        format!("SPZ header splat_count={count} but primitive declares {want}"),
                    )
                }
            }
            (_, _, None) => skip(
                Clause::SpzDecodedCount,
                "no primitive splatCount available to compare against",
            ),
            (Some(_), None, _) => skip(
                Clause::SpzDecodedCount,
                "SPZ blob splat_count not checkable on .gltf (no BIN chunk)",
            ),
            (None, _, _) => skip(Clause::SpzDecodedCount, "SPZ bufferView did not resolve"),
            _ => skip(Clause::SpzDecodedCount, "SPZ blob too small for header"),
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> serde_json::Value {
        // 4 splats, FLOAT accessors only.
        //   POSITION    VEC3 FLOAT 4*12 = 48
        //   ROTATION    VEC4 FLOAT 4*16 = 64
        //   SCALE       VEC3 FLOAT 4*12 = 48
        //   OPACITY     SCAL FLOAT 4*4  = 16
        //   SH_DC       VEC3 FLOAT 4*12 = 48
        // Total: 224.
        serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_gaussian_splatting"],
            "buffers": [{ "byteLength": 224 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0,   "byteLength": 48 },
                { "buffer": 0, "byteOffset": 48,  "byteLength": 64 },
                { "buffer": 0, "byteOffset": 112, "byteLength": 48 },
                { "buffer": 0, "byteOffset": 160, "byteLength": 16 },
                { "buffer": 0, "byteOffset": 176, "byteLength": 48 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                  "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0] },
                { "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC4" },
                { "bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC3" },
                { "bufferView": 3, "componentType": 5126, "count": 4, "type": "SCALAR" },
                { "bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "mode": 0,
                    "attributes": {
                        "POSITION": 0,
                        "KHR_gaussian_splatting:ROTATION": 1,
                        "KHR_gaussian_splatting:SCALE": 2,
                        "KHR_gaussian_splatting:OPACITY": 3,
                        "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 4
                    },
                    "extensions": {
                        "KHR_gaussian_splatting": {
                            "kernel": "ellipse",
                            "colorSpace": "srgb_rec709_display"
                        }
                    }
                }]
            }]
        })
    }

    #[test]
    fn baseline_passes() {
        let r = validate_json(&valid_json(), "test");
        let fails: Vec<_> = r
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect();
        assert!(r.is_pass(), "expected pass, got fails: {fails:?}");
    }

    #[test]
    fn missing_ext_used_fails() {
        let mut j = valid_json();
        j["extensionsUsed"] = serde_json::json!([]);
        let r = validate_json(&j, "test");
        assert!(!r.is_pass());
        assert!(r
            .clauses
            .iter()
            .any(|c| c.id == "EXT_USED" && c.status == Status::Fail));
    }

    #[test]
    fn wrong_mode_fails() {
        let mut j = valid_json();
        j["meshes"][0]["primitives"][0]["mode"] = serde_json::json!(4);
        let r = validate_json(&j, "test");
        assert!(r
            .clauses
            .iter()
            .any(|c| c.id == "PRIM_MODE_POINTS" && c.status == Status::Fail));
    }

    #[test]
    fn missing_kernel_fails() {
        let mut j = valid_json();
        let ext = j["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]
            .as_object_mut()
            .unwrap();
        ext.remove("kernel");
        let r = validate_json(&j, "test");
        assert!(r
            .clauses
            .iter()
            .any(|c| c.id == "EXT_KERNEL" && c.status == Status::Fail));
    }
}
