#![deny(clippy::all)]
//! Conformance test suite for the Khronos `KHR_gaussian_splatting` glTF
//! extension.
//!
//! Each public [`Clause`] corresponds to one normative requirement in the
//! KHR_gaussian_splatting RC text (and its optional sub-extension
//! `KHR_gaussian_splatting_compression_spz`).  The validator loads a glTF
//! 2.0 JSON document — either as an external `.gltf` or extracted from the
//! JSON chunk of a `.glb` container — and returns a [`Report`] that says
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
    ExtRequired,
    AssetVersion,
    PrimitiveExtensionPresent,
    AttributesObjectPresent,
    PositionPresent,
    RotationPresent,
    ScalePresent,
    OpacityPresent,
    ColorDcPresent,
    PositionAccessor,
    RotationAccessor,
    ScaleAccessor,
    OpacityAccessor,
    ColorDcAccessor,
    ColorShAccessor,
    PositionMinMax,
    ShDegreeRange,
    AccessorCountsAgree,
    BufferViewBounds,
    SpzExtensionDeclared,
    SpzExtensionConsistent,
    NoUnknownAttributes,
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
            Clause::ExtRequired,
            Clause::AssetVersion,
            Clause::PrimitiveExtensionPresent,
            Clause::AttributesObjectPresent,
            Clause::PositionPresent,
            Clause::RotationPresent,
            Clause::ScalePresent,
            Clause::OpacityPresent,
            Clause::ColorDcPresent,
            Clause::PositionAccessor,
            Clause::RotationAccessor,
            Clause::ScaleAccessor,
            Clause::OpacityAccessor,
            Clause::ColorDcAccessor,
            Clause::ColorShAccessor,
            Clause::PositionMinMax,
            Clause::ShDegreeRange,
            Clause::AccessorCountsAgree,
            Clause::BufferViewBounds,
            Clause::SpzExtensionDeclared,
            Clause::SpzExtensionConsistent,
            Clause::NoUnknownAttributes,
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
            Clause::ExtRequired => "EXT_REQUIRED",
            Clause::AssetVersion => "ASSET_VERSION",
            Clause::PrimitiveExtensionPresent => "PRIM_EXT",
            Clause::AttributesObjectPresent => "ATTRS_OBJECT",
            Clause::PositionPresent => "ATTR_POSITION",
            Clause::RotationPresent => "ATTR_ROTATION",
            Clause::ScalePresent => "ATTR_SCALE",
            Clause::OpacityPresent => "ATTR_OPACITY",
            Clause::ColorDcPresent => "ATTR_COLOR_DC",
            Clause::PositionAccessor => "ACC_POSITION",
            Clause::RotationAccessor => "ACC_ROTATION",
            Clause::ScaleAccessor => "ACC_SCALE",
            Clause::OpacityAccessor => "ACC_OPACITY",
            Clause::ColorDcAccessor => "ACC_COLOR_DC",
            Clause::ColorShAccessor => "ACC_COLOR_SH",
            Clause::PositionMinMax => "ACC_POSITION_MINMAX",
            Clause::ShDegreeRange => "SH_DEGREE_RANGE",
            Clause::AccessorCountsAgree => "ACC_COUNTS_AGREE",
            Clause::BufferViewBounds => "BUFFERVIEW_BOUNDS",
            Clause::SpzExtensionDeclared => "SPZ_DECLARED",
            Clause::SpzExtensionConsistent => "SPZ_CONSISTENT",
            Clause::NoUnknownAttributes => "ATTRS_KNOWN_ONLY",
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
            Clause::ExtUsed => "Root extensionsUsed array MUST list \"KHR_gaussian_splatting\".",
            Clause::ExtRequired => {
                "extensionsRequired SHOULD list \"KHR_gaussian_splatting\" \
                 when the asset cannot render without it."
            }
            Clause::AssetVersion => "asset.version MUST be \"2.0\" per glTF 2.0.",
            Clause::PrimitiveExtensionPresent => {
                "At least one mesh.primitives[i].extensions[\"KHR_gaussian_splatting\"] \
                 block MUST be present."
            }
            Clause::AttributesObjectPresent => {
                "The KHR_gaussian_splatting block on a primitive MUST contain an \
                 \"attributes\" object."
            }
            Clause::PositionPresent => "The attributes object MUST declare a POSITION accessor.",
            Clause::RotationPresent => "The attributes object MUST declare a _ROTATION accessor.",
            Clause::ScalePresent => "The attributes object MUST declare a _SCALE accessor.",
            Clause::OpacityPresent => "The attributes object MUST declare a _OPACITY accessor.",
            Clause::ColorDcPresent => "The attributes object MUST declare a _COLOR_DC accessor.",
            Clause::PositionAccessor => {
                "POSITION accessor MUST be VEC3 (componentType FLOAT, or normalized \
                 UNSIGNED_SHORT/UNSIGNED_BYTE under KHR_mesh_quantization)."
            }
            Clause::RotationAccessor => {
                "_ROTATION accessor MUST be VEC4 FLOAT (unit quaternion xyzw)."
            }
            Clause::ScaleAccessor => "_SCALE accessor MUST be VEC3 (FLOAT or normalized integer).",
            Clause::OpacityAccessor => {
                "_OPACITY accessor MUST be SCALAR (FLOAT or normalized integer in [0,1])."
            }
            Clause::ColorDcAccessor => {
                "_COLOR_DC accessor MUST be VEC3 (FLOAT or normalized integer in [0,1])."
            }
            Clause::ColorShAccessor => {
                "When present, _COLOR_SH accessor MUST be SCALAR FLOAT with 45 \
                 elements per splat (count = splat_count * 45)."
            }
            Clause::PositionMinMax => {
                "POSITION accessor MUST provide both min and max arrays (glTF 2.0 §3.6.2.4)."
            }
            Clause::ShDegreeRange => {
                "shDegree MUST be an integer in [0, 3]; when _COLOR_SH is absent \
                 it MUST be 0."
            }
            Clause::AccessorCountsAgree => {
                "All per-splat accessors (POSITION, _ROTATION, _SCALE, _OPACITY, \
                 _COLOR_DC) MUST share the same count."
            }
            Clause::BufferViewBounds => {
                "Every referenced accessor's bufferView MUST be in range and its byte \
                 footprint MUST fit inside the parent buffer."
            }
            Clause::SpzExtensionDeclared => {
                "If KHR_gaussian_splatting_compression_spz appears anywhere in the \
                 asset it MUST be listed in extensionsUsed."
            }
            Clause::SpzExtensionConsistent => {
                "If a primitive declares KHR_gaussian_splatting_compression_spz it \
                 MUST also declare KHR_gaussian_splatting on the same primitive."
            }
            Clause::NoUnknownAttributes => {
                "The KHR_gaussian_splatting attributes object MUST NOT contain unknown \
                 attribute keys (only POSITION, _ROTATION, _SCALE, _OPACITY, _COLOR_DC, \
                 _COLOR_SH are reserved)."
            }
            Clause::SpzExtPresent => {
                "When KHR_gaussian_splatting_compression_spz is in extensionsUsed, at \
                 least one primitive MUST declare it under its own extensions object."
            }
            Clause::SpzVersion => {
                "KHR_gaussian_splatting_compression_spz.version MUST be 2 (the SPZ wire \
                 format version)."
            }
            Clause::SpzBufferView => {
                "KHR_gaussian_splatting_compression_spz.bufferView MUST be an integer in \
                 range of bufferViews[] and the view MUST fit within its buffer."
            }
            Clause::SpzBlobMagic => {
                "The bytes referenced by the SPZ bufferView MUST start with the SPZ magic \
                 (0x5053_4e47)."
            }
            Clause::SpzDecodedCount => {
                "The splat_count decoded from the SPZ header MUST equal the primitive's \
                 declared splatCount (extension or _OPACITY accessor count)."
            }
        }
    }

    /// Whether this clause is a MUST (true) or SHOULD/optional (false).
    pub fn is_mandatory(self) -> bool {
        !matches!(self, Clause::ExtRequired)
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
    /// Clause did not apply to this asset (e.g. SPZ checks on a non-SPZ file).
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
    /// True when no clause failed (skips are tolerated — SPZ checks are
    /// expected to skip on non-SPZ fixtures, for example).
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
const SPZ: &str = "KHR_gaussian_splatting_compression_spz";

/// Validate a `.gltf` or `.glb` file and return the conformance report.
pub fn validate_path(path: &Path) -> Result<Report, ValidateError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let bytes = fs::read(path)?;
    let (json_str, container, bin): (String, &str, Option<Vec<u8>>) = match ext.as_str() {
        "gltf" => (String::from_utf8_lossy(&bytes).to_string(), "gltf", None),
        "glb" => {
            let (j, b) = extract_glb_parts(&bytes)?;
            (j, "glb", b)
        }
        other => return Err(ValidateError::UnsupportedExt(other.to_string())),
    };
    let value: serde_json::Value = serde_json::from_str(&json_str)?;
    let clauses = run_clauses(&value, bin.as_deref());
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
    Ok(Report {
        source: path.display().to_string(),
        container: container.to_string(),
        pass,
        fail,
        skip,
        clauses,
    })
}

/// Validate a glTF JSON document already in memory. Useful for tests and for
/// callers who have already extracted the JSON chunk from a GLB.
pub fn validate_json(json: &serde_json::Value, source: &str) -> Report {
    let clauses = run_clauses(json, None);
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
        source: source.to_string(),
        container: "gltf".to_string(),
        pass,
        fail,
        skip,
        clauses,
    }
}

/// Extract both the JSON chunk (as a UTF-8 string) and the BIN chunk (raw
/// bytes) from a GLB container. The BIN chunk is needed for the SPZ
/// blob-magic and decoded-count clauses.
fn extract_glb_parts(bytes: &[u8]) -> Result<(String, Option<Vec<u8>>), ValidateError> {
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

fn run_clauses(root: &serde_json::Value, bin: Option<&[u8]>) -> Vec<ClauseResult> {
    let mut out = Vec::with_capacity(Clause::all().len());

    // ----- root-level extension declaration -----
    let used: Vec<&str> = root
        .get("extensionsUsed")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    let required: Vec<&str> = root
        .get("extensionsRequired")
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
    out.push(if required.contains(&KHR) {
        pass(Clause::ExtRequired)
    } else {
        fail(
            Clause::ExtRequired,
            "KHR_gaussian_splatting not in extensionsRequired",
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

    // ----- find the KHR-bearing primitive -----
    let prim_ext_blob = root
        .get("meshes")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .and_then(|m| m.get("primitives"))
        .and_then(|p| p.as_array())
        .and_then(|p| p.first())
        .and_then(|p| p.get("extensions"))
        .and_then(|e| e.get(KHR));

    let attrs_obj = prim_ext_blob
        .and_then(|e| e.get("attributes"))
        .and_then(|a| a.as_object());

    out.push(match prim_ext_blob {
        Some(_) => pass(Clause::PrimitiveExtensionPresent),
        None => fail(
            Clause::PrimitiveExtensionPresent,
            "no primitive declares KHR_gaussian_splatting",
        ),
    });
    out.push(match attrs_obj {
        Some(_) => pass(Clause::AttributesObjectPresent),
        None => fail(
            Clause::AttributesObjectPresent,
            "no attributes object on KHR block",
        ),
    });

    let attr_idx = |name: &str| -> Option<usize> {
        attrs_obj
            .and_then(|a| a.get(name))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    };

    let names_required: &[(Clause, &str)] = &[
        (Clause::PositionPresent, "POSITION"),
        (Clause::RotationPresent, "_ROTATION"),
        (Clause::ScalePresent, "_SCALE"),
        (Clause::OpacityPresent, "_OPACITY"),
        (Clause::ColorDcPresent, "_COLOR_DC"),
    ];
    for (clause, name) in names_required {
        out.push(if attr_idx(name).is_some() {
            pass(*clause)
        } else {
            fail(*clause, format!("attribute {name} missing"))
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

    // FLOAT=5126, UBYTE=5121, USHORT=5123
    out.push(check_acc(
        Clause::PositionAccessor,
        "POSITION",
        &["VEC3"],
        &[5126, 5121, 5123],
    ));
    out.push(check_acc(
        Clause::RotationAccessor,
        "_ROTATION",
        &["VEC4"],
        &[5126],
    ));
    out.push(check_acc(
        Clause::ScaleAccessor,
        "_SCALE",
        &["VEC3"],
        &[5126, 5121, 5123],
    ));
    out.push(check_acc(
        Clause::OpacityAccessor,
        "_OPACITY",
        &["SCALAR"],
        &[5126, 5121, 5123],
    ));
    out.push(check_acc(
        Clause::ColorDcAccessor,
        "_COLOR_DC",
        &["VEC3"],
        &[5126, 5121, 5123],
    ));

    // _COLOR_SH is optional, but if present must be SCALAR FLOAT with 45*N.
    out.push(match attr_idx("_COLOR_SH") {
        None => skip(Clause::ColorShAccessor, "_COLOR_SH not declared"),
        Some(idx) => match accessors.get(idx) {
            None => fail(
                Clause::ColorShAccessor,
                format!("accessor {idx} out of range"),
            ),
            Some(acc) => {
                let ty = acc.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let ct = acc
                    .get("componentType")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let count = acc.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let pos_count = attr_idx("POSITION")
                    .and_then(|i| accessors.get(i))
                    .and_then(|a| a.get("count"))
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as usize;
                if ty != "SCALAR" {
                    fail(
                        Clause::ColorShAccessor,
                        format!("_COLOR_SH.type={ty:?}, want SCALAR"),
                    )
                } else if ct != 5126 {
                    fail(
                        Clause::ColorShAccessor,
                        format!("_COLOR_SH.componentType={ct}, want 5126 (FLOAT)"),
                    )
                } else if pos_count > 0 && count != pos_count * 45 {
                    fail(
                        Clause::ColorShAccessor,
                        format!(
                            "_COLOR_SH.count={count}, want {} (45 * splat_count)",
                            pos_count * 45
                        ),
                    )
                } else {
                    pass(Clause::ColorShAccessor)
                }
            }
        },
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

    // shDegree range, both at primitive level and (if present) scene level.
    out.push({
        let prim_sh = prim_ext_blob
            .and_then(|e| e.get("shDegree"))
            .and_then(|v| v.as_u64());
        let has_sh = attr_idx("_COLOR_SH").is_some();
        match prim_sh {
            None => fail(
                Clause::ShDegreeRange,
                "shDegree missing on primitive KHR block",
            ),
            Some(d) if d > 3 => fail(
                Clause::ShDegreeRange,
                format!("shDegree={d}, must be 0..=3"),
            ),
            Some(d) if !has_sh && d != 0 => fail(
                Clause::ShDegreeRange,
                format!("shDegree={d} but _COLOR_SH is absent (must be 0)"),
            ),
            Some(_) => pass(Clause::ShDegreeRange),
        }
    });

    // All per-splat accessors share count.
    out.push({
        let names = ["POSITION", "_ROTATION", "_SCALE", "_OPACITY", "_COLOR_DC"];
        let mut counts: Vec<(String, usize)> = Vec::new();
        for n in names {
            if let Some(i) = attr_idx(n) {
                if let Some(acc) = accessors.get(i) {
                    let c = acc.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    counts.push((n.to_string(), c));
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

    // SPZ checks.
    let primitive_declares_spz = root
        .get("meshes")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .and_then(|m| m.get("primitives"))
        .and_then(|p| p.as_array())
        .and_then(|p| p.first())
        .and_then(|p| p.get("extensions"))
        .and_then(|e| e.get(SPZ))
        .is_some();

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

    // Unknown attribute keys.
    out.push(match attrs_obj {
        None => skip(Clause::NoUnknownAttributes, "no attributes object"),
        Some(map) => {
            let known: &[&str] = &[
                "POSITION",
                "_ROTATION",
                "_SCALE",
                "_OPACITY",
                "_COLOR_DC",
                "_COLOR_SH",
            ];
            let unknown: Vec<&String> = map
                .keys()
                .filter(|k| !known.iter().any(|kk| kk == &k.as_str()))
                .collect();
            if unknown.is_empty() {
                pass(Clause::NoUnknownAttributes)
            } else {
                fail(
                    Clause::NoUnknownAttributes,
                    format!("unknown attributes: {unknown:?}"),
                )
            }
        }
    });

    // ----- KHR_gaussian_splatting_compression_spz clauses (5) -----
    //
    // These supplement SPZ_DECLARED / SPZ_CONSISTENT above and target the
    // body of the SPZ extension itself: version field, bufferView reference,
    // SPZ-blob magic, and decoded-count agreement.
    let spz_ext_blob = root
        .get("meshes")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .and_then(|m| m.get("primitives"))
        .and_then(|p| p.as_array())
        .and_then(|p| p.first())
        .and_then(|p| p.get("extensions"))
        .and_then(|e| e.get(SPZ));
    let spz_in_use = has_spz_used || spz_ext_blob.is_some();

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
        match spz_ext_blob.and_then(|e| e.get("version")).and_then(|v| v.as_u64()) {
            None => fail(Clause::SpzVersion, "SPZ extension missing version field"),
            Some(2) => pass(Clause::SpzVersion),
            Some(other) => fail(
                Clause::SpzVersion,
                format!("SPZ version={other}, want 2 (current SPZ wire format)"),
            ),
        }
    });

    // Compute the SPZ bufferView byte range up front; reused by magic + count clauses.
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
        match spz_ext_blob.and_then(|e| e.get("bufferView")).and_then(|v| v.as_u64()) {
            None => fail(Clause::SpzBufferView, "SPZ extension missing bufferView"),
            Some(idx) => {
                let idx = idx as usize;
                match buffer_views.get(idx) {
                    None => fail(
                        Clause::SpzBufferView,
                        format!("SPZ bufferView {idx} out of range"),
                    ),
                    Some(bv) => {
                        let off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let len = bv.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
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
                    // SPZ_MAGIC = 0x5053_4e47 little-endian => bytes [0x47, 0x4e, 0x53, 0x50] = "GNSP".
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
        // The extension's declared splat count. Fall back to the _OPACITY
        // accessor count when the extension omits the field (allowed but
        // discouraged per the spec).
        let declared = spz_ext_blob
            .and_then(|e| e.get("splatCount"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| {
                attr_idx("_OPACITY")
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
                // SPZ header: u32 magic, u32 version, u32 splat_count (LE).
                let header = &bin_bytes[off..off + 12];
                let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]])
                    as usize;
                if count == want {
                    pass(Clause::SpzDecodedCount)
                } else {
                    fail(
                        Clause::SpzDecodedCount,
                        format!(
                            "SPZ header splat_count={count} but primitive declares {want}"
                        ),
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
            (None, _, _) => skip(
                Clause::SpzDecodedCount,
                "SPZ bufferView did not resolve",
            ),
            _ => skip(Clause::SpzDecodedCount, "SPZ blob too small for header"),
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> serde_json::Value {
        serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_gaussian_splatting"],
            "extensionsRequired": ["KHR_gaussian_splatting"],
            "buffers": [{ "byteLength": 188 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0,  "byteLength": 24 },
                { "buffer": 0, "byteOffset": 24, "byteLength": 32 },
                { "buffer": 0, "byteOffset": 56, "byteLength": 24 },
                { "buffer": 0, "byteOffset": 80, "byteLength": 8  },
                { "buffer": 0, "byteOffset": 88, "byteLength": 24 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3",
                  "min": [0.0,0.0,0.0], "max": [1.0,1.0,1.0] },
                { "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC4" },
                { "bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3" },
                { "bufferView": 3, "componentType": 5126, "count": 2, "type": "SCALAR" },
                { "bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "extensions": {
                        "KHR_gaussian_splatting": {
                            "attributes": {
                                "POSITION": 0, "_ROTATION": 1, "_SCALE": 2,
                                "_OPACITY": 3, "_COLOR_DC": 4
                            },
                            "shDegree": 0
                        }
                    }
                }]
            }]
        })
    }

    #[test]
    fn baseline_passes() {
        let r = validate_json(&valid_json(), "test");
        assert!(r.is_pass(), "expected pass, got {:?}", r.clauses);
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
    fn wrong_rotation_type_fails() {
        let mut j = valid_json();
        j["accessors"][1]["type"] = serde_json::json!("VEC3");
        let r = validate_json(&j, "test");
        assert!(r
            .clauses
            .iter()
            .any(|c| c.id == "ACC_ROTATION" && c.status == Status::Fail));
    }
}
