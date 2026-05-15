#![deny(clippy::all)]
//! Conformance test suite for the OpenUSD 26.03 `ParticleField3DGaussianSplat`
//! schema.
//!
//! Each public [`Clause`] corresponds to one normative requirement we
//! extracted from the OpenUSD 26.03 schema definition (`pxr/usd/usdGeom/`
//! in PixarAnimationStudios/OpenUSD), the SPEC-0011/0012 SplatForge
//! mapping notes, and the inherited `GeomPoints` schema chain.  The
//! validator loads either a `.usda` text file or a `.usdc` binary crate
//! file (decoded via [`splatforge_usd::read_usdc`] then re-emitted as
//! USDA for inspection) and returns a [`Report`] saying whether every
//! clause passed, failed, or was skipped (not applicable).
//!
//! The report is JSON-serialisable so the same code drives Rust
//! integration tests, the `splatforge-usd-validate` CLI binary, and the
//! `splatforge spec-check` shell-out from the top-level CLI.
//!
//! Where the schema is ambiguous (see `crates/splatforge-usd/SPEC-GAPS.md`)
//! the validator picks the strictest defensible reading and documents the
//! choice on the [`Clause`] itself, so a future schema clarification can
//! flip the bit without breaking the public report contract.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifier for every spec clause the validator can check.
///
/// String forms (`"PRIM_PARTICLE_FIELD"`, …) are stable and part of the
/// public JSON report contract — renaming one is a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Clause {
    UsdaMagic,
    PrimParticleField,
    AttrPointsPresent,
    AttrPointsType,
    AttrOrientationsPresent,
    AttrOrientationsType,
    AttrScalesPresent,
    AttrScalesType,
    AttrOpacitiesPresent,
    AttrOpacitiesRange,
    AttrColorsDcPresent,
    AttrColorsDcRange,
    AttrWidthsOptional,
    AttrVelocitiesOptional,
    CountsAgree,
    ExtentConsistent,
    UpAxisValid,
    MetersPerUnitPositive,
    QuaternionsNormalized,
    RootXform,
    ShCoefficientsCount,
    DisplayColorInterpolation,
    SchemaRequiredAttrs,
}

impl Clause {
    /// All clauses in spec order. Used by the CLI to render the report.
    pub fn all() -> &'static [Clause] {
        &[
            Clause::UsdaMagic,
            Clause::PrimParticleField,
            Clause::RootXform,
            Clause::UpAxisValid,
            Clause::MetersPerUnitPositive,
            Clause::AttrPointsPresent,
            Clause::AttrPointsType,
            Clause::AttrOrientationsPresent,
            Clause::AttrOrientationsType,
            Clause::AttrScalesPresent,
            Clause::AttrScalesType,
            Clause::AttrOpacitiesPresent,
            Clause::AttrOpacitiesRange,
            Clause::AttrColorsDcPresent,
            Clause::AttrColorsDcRange,
            Clause::AttrWidthsOptional,
            Clause::AttrVelocitiesOptional,
            Clause::CountsAgree,
            Clause::ExtentConsistent,
            Clause::QuaternionsNormalized,
            Clause::ShCoefficientsCount,
            Clause::DisplayColorInterpolation,
            Clause::SchemaRequiredAttrs,
        ]
    }

    /// Short, stable identifier used in JSON output and CLI tables.
    pub fn id(self) -> &'static str {
        match self {
            Clause::UsdaMagic => "USDA_MAGIC",
            Clause::PrimParticleField => "PRIM_PARTICLE_FIELD",
            Clause::RootXform => "ROOT_XFORM",
            Clause::UpAxisValid => "UP_AXIS_VALID",
            Clause::MetersPerUnitPositive => "METERS_PER_UNIT_POSITIVE",
            Clause::AttrPointsPresent => "ATTR_POINTS",
            Clause::AttrPointsType => "ATTR_POINTS_TYPE",
            Clause::AttrOrientationsPresent => "ATTR_ORIENTATIONS",
            Clause::AttrOrientationsType => "ATTR_ORIENTATIONS_TYPE",
            Clause::AttrScalesPresent => "ATTR_SCALES",
            Clause::AttrScalesType => "ATTR_SCALES_TYPE",
            Clause::AttrOpacitiesPresent => "ATTR_OPACITIES",
            Clause::AttrOpacitiesRange => "ATTR_OPACITIES_RANGE",
            Clause::AttrColorsDcPresent => "ATTR_COLORS_DC",
            Clause::AttrColorsDcRange => "ATTR_COLORS_DC_RANGE",
            Clause::AttrWidthsOptional => "ATTR_WIDTHS_OPTIONAL",
            Clause::AttrVelocitiesOptional => "ATTR_VELOCITIES_OPTIONAL",
            Clause::CountsAgree => "COUNTS_AGREE",
            Clause::ExtentConsistent => "EXTENT_CONSISTENT",
            Clause::QuaternionsNormalized => "QUATS_NORMALIZED",
            Clause::ShCoefficientsCount => "SH_COEFFS_COUNT",
            Clause::DisplayColorInterpolation => "DISPLAYCOLOR_INTERP",
            Clause::SchemaRequiredAttrs => "SCHEMA_REQUIRED_ATTRS",
        }
    }

    /// Human-readable description, suitable for the conformance.md table.
    pub fn description(self) -> &'static str {
        match self {
            Clause::UsdaMagic => "File MUST begin with `#usda 1.0` magic line.",
            Clause::PrimParticleField => {
                "At least one `def ParticleField3DGaussianSplat` prim MUST be present."
            }
            Clause::RootXform => {
                "ParticleField3DGaussianSplat prim SHOULD be a descendant of a root `def Xform`."
            }
            Clause::UpAxisValid => {
                "Layer metadata `upAxis`, when authored, MUST be \"Y\" or \"Z\" per UsdGeomTokens."
            }
            Clause::MetersPerUnitPositive => {
                "Layer metadata `metersPerUnit`, when authored, MUST be a positive number."
            }
            Clause::AttrPointsPresent => {
                "ParticleField3DGaussianSplat MUST author `points` (inherited from GeomPoints)."
            }
            Clause::AttrPointsType => "`points` attribute MUST be typed `point3f[]`.",
            Clause::AttrOrientationsPresent => {
                "ParticleField3DGaussianSplat MUST author `orientations`."
            }
            Clause::AttrOrientationsType => "`orientations` attribute MUST be typed `quatf[]`.",
            Clause::AttrScalesPresent => "ParticleField3DGaussianSplat MUST author `scales`.",
            Clause::AttrScalesType => "`scales` attribute MUST be typed `float3[]`.",
            Clause::AttrOpacitiesPresent => "ParticleField3DGaussianSplat MUST author `opacities`.",
            Clause::AttrOpacitiesRange => {
                "`opacities` values MUST lie in [0, 1] (post-sigmoid convention — \
                 see SPEC-GAPS #4)."
            }
            Clause::AttrColorsDcPresent => "ParticleField3DGaussianSplat MUST author `colorsDC`.",
            Clause::AttrColorsDcRange => {
                "`colorsDC` values MUST lie in [0, 1] per `color3f` convention."
            }
            Clause::AttrWidthsOptional => {
                "When authored, `widths` (inherited from GeomPoints) MUST be typed `float[]`."
            }
            Clause::AttrVelocitiesOptional => {
                "When authored, `velocities` (inherited from GeomPoints) MUST be typed \
                 `vector3f[]`."
            }
            Clause::CountsAgree => {
                "Lengths of `points`, `orientations`, `scales`, `opacities`, `colorsDC` \
                 MUST agree (one element per splat)."
            }
            Clause::ExtentConsistent => {
                "When authored, `extent` (2 × float3) MUST enclose every authored point."
            }
            Clause::QuaternionsNormalized => {
                "`orientations` quaternions MUST be unit-length within 1e-3 tolerance."
            }
            Clause::ShCoefficientsCount => {
                "When authored, `custom float[] splatforge:shCoefficients` count MUST equal \
                 `splat_count * (degree+1)^2 * 3` for degree ∈ {0,1,2,3}."
            }
            Clause::DisplayColorInterpolation => {
                "When authored, `primvars:displayColor:interpolation` MUST be one of \
                 \"vertex\" or \"varying\" (see SPEC-GAPS #10)."
            }
            Clause::SchemaRequiredAttrs => {
                "All five mandatory schema attributes (`points`, `orientations`, `scales`, \
                 `opacities`, `colorsDC`) MUST be authored on every ParticleField3DGaussianSplat."
            }
        }
    }

    /// Whether this clause is a MUST (true) or SHOULD/optional (false).
    pub fn is_mandatory(self) -> bool {
        !matches!(self, Clause::RootXform)
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
    /// Clause did not apply (e.g. optional attribute absent).
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
    /// Container variant: `"usda"` or `"usdc"`.
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
    /// True when no clause failed (skips are tolerated — optional clauses
    /// skip on minimal fixtures, for example).
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
    #[error("usd: {0}")]
    Usd(#[from] splatforge_usd::UsdError),
    #[error("unsupported file extension: {0}")]
    UnsupportedExt(String),
}

/// Validate a `.usda` or `.usdc` file and return the conformance report.
pub fn validate_path(path: &Path) -> Result<Report, ValidateError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let (text, container) = match ext.as_str() {
        "usda" => (fs::read_to_string(path)?, "usda"),
        "usdc" => {
            // Decode the binary into the IR and re-render the canonical USDA
            // form. The resulting text is what we then check against the
            // clause matrix. This is deliberate: USDC and USDA encode the
            // same semantic content, and the schema clauses are written
            // against the textual schema in the OpenUSD docs.
            let scene = splatforge_usd::read_usdc(path)?;
            let text =
                splatforge_usd::render_usda(&scene, &splatforge_usd::UsdWriteOpts::default());
            (text, "usdc")
        }
        other => return Err(ValidateError::UnsupportedExt(other.to_string())),
    };
    let clauses = run_clauses(&text);
    let (pass, fail, skip) = tally(&clauses);
    Ok(Report {
        source: path.display().to_string(),
        container: container.to_string(),
        pass,
        fail,
        skip,
        clauses,
    })
}

/// Validate a USDA text document already in memory. Useful for tests and
/// for callers who have already decoded a USDC file out-of-process.
pub fn validate_usda(text: &str, source: &str) -> Report {
    let clauses = run_clauses(text);
    let (pass, fail, skip) = tally(&clauses);
    Report {
        source: source.to_string(),
        container: "usda".to_string(),
        pass,
        fail,
        skip,
        clauses,
    }
}

fn tally(clauses: &[ClauseResult]) -> (usize, usize, usize) {
    let mut pass = 0;
    let mut fail = 0;
    let mut skip = 0;
    for c in clauses {
        match c.status {
            Status::Pass => pass += 1,
            Status::Fail => fail += 1,
            Status::Skip => skip += 1,
        }
    }
    (pass, fail, skip)
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

/// Container for the parsed shape of the file relevant to clause checks.
/// We intentionally keep this narrow — full USDA parsing lives in
/// `splatforge-usd`. We only re-derive enough structure to answer the
/// clause questions.
struct ParsedDoc<'a> {
    /// Raw text body.
    raw: &'a str,
    /// True if the file begins with `#usda 1.0`.
    has_magic: bool,
    /// True if any `def ParticleField3DGaussianSplat` token appears.
    has_particle_field: bool,
    /// True if a `def Xform` is found above the particle field token.
    has_root_xform: bool,
    /// Layer metadata `upAxis` value, if any.
    up_axis: Option<String>,
    /// Layer metadata `metersPerUnit` value, if any.
    meters_per_unit: Option<f32>,
    /// `point3f[] points` array, if present.
    points: Option<Vec<[f32; 3]>>,
    /// `quatf[] orientations` array, if present (USDA `(w,x,y,z)` tuples).
    orientations: Option<Vec<[f32; 4]>>,
    /// `float3[] scales` array, if present.
    scales: Option<Vec<[f32; 3]>>,
    /// `float[] opacities` array, if present.
    opacities: Option<Vec<f32>>,
    /// `color3f[] colorsDC` array, if present.
    colors_dc: Option<Vec<[f32; 3]>>,
    /// `float[] widths` array, if present.
    widths: Option<Vec<f32>>,
    /// `vector3f[] velocities` array, if present.
    velocities: Option<Vec<[f32; 3]>>,
    /// `float3[] extent` array, if present.
    extent: Option<Vec<[f32; 3]>>,
    /// `custom float[] splatforge:shCoefficients` array, if present.
    sh_coefficients: Option<Vec<f32>>,
    /// `primvars:displayColor:interpolation` value, if any.
    display_color_interp: Option<String>,
    /// True if a typed declaration matched for `points`.
    points_typed: Option<bool>,
    /// True if a typed declaration matched for `orientations`.
    orientations_typed: Option<bool>,
    /// True if a typed declaration matched for `scales`.
    scales_typed: Option<bool>,
    /// True if a typed declaration matched for `widths`.
    widths_typed: Option<bool>,
    /// True if a typed declaration matched for `velocities`.
    velocities_typed: Option<bool>,
}

fn parse_doc(raw: &str) -> ParsedDoc<'_> {
    let has_magic = raw.trim_start().starts_with("#usda 1.0");
    let has_particle_field = raw.contains("def ParticleField3DGaussianSplat");
    let has_root_xform = {
        if let Some(pf_idx) = raw.find("def ParticleField3DGaussianSplat") {
            raw[..pf_idx].contains("def Xform")
        } else {
            false
        }
    };

    let up_axis = pull_quoted(raw, "upAxis");
    let meters_per_unit = pull_scalar(raw, "metersPerUnit");

    let points = pull_vec3_array(raw, &["point3f[] points", "point3f[]   points"]);
    let orientations = pull_quat_array(raw, &["quatf[] orientations"]);
    let scales = pull_vec3_array(raw, &["float3[] scales"]);
    let opacities = pull_scalar_array(raw, &["float[] opacities"]);
    let colors_dc = pull_vec3_array(raw, &["color3f[] colorsDC"]);
    let widths = pull_scalar_array(raw, &["float[] widths"]);
    let velocities = pull_vec3_array(raw, &["vector3f[] velocities"]);
    let extent = pull_vec3_array(raw, &["float3[] extent"]);
    let sh_coefficients = pull_scalar_array(raw, &["custom float[] splatforge:shCoefficients"]);
    let display_color_interp = pull_quoted(raw, "primvars:displayColor:interpolation");

    // Type checks: only need to know whether the *type* matched if the key is
    // present under a different type. We treat None to mean "attribute absent
    // under any type", Some(true)/false to mean present under correct/wrong.
    let points_typed = typed_presence(raw, "points", "point3f[]");
    let orientations_typed = typed_presence(raw, "orientations", "quatf[]");
    let scales_typed = typed_presence(raw, "scales", "float3[]");
    let widths_typed = typed_presence(raw, "widths", "float[]");
    let velocities_typed = typed_presence(raw, "velocities", "vector3f[]");

    ParsedDoc {
        raw,
        has_magic,
        has_particle_field,
        has_root_xform,
        up_axis,
        meters_per_unit,
        points,
        orientations,
        scales,
        opacities,
        colors_dc,
        widths,
        velocities,
        extent,
        sh_coefficients,
        display_color_interp,
        points_typed,
        orientations_typed,
        scales_typed,
        widths_typed,
        velocities_typed,
    }
}

fn run_clauses(raw: &str) -> Vec<ClauseResult> {
    let doc = parse_doc(raw);
    let mut out = Vec::with_capacity(Clause::all().len());

    // USDA_MAGIC
    out.push(if doc.has_magic {
        pass(Clause::UsdaMagic)
    } else {
        fail(Clause::UsdaMagic, "file does not begin with `#usda 1.0`")
    });

    // PRIM_PARTICLE_FIELD
    out.push(if doc.has_particle_field {
        pass(Clause::PrimParticleField)
    } else {
        fail(
            Clause::PrimParticleField,
            "no `def ParticleField3DGaussianSplat` prim found",
        )
    });

    // ROOT_XFORM (SHOULD; surface as pass/fail but it's marked non-mandatory)
    out.push(if doc.has_root_xform {
        pass(Clause::RootXform)
    } else {
        fail(
            Clause::RootXform,
            "ParticleField3DGaussianSplat is not under a root Xform",
        )
    });

    // UP_AXIS_VALID
    out.push(match doc.up_axis.as_deref() {
        None => skip(Clause::UpAxisValid, "upAxis not authored"),
        Some("Y") | Some("Z") => pass(Clause::UpAxisValid),
        Some(other) => fail(
            Clause::UpAxisValid,
            format!("upAxis={other:?}, want \"Y\" or \"Z\""),
        ),
    });

    // METERS_PER_UNIT_POSITIVE
    out.push(match doc.meters_per_unit {
        None => skip(Clause::MetersPerUnitPositive, "metersPerUnit not authored"),
        Some(v) if v > 0.0 && v.is_finite() => pass(Clause::MetersPerUnitPositive),
        Some(v) => fail(
            Clause::MetersPerUnitPositive,
            format!("metersPerUnit={v}, must be > 0"),
        ),
    });

    // ATTR_POINTS / ATTR_POINTS_TYPE
    out.push(if doc.points.is_some() {
        pass(Clause::AttrPointsPresent)
    } else if doc.points_typed.is_some() {
        // present but under wrong type — still fail "present" because the
        // typed lookup is the real check; a misnamed type means we never
        // recover the array.
        fail(
            Clause::AttrPointsPresent,
            "`points` attribute present but typed array not parseable",
        )
    } else {
        fail(Clause::AttrPointsPresent, "missing `points` attribute")
    });
    out.push(match doc.points_typed {
        None => fail(Clause::AttrPointsType, "missing `points` attribute"),
        Some(true) => pass(Clause::AttrPointsType),
        Some(false) => fail(
            Clause::AttrPointsType,
            "`points` is authored with a type other than `point3f[]`",
        ),
    });

    // ATTR_ORIENTATIONS / TYPE
    out.push(if doc.orientations.is_some() {
        pass(Clause::AttrOrientationsPresent)
    } else if doc.orientations_typed.is_some() {
        fail(
            Clause::AttrOrientationsPresent,
            "`orientations` present but typed array not parseable",
        )
    } else {
        fail(
            Clause::AttrOrientationsPresent,
            "missing `orientations` attribute",
        )
    });
    out.push(match doc.orientations_typed {
        None => fail(
            Clause::AttrOrientationsType,
            "missing `orientations` attribute",
        ),
        Some(true) => pass(Clause::AttrOrientationsType),
        Some(false) => fail(
            Clause::AttrOrientationsType,
            "`orientations` is authored with a type other than `quatf[]`",
        ),
    });

    // ATTR_SCALES / TYPE
    out.push(if doc.scales.is_some() {
        pass(Clause::AttrScalesPresent)
    } else if doc.scales_typed.is_some() {
        fail(
            Clause::AttrScalesPresent,
            "`scales` present but typed array not parseable",
        )
    } else {
        fail(Clause::AttrScalesPresent, "missing `scales` attribute")
    });
    out.push(match doc.scales_typed {
        None => fail(Clause::AttrScalesType, "missing `scales` attribute"),
        Some(true) => pass(Clause::AttrScalesType),
        Some(false) => fail(
            Clause::AttrScalesType,
            "`scales` is authored with a type other than `float3[]`",
        ),
    });

    // ATTR_OPACITIES + range
    out.push(if doc.opacities.is_some() {
        pass(Clause::AttrOpacitiesPresent)
    } else {
        fail(
            Clause::AttrOpacitiesPresent,
            "missing `opacities` attribute",
        )
    });
    out.push(match &doc.opacities {
        None => skip(Clause::AttrOpacitiesRange, "`opacities` not authored"),
        Some(v) => match first_out_of_range(v, 0.0, 1.0) {
            None => pass(Clause::AttrOpacitiesRange),
            Some((i, val)) => fail(
                Clause::AttrOpacitiesRange,
                format!("opacities[{i}]={val} outside [0, 1]"),
            ),
        },
    });

    // ATTR_COLORS_DC + range
    out.push(if doc.colors_dc.is_some() {
        pass(Clause::AttrColorsDcPresent)
    } else {
        fail(Clause::AttrColorsDcPresent, "missing `colorsDC` attribute")
    });
    out.push(match &doc.colors_dc {
        None => skip(Clause::AttrColorsDcRange, "`colorsDC` not authored"),
        Some(rows) => {
            let mut bad: Option<(usize, usize, f32)> = None;
            'outer: for (i, c) in rows.iter().enumerate() {
                for (k, v) in c.iter().enumerate() {
                    if !(*v >= 0.0 && *v <= 1.0 && v.is_finite()) {
                        bad = Some((i, k, *v));
                        break 'outer;
                    }
                }
            }
            match bad {
                None => pass(Clause::AttrColorsDcRange),
                Some((i, k, v)) => fail(
                    Clause::AttrColorsDcRange,
                    format!("colorsDC[{i}][{k}]={v} outside [0, 1]"),
                ),
            }
        }
    });

    // ATTR_WIDTHS optional type
    out.push(match doc.widths_typed {
        None => skip(Clause::AttrWidthsOptional, "`widths` not authored"),
        Some(true) => pass(Clause::AttrWidthsOptional),
        Some(false) => fail(
            Clause::AttrWidthsOptional,
            "`widths` present but typed as something other than `float[]`",
        ),
    });

    // ATTR_VELOCITIES optional type
    out.push(match doc.velocities_typed {
        None => skip(Clause::AttrVelocitiesOptional, "`velocities` not authored"),
        Some(true) => pass(Clause::AttrVelocitiesOptional),
        Some(false) => fail(
            Clause::AttrVelocitiesOptional,
            "`velocities` present but typed as something other than `vector3f[]`",
        ),
    });

    // COUNTS_AGREE
    out.push({
        let mut lengths: Vec<(&str, usize)> = Vec::new();
        if let Some(v) = &doc.points {
            lengths.push(("points", v.len()));
        }
        if let Some(v) = &doc.orientations {
            lengths.push(("orientations", v.len()));
        }
        if let Some(v) = &doc.scales {
            lengths.push(("scales", v.len()));
        }
        if let Some(v) = &doc.opacities {
            lengths.push(("opacities", v.len()));
        }
        if let Some(v) = &doc.colors_dc {
            lengths.push(("colorsDC", v.len()));
        }
        if let Some(v) = &doc.widths {
            lengths.push(("widths", v.len()));
        }
        if let Some(v) = &doc.velocities {
            lengths.push(("velocities", v.len()));
        }
        if lengths.is_empty() {
            skip(Clause::CountsAgree, "no per-splat attributes parsed")
        } else {
            let first = lengths[0].1;
            if lengths.iter().all(|(_, l)| *l == first) {
                pass(Clause::CountsAgree)
            } else {
                fail(
                    Clause::CountsAgree,
                    format!(
                        "per-splat array lengths disagree: {:?}",
                        lengths.iter().collect::<Vec<_>>()
                    ),
                )
            }
        }
    });

    // EXTENT_CONSISTENT
    out.push(match (&doc.extent, &doc.points) {
        (None, _) => skip(Clause::ExtentConsistent, "`extent` not authored"),
        (Some(_), None) => skip(
            Clause::ExtentConsistent,
            "`extent` authored but `points` not parseable",
        ),
        (Some(ext), Some(pts)) => {
            if ext.len() != 2 {
                fail(
                    Clause::ExtentConsistent,
                    format!("`extent` must have 2 elements, got {}", ext.len()),
                )
            } else {
                let (lo, hi) = (ext[0], ext[1]);
                let mut violated: Option<(usize, usize, f32)> = None;
                'extent_check: for (i, p) in pts.iter().enumerate() {
                    for k in 0..3 {
                        if p[k] < lo[k] || p[k] > hi[k] {
                            violated = Some((i, k, p[k]));
                            break 'extent_check;
                        }
                    }
                }
                match violated {
                    None => pass(Clause::ExtentConsistent),
                    Some((i, k, v)) => fail(
                        Clause::ExtentConsistent,
                        format!("points[{i}][{k}]={v} outside extent {:?}..={:?}", lo, hi),
                    ),
                }
            }
        }
    });

    // QUATS_NORMALIZED
    out.push(match &doc.orientations {
        None => skip(Clause::QuaternionsNormalized, "`orientations` not authored"),
        Some(qs) => {
            let mut worst: Option<(usize, f32)> = None;
            for (i, q) in qs.iter().enumerate() {
                let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
                let n = n2.sqrt();
                let dev = (n - 1.0).abs();
                if worst.map(|(_, w)| dev > w).unwrap_or(true) {
                    worst = Some((i, dev));
                }
            }
            match worst {
                Some((i, dev)) if dev > 1e-3 => fail(
                    Clause::QuaternionsNormalized,
                    format!("orientations[{i}] norm deviates from 1.0 by {dev:.6}"),
                ),
                _ => pass(Clause::QuaternionsNormalized),
            }
        }
    });

    // SH_COEFFS_COUNT — accept any degree in {0,1,2,3} that yields exact match.
    out.push(match (&doc.sh_coefficients, &doc.points) {
        (None, _) => skip(Clause::ShCoefficientsCount, "shCoefficients not authored"),
        (Some(_), None) => skip(
            Clause::ShCoefficientsCount,
            "shCoefficients authored but splat count unknown",
        ),
        (Some(c), Some(p)) => {
            let n = p.len();
            let total = c.len();
            let mut matched = false;
            for d in 0..=3 {
                let band = (d + 1) * (d + 1);
                if total == n * band * 3 {
                    matched = true;
                    break;
                }
            }
            if matched {
                pass(Clause::ShCoefficientsCount)
            } else {
                fail(
                    Clause::ShCoefficientsCount,
                    format!(
                        "shCoefficients.count={total}, want n*((d+1)^2)*3 for n={n}, d in 0..=3"
                    ),
                )
            }
        }
    });

    // DISPLAYCOLOR_INTERP
    out.push(match doc.display_color_interp.as_deref() {
        None => skip(
            Clause::DisplayColorInterpolation,
            "displayColor interpolation not authored",
        ),
        Some("vertex") | Some("varying") => pass(Clause::DisplayColorInterpolation),
        Some(other) => fail(
            Clause::DisplayColorInterpolation,
            format!(
                "primvars:displayColor:interpolation={other:?}, want \"vertex\" or \"varying\""
            ),
        ),
    });

    // SCHEMA_REQUIRED_ATTRS — all five must be parseable.
    out.push({
        let missing: Vec<&str> = [
            ("points", doc.points.is_some()),
            ("orientations", doc.orientations.is_some()),
            ("scales", doc.scales.is_some()),
            ("opacities", doc.opacities.is_some()),
            ("colorsDC", doc.colors_dc.is_some()),
        ]
        .iter()
        .filter_map(|(n, ok)| if *ok { None } else { Some(*n) })
        .collect();
        if missing.is_empty() {
            pass(Clause::SchemaRequiredAttrs)
        } else {
            fail(
                Clause::SchemaRequiredAttrs,
                format!("missing required attrs: {missing:?}"),
            )
        }
    });

    // Sanity: silence the unused-raw-on-doc warning by referencing it.
    let _ = doc.raw;

    out
}

// ---------- helper parsers ----------

fn first_out_of_range(values: &[f32], lo: f32, hi: f32) -> Option<(usize, f32)> {
    for (i, v) in values.iter().enumerate() {
        if !(*v >= lo && *v <= hi && v.is_finite()) {
            return Some((i, *v));
        }
    }
    None
}

/// Look for `key = "..."` style lines and return the unquoted value.
fn pull_quoted(raw: &str, key: &str) -> Option<String> {
    let needle = format!("{key} = \"");
    let idx = raw.find(&needle)?;
    let start = idx + needle.len();
    let rest = &raw[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Look for `key = <number>` and parse the number.
fn pull_scalar(raw: &str, key: &str) -> Option<f32> {
    let needle = format!("{key} = ");
    let idx = raw.find(&needle)?;
    let start = idx + needle.len();
    let rest = &raw[start..];
    // Token ends at first whitespace or `)`.
    let end = rest
        .find(|c: char| c.is_whitespace() || c == ')')
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f32>().ok()
}

/// Detect whether the named attribute is present under any of the candidate
/// type prefixes.
fn typed_presence(raw: &str, attr_name: &str, want_type: &str) -> Option<bool> {
    let want = format!("{want_type} {attr_name}");
    if raw.contains(&want) {
        return Some(true);
    }
    // Variant: the splatforge USDA writer never authors attribute keys with
    // a leading `uniform` or `custom` modifier for the schema attrs, but
    // tolerate it here for foreign inputs.
    if raw.contains(&format!("uniform {want_type} {attr_name}"))
        || raw.contains(&format!("custom {want_type} {attr_name}"))
    {
        return Some(true);
    }
    // Heuristic for the "present but mis-typed" case: look for whitespace-
    // delimited `<TOKEN> <attr> =`. If found we report Some(false).
    for line in raw.lines() {
        let trimmed = line.trim_start();
        // Skip lines that don't look like attribute declarations.
        if !trimmed.contains(&format!(" {attr_name} "))
            && !trimmed.contains(&format!(" {attr_name}="))
        {
            continue;
        }
        // Tokens are roughly: [modifier?] <type> <attr> = ...
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        for w in tokens.windows(2) {
            if w[1] == attr_name {
                // w[0] is the type; we already know it isn't `want_type`.
                if w[0] != want_type {
                    return Some(false);
                }
            }
        }
    }
    None
}

fn pull_vec3_array(raw: &str, keys: &[&str]) -> Option<Vec<[f32; 3]>> {
    for key in keys {
        if let Some(body) = pull_array_body(raw, key) {
            let mut out = Vec::new();
            for tuple in split_parens(&body) {
                let parts: Vec<&str> = tuple.split(',').map(str::trim).collect();
                if parts.len() != 3 {
                    return None;
                }
                let a = parts[0].parse::<f32>().ok()?;
                let b = parts[1].parse::<f32>().ok()?;
                let c = parts[2].parse::<f32>().ok()?;
                out.push([a, b, c]);
            }
            return Some(out);
        }
    }
    None
}

fn pull_quat_array(raw: &str, keys: &[&str]) -> Option<Vec<[f32; 4]>> {
    for key in keys {
        if let Some(body) = pull_array_body(raw, key) {
            let mut out = Vec::new();
            for tuple in split_parens(&body) {
                let parts: Vec<&str> = tuple.split(',').map(str::trim).collect();
                if parts.len() != 4 {
                    return None;
                }
                let mut q = [0.0f32; 4];
                for (i, p) in parts.iter().enumerate() {
                    q[i] = p.parse::<f32>().ok()?;
                }
                out.push(q);
            }
            return Some(out);
        }
    }
    None
}

fn pull_scalar_array(raw: &str, keys: &[&str]) -> Option<Vec<f32>> {
    for key in keys {
        if let Some(body) = pull_array_body(raw, key) {
            let mut out = Vec::new();
            for tok in body.split(',') {
                let tok = tok.trim();
                if tok.is_empty() {
                    continue;
                }
                out.push(tok.parse::<f32>().ok()?);
            }
            return Some(out);
        }
    }
    None
}

fn pull_array_body(raw: &str, key: &str) -> Option<String> {
    let idx = raw.find(key)?;
    let after = &raw[idx + key.len()..];
    let eq = after.find('=')?;
    let lb = after[eq..].find('[')?;
    let start = eq + lb + 1;
    let rb = after[start..].find(']')?;
    Some(after[start..start + rb].to_string())
}

fn split_parens(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut buf = String::new();
    for ch in body.chars() {
        match ch {
            '(' => {
                depth += 1;
                if depth > 1 {
                    buf.push(ch);
                }
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    if !buf.trim().is_empty() {
                        out.push(buf.trim().to_string());
                    }
                    buf.clear();
                } else {
                    buf.push(ch);
                }
            }
            _ if depth > 0 => buf.push(ch),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_valid() -> String {
        "#usda 1.0\n\
         (\n    upAxis = \"Y\"\n    metersPerUnit = 1\n)\n\n\
         def Xform \"World\"\n{\n\
             def ParticleField3DGaussianSplat \"Splats\"\n    {\n\
                 point3f[] points = [(0.0, 0.0, 0.0)]\n        \
                 quatf[] orientations = [(1.0, 0.0, 0.0, 0.0)]\n        \
                 float3[] scales = [(1.0, 1.0, 1.0)]\n        \
                 float[] opacities = [1.0]\n        \
                 color3f[] colorsDC = [(0.5, 0.5, 0.5)]\n    }\n}\n"
            .to_string()
    }

    #[test]
    fn minimal_valid_passes() {
        let r = validate_usda(&minimal_valid(), "test");
        assert!(r.is_pass(), "expected pass; got {:?}", r.clauses);
        assert!(r.clauses.len() >= 15);
    }

    #[test]
    fn missing_orientations_fails() {
        let bad = minimal_valid().replace(
            "quatf[] orientations = [(1.0, 0.0, 0.0, 0.0)]\n        ",
            "",
        );
        let r = validate_usda(&bad, "test");
        assert!(!r.is_pass());
        assert!(r
            .clauses
            .iter()
            .any(|c| c.id == "ATTR_ORIENTATIONS" && c.status == Status::Fail));
    }

    #[test]
    fn opacity_out_of_range_fails() {
        let bad = minimal_valid().replace("float[] opacities = [1.0]", "float[] opacities = [1.5]");
        let r = validate_usda(&bad, "test");
        assert!(r
            .clauses
            .iter()
            .any(|c| c.id == "ATTR_OPACITIES_RANGE" && c.status == Status::Fail));
    }

    #[test]
    fn quat_not_normalized_fails() {
        let bad = minimal_valid().replace(
            "quatf[] orientations = [(1.0, 0.0, 0.0, 0.0)]",
            "quatf[] orientations = [(2.0, 0.0, 0.0, 0.0)]",
        );
        let r = validate_usda(&bad, "test");
        assert!(r
            .clauses
            .iter()
            .any(|c| c.id == "QUATS_NORMALIZED" && c.status == Status::Fail));
    }
}
