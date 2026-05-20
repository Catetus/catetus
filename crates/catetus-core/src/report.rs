//! Deterministic analyze-report types & serializer.
//!
//! Implements SPEC-0005. JSON output has lexically sorted keys and floats are
//! formatted via `ryu` so the same scene always produces byte-identical JSON.

use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

use crate::hash::scene_hash;
use crate::ir::{Color, SplatScene};

/// Top-level analyze report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeReport {
    /// Schema version. Always `"1"` for now.
    pub schema_version: String,
    /// Source format (`"ply"`, `"spz"`, `"gltf"`, `"glb"`).
    pub format: String,
    /// Number of splats in the scene.
    pub splat_count: usize,
    /// On-disk size of the input, in bytes.
    pub file_size: u64,
    /// World-space axis-aligned bounding box.
    pub bounding_box: BoundingBox,
    /// Coordinate-system metadata.
    pub coordinate_system: CoordSystemReport,
    /// Which IR attributes are present.
    pub attributes: Attributes,
    /// Summary statistics over opacity.
    pub opacity_distribution: OpacityDistribution,
    /// Summary statistics over per-axis scale.
    pub scale_distribution: ScaleDistribution,
    /// SH degree (0 for plain RGB).
    pub sh_degree: u8,
    /// Estimated runtime memory cost.
    pub estimated_memory: EstimatedMemory,
    /// Non-fatal warnings raised by the analyzer.
    pub warnings: Vec<Warning>,
    /// Suggested optimization presets.
    pub recommendations: Vec<Recommendation>,
    /// BLAKE3 hash of the IR (`"blake3:<hex>"`).
    pub hash: String,
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

/// Coordinate-system summary fields exposed in the JSON report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordSystemReport {
    /// `"Y"` or `"Z"`.
    pub up: String,
    /// `"right"` or `"left"`.
    pub handedness: String,
}

/// Which IR attributes are present in the scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attributes {
    /// Splats have positions (always true for a valid scene).
    pub position: bool,
    /// Splats have rotations.
    pub rotation: bool,
    /// Splats have scales.
    pub scale: bool,
    /// Splats have opacity.
    pub opacity: bool,
    /// Splats have a DC color term.
    pub color_dc: bool,
    /// Splats have higher-order SH coefficients.
    pub sh_rest: bool,
}

/// Distribution stats for opacity values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpacityDistribution {
    /// Minimum opacity.
    pub min: f32,
    /// Maximum opacity.
    pub max: f32,
    /// Mean opacity.
    pub mean: f32,
    /// Median opacity.
    pub median: f32,
}

/// Distribution stats for per-axis scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleDistribution {
    /// Per-axis minimum scale.
    pub min: [f32; 3],
    /// Per-axis maximum scale.
    pub max: [f32; 3],
    /// Per-axis mean scale.
    pub mean: [f32; 3],
}

/// Estimated runtime memory usage, in megabytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EstimatedMemory {
    /// Approximate host RAM, MB.
    pub ram_mb: u64,
    /// Approximate device VRAM, MB.
    pub vram_mb: u64,
}

/// A non-fatal analyzer warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Warning {
    /// Stable machine-readable code (e.g. `"floater_cluster_detected"`).
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// `"info"`, `"warn"`, or `"error"`.
    pub severity: String,
}

/// A suggested optimization preset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    /// Preset identifier (e.g. `"web-mobile"`).
    pub preset: String,
    /// Why the preset is being recommended.
    pub rationale: String,
}

impl AnalyzeReport {
    /// Compute an analyze report from an in-memory scene and source metadata.
    pub fn from_scene(scene: &SplatScene, format: &str, file_size: u64) -> Self {
        let n = scene.splats.len();
        let mut bbox_min = [f32::INFINITY; 3];
        let mut bbox_max = [f32::NEG_INFINITY; 3];
        let mut opacities: Vec<f32> = Vec::with_capacity(n);
        let mut scale_min = [f32::INFINITY; 3];
        let mut scale_max = [f32::NEG_INFINITY; 3];
        let mut scale_sum = [0.0f64; 3];
        let mut sh_rest = false;
        let mut sh_degree: u8 = 0;

        for s in &scene.splats {
            for i in 0..3 {
                let p = s.position[i];
                if p < bbox_min[i] {
                    bbox_min[i] = p;
                }
                if p > bbox_max[i] {
                    bbox_max[i] = p;
                }
                if s.scale[i] < scale_min[i] {
                    scale_min[i] = s.scale[i];
                }
                if s.scale[i] > scale_max[i] {
                    scale_max[i] = s.scale[i];
                }
                scale_sum[i] += s.scale[i] as f64;
            }
            opacities.push(s.opacity);
            if let Color::Sh { degree, coeffs } = &s.color {
                if *degree > 0 && !coeffs.is_empty() {
                    sh_rest = true;
                }
                if *degree > sh_degree {
                    sh_degree = *degree;
                }
            }
        }

        if n == 0 {
            bbox_min = [0.0; 3];
            bbox_max = [0.0; 3];
            scale_min = [0.0; 3];
            scale_max = [0.0; 3];
        }

        let opacity_distribution = compute_opacity_distribution(&opacities);
        let scale_mean = if n == 0 {
            [0.0; 3]
        } else {
            [
                (scale_sum[0] / n as f64) as f32,
                (scale_sum[1] / n as f64) as f32,
                (scale_sum[2] / n as f64) as f32,
            ]
        };

        let mut warnings = Vec::new();
        if let Some(w) = detect_floater_warning(scene) {
            warnings.push(w);
        }

        let mut recommendations = Vec::new();
        if file_size > 100 * 1024 * 1024 {
            recommendations.push(Recommendation {
                preset: "web-mobile".to_string(),
                rationale: "input larger than 100MB; web-mobile preset recommended".to_string(),
            });
        }

        let ram_mb = ((n as u64).saturating_mul(60)) / (1024 * 1024);
        let vram_mb = ((n as u64).saturating_mul(110)) / (1024 * 1024);

        AnalyzeReport {
            schema_version: "1".to_string(),
            format: format.to_string(),
            splat_count: n,
            file_size,
            bounding_box: BoundingBox {
                min: bbox_min,
                max: bbox_max,
            },
            coordinate_system: CoordSystemReport {
                up: scene.coordinate_system.up_label().to_string(),
                handedness: scene.coordinate_system.handedness_label().to_string(),
            },
            attributes: Attributes {
                position: n > 0,
                rotation: n > 0,
                scale: n > 0,
                opacity: n > 0,
                color_dc: n > 0,
                sh_rest,
            },
            opacity_distribution,
            scale_distribution: ScaleDistribution {
                min: scale_min,
                max: scale_max,
                mean: scale_mean,
            },
            sh_degree,
            estimated_memory: EstimatedMemory { ram_mb, vram_mb },
            warnings,
            recommendations,
            hash: scene_hash(scene),
        }
    }

    /// Serialize the report to deterministic JSON.
    ///
    /// Keys are emitted in lexical order, floats are formatted with `ryu`, and
    /// the result is byte-identical for identical inputs.
    pub fn to_json(&self, pretty: bool) -> String {
        let value = serde_json::to_value(self).expect("AnalyzeReport always serializes");
        let mut buf = String::new();
        emit(&value, &mut buf, pretty, 0);
        buf
    }
}

fn compute_opacity_distribution(opacities: &[f32]) -> OpacityDistribution {
    if opacities.is_empty() {
        return OpacityDistribution {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
        };
    }
    let mut sorted: Vec<f32> = opacities.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = (sorted.iter().copied().map(|x| x as f64).sum::<f64>() / sorted.len() as f64) as f32;
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    OpacityDistribution {
        min,
        max,
        mean,
        median,
    }
}

fn detect_floater_warning(scene: &SplatScene) -> Option<Warning> {
    let n = scene.splats.len();
    if n < 4 {
        return None;
    }
    let mut centroid = [0.0f64; 3];
    for s in &scene.splats {
        for (i, c) in centroid.iter_mut().enumerate() {
            *c += s.position[i] as f64;
        }
    }
    for c in &mut centroid {
        *c /= n as f64;
    }
    let mut sum_sq = 0.0f64;
    let mut dists = Vec::with_capacity(n);
    for s in &scene.splats {
        let dx = s.position[0] as f64 - centroid[0];
        let dy = s.position[1] as f64 - centroid[1];
        let dz = s.position[2] as f64 - centroid[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        dists.push(d);
        sum_sq += d * d;
    }
    let variance = sum_sq / n as f64;
    let sigma = variance.sqrt();
    if sigma <= f64::EPSILON {
        return None;
    }
    let count = dists.iter().filter(|d| **d > 5.0 * sigma).count();
    if count > 0 {
        Some(Warning {
            code: "floater_cluster_detected".to_string(),
            message: format!("{count} splats are >5σ from the scene centroid; possible floaters"),
            severity: "warn".to_string(),
        })
    } else {
        None
    }
}

// ---------- deterministic JSON emitter ----------

fn emit(value: &serde_json::Value, out: &mut String, pretty: bool, depth: usize) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => emit_number(n, out),
        serde_json::Value::String(s) => emit_string(s, out),
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if pretty {
                    out.push('\n');
                    indent(out, depth + 1);
                }
                emit(item, out, pretty, depth + 1);
                if i + 1 < items.len() {
                    out.push(',');
                    if !pretty {
                        out.push(' ');
                    }
                }
            }
            if pretty {
                out.push('\n');
                indent(out, depth);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if pretty {
                    out.push('\n');
                    indent(out, depth + 1);
                }
                emit_string(k, out);
                out.push(':');
                // Both pretty and compact emit a single space after the colon
                // — that's by design to keep the snapshot stable across modes.
                out.push(' ');
                emit(&map[*k], out, pretty, depth + 1);
                if i + 1 < keys.len() {
                    out.push(',');
                    if !pretty {
                        out.push(' ');
                    }
                }
            }
            if pretty {
                out.push('\n');
                indent(out, depth);
            }
            out.push('}');
        }
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn emit_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn emit_number(n: &serde_json::Number, out: &mut String) {
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else if let Some(f) = n.as_f64() {
        let mut ryu_buf = ryu::Buffer::new();
        out.push_str(ryu_buf.format(f));
    } else {
        out.push_str(&n.to_string());
    }
}
