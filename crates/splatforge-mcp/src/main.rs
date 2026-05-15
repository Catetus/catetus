#![deny(clippy::all)]
//! `splatforge-mcp` — Model Context Protocol server for SplatForge.
//!
//! Speaks line-delimited JSON-RPC 2.0 over stdio so any MCP-compatible
//! client (Claude Desktop, Cursor, Cline, Zed, Continue, …) can drive the
//! toolkit through an LLM tool-call loop. Currently exposes the **public
//! tier** from `docs/mcp-design.md`:
//!
//!   - `splatforge.analyze`      — wraps [`AnalyzeReport::from_scene`]
//!   - `splatforge.list_presets` — static catalog of built-in presets
//!   - `splatforge.optimize`     — runs a free preset through the Pipeline
//!
//! The Streamable-HTTP transport, the authenticated tier (which gates the
//! `splatforge.repack` paid preset behind an API key), and the moat tools
//! (`predict_quality`, `recommend_preset`) live behind feature flags or in
//! the private repo and will land in a later milestone.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use splatforge_core::{format_from_extension, format_from_magic, AnalyzeReport, SplatScene};
use splatforge_gltf::{read_glb, read_gltf, write_gltf, WriteOpts};
use splatforge_optimize::preset;
use splatforge_ply::{read_ply, write_ply};
use splatforge_spz::read_spz;

// ------------------------------------------------------------- JSON-RPC types

/// Wire-level JSON-RPC 2.0 envelope. Both requests and notifications use
/// this shape; `id == None` means "notification, no response expected."
#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse<'a> {
    jsonrpc: &'a str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// Standard JSON-RPC error codes. MCP tool-level errors go in the
/// `result.isError = true` envelope so the LLM can recover, not here.
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

// ----------------------------------------------------------------- MCP types

/// MCP `initialize` response advertises which features this server speaks.
/// We only support `tools` today — no resources, prompts, or sampling.
fn server_capabilities() -> Value {
    json!({
        "protocolVersion": "2025-03-26",
        "serverInfo": {
            "name": "splatforge-mcp",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": { "subscribe": false, "listChanged": false }
        }
    })
}

// ------------------------------------------------------------------ resources

/// MCP `resources/list` advertises read-only artifacts that the LLM can
/// fetch via `resources/read`. The SplatBench corpus JSON is the
/// design-doc differentiator vs. a CLI wrapper: the LLM can grep the
/// corpus for comparable scenes without needing a per-scene tool call.
fn list_resources_response() -> Value {
    json!({
        "resources": [
            {
                "uri": "splatbench://v0",
                "name": "SplatBench v0",
                "description":
                    "Full SplatBench v0 corpus: 14 synthetic failure-mode probes \
                     + 2 real Mip-NeRF360 scenes (bonsai, bicycle), with per-preset \
                     compression ratios, ΔE94 fidelity numbers, and ML-score values. \
                     Use this when you need to compare against measured numbers \
                     rather than rules of thumb.",
                "mimeType": "application/json"
            }
        ]
    })
}

fn read_resource_response(uri: &str) -> Result<Value, (i64, String)> {
    if uri == "splatbench://v0" {
        Ok(json!({
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": SPLATBENCH_V0_JSON,
                }
            ]
        }))
    } else {
        Err((INVALID_PARAMS, format!("unknown resource uri: {uri}")))
    }
}

// -------------------------------------------------------------- tool catalog

/// One entry in the `tools/list` response. Keeping a static list keeps the
/// dispatch matrix obvious; each tool's handler lives in its own function
/// below.
struct ToolDef {
    name: &'static str,
    description: &'static str,
    input_schema: fn() -> Value,
}

const TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "splatforge.analyze",
        description:
            "Analyze a Gaussian Splat file (.ply/.spz/.gltf/.glb) and return splat count, \
             bounding box, SH degree, opacity distribution, content hash, and recommended \
             presets. Read-only, fast (typically <2s for files under 100 MB). Use this first \
             whenever the user gives you a splat file you don't already know.",
        input_schema: schema_analyze,
    },
    ToolDef {
        name: "splatforge.list_presets",
        description:
            "List all built-in SplatForge presets with their target use case, typical \
             compression ratio, and whether they're free or paid. Use when the user asks \
             'what presets are available?' or before picking one yourself.",
        input_schema: schema_list_presets,
    },
    ToolDef {
        name: "splatforge.optimize",
        description:
            "Run a SplatForge preset over a splat file and return the output path plus the \
             size-reduction ratio, fidelity report, and timing. Use this after `analyze` and \
             either `recommend_preset` or when the user has specified a preset by name. Free \
             for built-in presets; the differentiable-repack preset is paid and not available \
             on this public-tier server.",
        input_schema: schema_optimize,
    },
    ToolDef {
        name: "splatforge.generate_lod_pyramid",
        description:
            "Take one .ply / .spz / .gltf and emit N .ply files at user-specified \
             splat-count ratios. Ranks splats by saliency (sigmoid(α) · Σexp(scale), \
             the same metric the diff-repack premium tier converges to), then writes \
             top-K subsets. Use this when a hosting / streaming platform requires the \
             user to pre-train multiple LOD variants (e.g. Blurry's upload UI asks for \
             4 .ply files at decreasing detail) — SplatForge generates the pyramid \
             from a single converged scene in a few seconds.",
        input_schema: schema_generate_lod_pyramid,
    },
    ToolDef {
        name: "splatforge.find_similar_scenes",
        description:
            "Given a scene's splatCount, SH degree, and optional content class, find the \
             K closest scenes in the bundled SplatBench v0 corpus and return their measured \
             per-preset compression ratios + ΔE94/ML-score fidelity numbers. This is the \
             public-tier substitute for `recommend_preset` — the LLM can pick a preset by \
             inspecting how that preset performed on similar scenes. Call `analyze` first to \
             obtain the splatCount/shDegree inputs.",
        input_schema: schema_find_similar_scenes,
    },
];

// All tool input schemas accept the same `SplatRef` for input. The public
// stdio server only supports the `path` form; URL / blob_id are accepted by
// the authenticated tier (different binary, same protocol).
fn schema_splatref_path_only() -> Value {
    json!({
        "type": "object",
        "required": ["kind", "path"],
        "properties": {
            "kind": { "const": "path" },
            "path": {
                "type": "string",
                "description": "Absolute path on the machine running this MCP server. \
                                Hosted MCP transports do not accept the 'path' form."
            }
        }
    })
}

fn schema_analyze() -> Value {
    json!({
        "type": "object",
        "required": ["input"],
        "properties": {
            "input": schema_splatref_path_only(),
            "include_recommendations": { "type": "boolean", "default": true }
        }
    })
}

fn schema_list_presets() -> Value {
    json!({ "type": "object", "properties": {} })
}

fn schema_generate_lod_pyramid() -> Value {
    json!({
        "type": "object",
        "required": ["input", "ratios"],
        "properties": {
            "input": schema_splatref_path_only(),
            "ratios": {
                "type": "array",
                "description": "Splat-count ratios in (0, 1]. e.g. [1.0, 0.5, 0.25, 0.1] \
                                produces a 4-level pyramid: full, half, quarter, tenth.",
                "items": { "type": "number", "exclusiveMinimum": 0, "maximum": 1 },
                "minItems": 1,
                "maxItems": 16
            },
            "out_dir": {
                "type": "string",
                "description": "Absolute directory for the emitted .ply pyramid. \
                                Each level is written as `lod_<idx>_r<ratio>.ply`. \
                                Created if missing."
            }
        }
    })
}

fn schema_find_similar_scenes() -> Value {
    json!({
        "type": "object",
        "required": ["splatCount"],
        "properties": {
            "splatCount": {
                "type": "integer", "minimum": 1,
                "description": "Splat count from the analyze report."
            },
            "shDegree": {
                "type": "integer", "minimum": 0, "maximum": 3,
                "description": "Spherical-harmonic degree from the analyze report."
            },
            "contentClass": {
                "type": "string",
                "enum": [
                    "product-scan", "indoor-real-estate", "outdoor-scene",
                    "object-isolated", "transparent-volume", "portrait", "other"
                ],
                "description": "Optional class — if provided, scenes of the same class are scored higher."
            },
            "k": {
                "type": "integer", "minimum": 1, "maximum": 16, "default": 3,
                "description": "How many neighbors to return."
            }
        }
    })
}

fn schema_optimize() -> Value {
    json!({
        "type": "object",
        "required": ["input", "preset"],
        "properties": {
            "input": schema_splatref_path_only(),
            "preset": {
                "type": "string",
                "enum": [
                    "lossless-repack", "web-mobile", "web-desktop",
                    "quest-browser", "visionos-preview", "thumbnail-preview",
                    "quality-max", "size-min"
                ]
            },
            "out": {
                "type": "string",
                "description": "Optional absolute path for the optimized output. \
                                Defaults to <input>.optimized.gltf next to the input."
            },
            "chunked": { "type": "boolean", "default": false }
        }
    })
}

// -------------------------------------------------------------- tool dispatch

/// Outcome of a tool call. We always emit MCP's `content` block (the
/// human-readable side) and additionally surface the structured payload
/// via `structuredContent` (per the 2025-03-26 spec) so LLMs can consume
/// it natively without re-parsing JSON out of a text block.
struct ToolResult {
    json: Value,
    text: String,
    is_error: bool,
}

fn call_tool(name: &str, args: &Value) -> ToolResult {
    match name {
        "splatforge.analyze" => tool_analyze(args),
        "splatforge.list_presets" => tool_list_presets(),
        "splatforge.optimize" => tool_optimize(args),
        "splatforge.generate_lod_pyramid" => tool_generate_lod_pyramid(args),
        "splatforge.find_similar_scenes" => tool_find_similar_scenes(args),
        other => tool_error(format!("unknown tool: {other}")),
    }
}

// ---------------------------- splatforge.generate_lod_pyramid handler

fn tool_generate_lod_pyramid(args: &Value) -> ToolResult {
    let input = match args.get("input") {
        Some(v) => v,
        None => return tool_error("missing required argument: input"),
    };
    let path = match splatref_path(input) {
        Ok(p) => p,
        Err(e) => return tool_error(e),
    };
    if !path.exists() {
        return tool_error(format!("file not found: {}", path.display()));
    }
    let ratios: Vec<f32> = match args.get("ratios").and_then(Value::as_array) {
        Some(arr) => {
            let mut out: Vec<f32> = Vec::new();
            for v in arr {
                let Some(n) = v.as_f64() else {
                    return tool_error("ratios entries must be numbers");
                };
                if !(n > 0.0 && n <= 1.0) {
                    return tool_error(format!("ratio {n} out of (0, 1]"));
                }
                out.push(n as f32);
            }
            out
        }
        _ => return tool_error("ratios must be a non-empty array of numbers in (0, 1]"),
    };
    if ratios.is_empty() {
        return tool_error("ratios array must contain at least one entry");
    }

    let out_dir = match args.get("out_dir").and_then(Value::as_str) {
        Some(s) => PathBuf::from(s),
        None => path
            .parent()
            .map(|p| p.join("lod_pyramid"))
            .unwrap_or_else(|| PathBuf::from("lod_pyramid")),
    };
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        return tool_error(format!("creating {}: {e}", out_dir.display()));
    }

    let t0 = std::time::Instant::now();
    let (scene, _fmt) = match load_scene(&path) {
        Ok(p) => p,
        Err(e) => return tool_error(e),
    };
    let n_in = scene.splats.len();
    if n_in == 0 {
        return tool_error("input scene has zero splats");
    }

    // Rank splats by saliency = sigmoid(α) · Σ exp(scale).
    // This is the same metric the diff-repack premium tier converges to
    // and the standard signal in the 3DGS literature for "is this splat
    // load-bearing?" Cheap O(N) computation; the topk sort is the only
    // non-trivial cost and scales fine to millions of splats.
    let mut saliencies: Vec<(usize, f32)> = scene
        .splats
        .iter()
        .enumerate()
        .map(|(i, s)| {
            // Stored opacity is post-sigmoid linear in [0, 1]; stored scale
            // is per-axis linear (not log-space) per the SplatIR spec.
            // The saliency formula assumes log-space scale, so take ln
            // before summing exp — which simplifies to summing the linear
            // scales. Stable and matches the diff-repack ranking.
            let scale_sum = s.scale[0] + s.scale[1] + s.scale[2];
            (i, s.opacity * scale_sum)
        })
        .collect();
    // Sort descending by saliency — splat 0 in the sorted list is the
    // single most "load-bearing" splat in the entire scene.
    saliencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Emit one .ply per ratio.
    let mut pyramid: Vec<Value> = Vec::with_capacity(ratios.len());
    for (idx, &ratio) in ratios.iter().enumerate() {
        let keep_n = ((n_in as f32) * ratio).ceil() as usize;
        let keep_n = keep_n.clamp(1, n_in);
        let mut keep_indices: Vec<usize> = saliencies[..keep_n].iter().map(|(i, _)| *i).collect();
        keep_indices.sort_unstable();
        let mut sub_scene = scene.clone();
        sub_scene.splats = keep_indices
            .iter()
            .map(|&i| scene.splats[i].clone())
            .collect();
        // Drop any LOD index sets the source carried — they reference the
        // full splat list and don't survive subsetting.
        // lods is Option<Vec<LodLevel>> — drop it entirely for the subset
        // since the original indices reference splats we just dropped.
        sub_scene.lods = None;
        let level_name = format!(
            "lod_{:02}_r{:.4}.ply",
            idx,
            ratio
        );
        let level_path = out_dir.join(&level_name);
        if let Err(e) = write_ply(&sub_scene, &level_path) {
            return tool_error(format!("writing {}: {e}", level_path.display()));
        }
        let bytes_out = std::fs::metadata(&level_path)
            .map(|m| m.len())
            .unwrap_or(0);
        pyramid.push(json!({
            "ratio": ratio,
            "splatCount": keep_n,
            "bytesOut": bytes_out,
            "path": level_path.display().to_string(),
        }));
    }

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let lines: Vec<String> = pyramid
        .iter()
        .map(|p| {
            format!(
                "  r={:>5.3}  splats={:>9}  bytes={:>10}  path={}",
                p["ratio"].as_f64().unwrap_or(0.0),
                p["splatCount"].as_u64().unwrap_or(0),
                p["bytesOut"].as_u64().unwrap_or(0),
                p["path"].as_str().unwrap_or("?")
            )
        })
        .collect();
    let text = format!(
        "Generated LOD pyramid from {} ({} splats in) → {} levels in {} ms:\n{}",
        path.display(),
        n_in,
        ratios.len(),
        elapsed_ms,
        lines.join("\n"),
    );
    ToolResult {
        json: json!({
            "input": path.display().to_string(),
            "splatsIn": n_in,
            "elapsedMs": elapsed_ms,
            "pyramid": pyramid,
        }),
        text,
        is_error: false,
    }
}

// --------------------------------------------------- bundled SplatBench corpus

/// The `benches/reports/splatbench-v0.json` corpus is baked into the binary
/// at build time so the tier-1 server can serve recommendations offline.
/// Refreshing it requires rebuilding the crate — that's intentional, so a
/// stale binary can't quietly recommend numbers that don't match the public
/// leaderboard.
const SPLATBENCH_V0_JSON: &str =
    include_str!("../../../benches/reports/splatbench-v0.json");

/// Compact per-scene record used for similarity scoring. Built once at
/// startup from the bundled JSON.
struct CorpusScene {
    id: String,
    class: String,
    splat_count: u64,
    sh_degree: u32,
    bytes_in: u64,
    /// Full per-scene record from splatbench-v0.json, kept opaque so we can
    /// echo the entire fidelity block to the caller without re-deriving it.
    full: Value,
}

fn load_corpus() -> Vec<CorpusScene> {
    let Ok(doc): Result<Value, _> = serde_json::from_str(SPLATBENCH_V0_JSON) else {
        return Vec::new();
    };
    let Some(scenes) = doc.get("scenes").and_then(Value::as_array) else {
        return Vec::new();
    };
    scenes
        .iter()
        .filter_map(|s| {
            Some(CorpusScene {
                id: s.get("id")?.as_str()?.to_string(),
                class: s
                    .get("class")
                    .and_then(Value::as_str)
                    .unwrap_or("other")
                    .to_string(),
                splat_count: s.get("splatCount")?.as_u64()?,
                sh_degree: s.get("shDegree")?.as_u64()? as u32,
                bytes_in: s.get("bytesIn").and_then(Value::as_u64).unwrap_or(0),
                full: s.clone(),
            })
        })
        .collect()
}

// --------------------------------- splatforge.find_similar_scenes handler

fn tool_find_similar_scenes(args: &Value) -> ToolResult {
    let splat_count = match args.get("splatCount").and_then(Value::as_u64) {
        Some(n) if n > 0 => n,
        _ => return tool_error("splatCount must be a positive integer"),
    };
    let sh_degree = args.get("shDegree").and_then(Value::as_u64).unwrap_or(3) as i64;
    let content_class = args
        .get("contentClass")
        .and_then(Value::as_str)
        .map(str::to_string);
    let k = args
        .get("k")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(3)
        .clamp(1, 16);

    let corpus = load_corpus();
    if corpus.is_empty() {
        return tool_error(
            "bundled SplatBench corpus is empty — rebuild splatforge-mcp \
             with the benches/reports/splatbench-v0.json present"
                .to_string(),
        );
    }

    // Score = distance in log-splatCount space + SH-degree mismatch penalty
    // + class-mismatch penalty. Lower is closer.
    let query_log_n = (splat_count as f64).ln();
    let mut scored: Vec<(f64, &CorpusScene)> = corpus
        .iter()
        .map(|s| {
            let log_n = (s.splat_count as f64).ln();
            let n_dist = (log_n - query_log_n).abs();
            let sh_dist = ((s.sh_degree as i64) - sh_degree).abs() as f64 * 0.5;
            let class_dist = match &content_class {
                Some(c) if c == &s.class => 0.0,
                Some(_) => 0.75,
                None => 0.0,
            };
            (n_dist + sh_dist + class_dist, s)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let neighbors: Vec<Value> = scored
        .iter()
        .take(k)
        .map(|(score, s)| {
            json!({
                "id": s.id,
                "class": s.class,
                "splatCount": s.splat_count,
                "shDegree": s.sh_degree,
                "bytesIn": s.bytes_in,
                "distance": score,
                "scene": s.full,
            })
        })
        .collect();

    let summary_lines: Vec<String> = neighbors
        .iter()
        .map(|n| {
            format!(
                "  - {} ({}, {} splats, dist={:.3})",
                n["id"].as_str().unwrap_or("?"),
                n["class"].as_str().unwrap_or("?"),
                n["splatCount"].as_u64().unwrap_or(0),
                n["distance"].as_f64().unwrap_or(0.0),
            )
        })
        .collect();
    let text = format!(
        "Top {} similar scenes for splatCount={} sh={} class={:?}:\n{}",
        k,
        splat_count,
        sh_degree,
        content_class,
        summary_lines.join("\n"),
    );

    ToolResult {
        json: json!({ "neighbors": neighbors }),
        text,
        is_error: false,
    }
}

fn tool_error(msg: impl Into<String>) -> ToolResult {
    let msg = msg.into();
    ToolResult {
        json: json!({ "error": msg }),
        text: msg,
        is_error: true,
    }
}

/// Pull `input.path` out of the SplatRef argument. We only support
/// `kind == "path"` in this tier; anything else is rejected so we never
/// silently behave differently than the schema advertises.
fn splatref_path(input: &Value) -> Result<PathBuf, String> {
    let kind = input
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "input.kind is required".to_string())?;
    if kind != "path" {
        return Err(format!(
            "this MCP tier only supports input.kind = \"path\"; got \"{kind}\". \
             The url / blob_id / content_b64 forms require the authenticated tier."
        ));
    }
    let p = input
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "input.path is required when kind=path".to_string())?;
    Ok(PathBuf::from(p))
}

fn detect_format(path: &Path) -> Result<&'static str, String> {
    if let Some(fmt) = format_from_extension(path) {
        return Ok(fmt);
    }
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    format_from_magic(&bytes)
        .ok_or_else(|| format!("could not detect format of {}", path.display()))
}

fn load_scene(path: &Path) -> Result<(SplatScene, &'static str), String> {
    let fmt = detect_format(path)?;
    let scene = match fmt {
        "ply" => read_ply(path).map_err(|e| format!("reading PLY: {e}"))?,
        "spz" => read_spz(path).map_err(|e| format!("reading SPZ: {e}"))?,
        "gltf" => read_gltf(path).map_err(|e| format!("reading glTF: {e}"))?,
        "glb" => read_glb(path).map_err(|e| format!("reading GLB: {e}"))?,
        other => return Err(format!("unsupported format: {other}")),
    };
    Ok((scene, fmt))
}

// ------------------------------------------------- splatforge.analyze handler

fn tool_analyze(args: &Value) -> ToolResult {
    let input = match args.get("input") {
        Some(v) => v,
        None => return tool_error("missing required argument: input"),
    };
    let path = match splatref_path(input) {
        Ok(p) => p,
        Err(e) => return tool_error(e),
    };
    if !path.exists() {
        return tool_error(format!("file not found: {}", path.display()));
    }
    let bytes = match std::fs::metadata(&path) {
        Ok(m) => m.len(),
        Err(e) => return tool_error(format!("stat {}: {e}", path.display())),
    };
    let (scene, fmt) = match load_scene(&path) {
        Ok(p) => p,
        Err(e) => return tool_error(e),
    };
    let report = AnalyzeReport::from_scene(&scene, fmt, bytes);
    let json = match serde_json::to_value(&report) {
        Ok(v) => v,
        Err(e) => return tool_error(format!("serialize analyze report: {e}")),
    };
    let text = format!(
        "Analyzed {} ({}): {} splats, {} bytes, SH degree {}, hash {}.",
        path.display(),
        fmt,
        report.splat_count,
        report.file_size,
        report.sh_degree,
        report.hash,
    );
    ToolResult { json, text, is_error: false }
}

// ------------------------------------------- splatforge.list_presets handler

/// Static metadata for each preset. Description and `bestFor` are derived
/// from SPEC-0006 + the preset pipeline definitions in
/// `splatforge_optimize::presets`. `typicalRatio` is a rough number from
/// the SplatBench v0 corpus medians; the predicted-fidelity tools will
/// give a per-scene number, this is just so the LLM has a rule-of-thumb
/// for hero copy.
const PRESET_INFO: &[(&str, &str, f32, &[&str])] = &[
    (
        "lossless-repack",
        "Strip junk splats and Morton-sort — no quantization, byte-identical \
         round-trip on supported formats. Use when the user wants the smallest \
         file that still preserves every coefficient.",
        1.05,
        &["archival", "byte-identical reproducibility"],
    ),
    (
        "web-mobile",
        "Mobile-web target: 15-bit positions, 8-bit scale/rotation, SH degree 0, \
         opacity-prune + floater-prune, LOD chain at 0.5 / 0.25. Typical 5-10× \
         smaller than the input PLY.",
        7.0,
        &["mobile-Safari/Chrome", "low-bandwidth web embed"],
    ),
    (
        "web-desktop",
        "Desktop-web target: 16-bit positions, SH degree 1, opacity-prune + \
         floater-prune. Bigger than `web-mobile` but keeps view-dependent shading.",
        3.5,
        &["desktop-browser viewer", "moderate-bandwidth web"],
    ),
    (
        "quest-browser",
        "Meta Quest browser target: 14-bit positions, SH 0, aggressive opacity \
         threshold, single LOD at 0.3.",
        8.0,
        &["Quest browser", "standalone XR"],
    ),
    (
        "visionos-preview",
        "Apple Vision Pro preview target: 15-bit positions, SH 0, opacity-prune \
         (no floater-prune so transparent content stays intact).",
        5.5,
        &["visionOS preview", "spatial walkthroughs"],
    ),
    (
        "thumbnail-preview",
        "Aggressive size target for thumbnails and previews: 12-bit positions, \
         high opacity threshold, single LOD.",
        12.0,
        &["thumbnails", "list previews"],
    ),
    (
        "quality-max",
        "Same pipeline as `lossless-repack` — alias for clarity when the user \
         asks for 'maximum quality'.",
        1.05,
        &["highest fidelity", "archival masters"],
    ),
    (
        "size-min",
        "Smallest practical output: 12-bit positions, aggressive opacity threshold, \
         floater-prune, LOD chain at 0.25 / 0.1.",
        15.0,
        &["smallest file", "preview/thumbnail flows"],
    ),
];

fn tool_list_presets() -> ToolResult {
    let entries: Vec<Value> = PRESET_INFO
        .iter()
        .map(|(name, desc, ratio, best)| {
            json!({
                "name": name,
                "tier": "free",
                "description": desc,
                "typicalRatio": ratio,
                "bestFor": best,
            })
        })
        .collect();
    // Also surface the paid preset so the LLM knows it exists even though
    // it can't be called from this tier — saves a round trip when the user
    // says "compress this with maximum quality but go aggressive on size."
    let mut presets = entries;
    presets.push(json!({
        "name": "differentiable-repack",
        "tier": "paid",
        "description":
            "Self-distillation pass on an A100 — render the scene from 32 pseudo \
             cameras, train a smaller cloud to match. Typically beats `size-min` by \
             3-15 dB at the same byte budget on opaque scenes. Requires the \
             authenticated MCP tier (set SPLATFORGE_API_KEY and use \
             `splatforge.repack` instead of this tool).",
        "typicalRatio": 22.0,
        "bestFor": ["maximum compression with measurable fidelity"],
    }));
    let text = format!(
        "{} free presets and 1 paid preset available. Use a free preset with \
         `splatforge.optimize`; the paid preset requires the authenticated tier.",
        PRESET_INFO.len()
    );
    ToolResult {
        json: json!({ "presets": presets }),
        text,
        is_error: false,
    }
}

// -------------------------------------------- splatforge.optimize handler

fn tool_optimize(args: &Value) -> ToolResult {
    let input = match args.get("input") {
        Some(v) => v,
        None => return tool_error("missing required argument: input"),
    };
    let path = match splatref_path(input) {
        Ok(p) => p,
        Err(e) => return tool_error(e),
    };
    if !path.exists() {
        return tool_error(format!("file not found: {}", path.display()));
    }
    let preset_name = match args.get("preset").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None => return tool_error("missing required argument: preset"),
    };
    if preset_name == "differentiable-repack" {
        return tool_error(
            "preset 'differentiable-repack' is paid and requires the authenticated tier. \
             Use SPLATFORGE_API_KEY + the `splatforge.repack` tool on the hosted MCP."
                .to_string(),
        );
    }
    let chunked = args
        .get("chunked")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let out_path = match args.get("out").and_then(Value::as_str) {
        Some(s) => PathBuf::from(s),
        None => path.with_extension("optimized.gltf"),
    };

    let t0 = std::time::Instant::now();
    let (mut scene, _) = match load_scene(&path) {
        Ok(p) => p,
        Err(e) => return tool_error(e),
    };
    let pipe = match preset(&preset_name) {
        Ok(p) => p,
        Err(e) => return tool_error(format!("preset error: {e}")),
    };
    let report = match pipe.run(&mut scene) {
        Ok(r) => r,
        Err(e) => return tool_error(format!("optimize failed: {e}")),
    };
    // Mirror the CLI's quantization choices (SPEC-0013).
    let quantize = matches!(
        preset_name.as_str(),
        "web-mobile"
            | "web-desktop"
            | "quest-browser"
            | "visionos-preview"
            | "thumbnail-preview"
            | "size-min"
    );
    let opts = WriteOpts {
        chunked,
        chunk_target_splats: 100_000,
        lod_fractions: vec![1.0],
        quantize,
        compress: None,
    };
    if let Err(e) = write_gltf(&scene, &out_path, &opts) {
        return tool_error(format!("write glTF: {e}"));
    }
    let bytes_in = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    // The .gltf is just a JSON manifest pointing at one or more sibling
    // `.bin` buffers; we have to sum the buffer byteLengths to get the
    // real on-disk payload size.
    let bytes_out = total_gltf_payload_bytes(&out_path);
    let report_value = match serde_json::to_value(&report) {
        Ok(v) => v,
        Err(e) => return tool_error(format!("serialize report: {e}")),
    };
    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let ratio = if bytes_out > 0 {
        bytes_in as f64 / bytes_out as f64
    } else {
        0.0
    };

    let payload = json!({
        "output": {
            "kind": "path",
            "path": out_path.display().to_string(),
        },
        "preset": preset_name,
        "bytesIn": bytes_in,
        "bytesOut": bytes_out,
        "ratio": ratio,
        "elapsedMs": elapsed_ms,
        "report": report_value,
    });
    let text = format!(
        "Optimized {} → {} with preset `{}`. {} → {} bytes ({:.2}× smaller) in {} ms.",
        path.display(),
        out_path.display(),
        preset_name,
        bytes_in,
        bytes_out,
        ratio,
        elapsed_ms,
    );
    ToolResult {
        json: payload,
        text,
        is_error: false,
    }
}

/// Sum the .gltf manifest's own size plus all of its referenced buffer
/// `.bin` files. Falls back to just the .gltf size if anything fails to
/// parse — the caller only uses this for an informational `ratio` field
/// so a degraded answer is better than a tool-call failure.
fn total_gltf_payload_bytes(gltf_path: &Path) -> u64 {
    let manifest_bytes = std::fs::metadata(gltf_path).map(|m| m.len()).unwrap_or(0);
    let parent = gltf_path.parent().unwrap_or_else(|| Path::new("."));
    let Ok(json) = std::fs::read_to_string(gltf_path) else {
        return manifest_bytes;
    };
    let Ok(doc): Result<Value, _> = serde_json::from_str(&json) else {
        return manifest_bytes;
    };
    let mut total = manifest_bytes;
    if let Some(buffers) = doc.get("buffers").and_then(Value::as_array) {
        for b in buffers {
            // Prefer the actual on-disk size; if the URI is a data: URI or
            // we can't stat it, fall back to the declared byteLength.
            if let Some(uri) = b.get("uri").and_then(Value::as_str) {
                if uri.starts_with("data:") {
                    if let Some(len) = b.get("byteLength").and_then(Value::as_u64) {
                        total = total.saturating_add(len);
                    }
                    continue;
                }
                let bin_path = parent.join(uri);
                if let Ok(meta) = std::fs::metadata(&bin_path) {
                    total = total.saturating_add(meta.len());
                    continue;
                }
            }
            if let Some(len) = b.get("byteLength").and_then(Value::as_u64) {
                total = total.saturating_add(len);
            }
        }
    }
    total
}

// ----------------------------------------------------------- request handler

/// Convert a tool result into the MCP `tools/call` response shape:
///
/// ```jsonc
/// {
///   "content": [ { "type": "text", "text": "..." } ],
///   "structuredContent": { ...JSON the LLM can consume directly... },
///   "isError": false
/// }
/// ```
fn tool_response(result: ToolResult) -> Value {
    json!({
        "content": [ { "type": "text", "text": result.text } ],
        "structuredContent": result.json,
        "isError": result.is_error,
    })
}

fn list_tools_response() -> Value {
    let tools: Vec<Value> = TOOLS
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": (t.input_schema)(),
            })
        })
        .collect();
    json!({ "tools": tools })
}

fn handle_request(req: RpcRequest) -> Option<Value> {
    let id = req.id.clone();
    let result = match req.method.as_str() {
        "initialize" => Ok(server_capabilities()),
        // `notifications/initialized` is a one-way ack — no response.
        "notifications/initialized" => return None,
        "tools/list" => Ok(list_tools_response()),
        "tools/call" => {
            let name = req
                .params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let args = req
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Object(Map::new()));
            if name.is_empty() {
                Err((INVALID_PARAMS, "tools/call requires `name`".to_string()))
            } else {
                Ok(tool_response(call_tool(&name, &args)))
            }
        }
        "resources/list" => Ok(list_resources_response()),
        "resources/read" => {
            let uri = req
                .params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if uri.is_empty() {
                Err((INVALID_PARAMS, "resources/read requires `uri`".to_string()))
            } else {
                read_resource_response(uri)
            }
        }
        // Optional MCP methods we don't support — answer empty so clients
        // don't error out on capability negotiation mismatches.
        "prompts/list" => Ok(json!({ "prompts": [] })),
        "ping" => Ok(json!({})),
        other => Err((METHOD_NOT_FOUND, format!("method not found: {other}"))),
    };

    // Notifications have id == None; only return for requests.
    id.as_ref()?;
    Some(match result {
        Ok(result) => serde_json::to_value(RpcResponse {
            jsonrpc: "2.0",
            id: id.unwrap_or(Value::Null),
            result: Some(result),
            error: None,
        })
        .expect("response is serializable"),
        Err((code, message)) => serde_json::to_value(RpcResponse {
            jsonrpc: "2.0",
            id: id.unwrap_or(Value::Null),
            result: None,
            error: Some(RpcError {
                code,
                message,
                data: None,
            }),
        })
        .expect("error response is serializable"),
    })
}

// ----------------------------------------------------------------- run-loop

fn main() -> anyhow::Result<()> {
    // Logs go to stderr so they never collide with the JSON-RPC stream on
    // stdout. The default filter is `info` so clients can debug without
    // an env override; bump down to `warn` once the server is stable.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "splatforge_mcp=info".into()),
        )
        .with_writer(std::io::stderr)
        .try_init();
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "splatforge-mcp ready");

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: RpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = serde_json::to_value(RpcResponse {
                    jsonrpc: "2.0",
                    id: Value::Null,
                    result: None,
                    error: Some(RpcError {
                        code: PARSE_ERROR,
                        message: format!("parse error: {e}"),
                        data: None,
                    }),
                })
                .expect("error response is serializable");
                writeln!(stdout, "{resp}")?;
                stdout.flush()?;
                continue;
            }
        };
        if request.jsonrpc != "2.0" {
            let resp = serde_json::to_value(RpcResponse {
                jsonrpc: "2.0",
                id: request.id.unwrap_or(Value::Null),
                result: None,
                error: Some(RpcError {
                    code: INVALID_REQUEST,
                    message: format!("expected jsonrpc=\"2.0\"; got {:?}", request.jsonrpc),
                    data: None,
                }),
            })
            .expect("error response is serializable");
            writeln!(stdout, "{resp}")?;
            stdout.flush()?;
            continue;
        }
        if let Some(resp) = handle_request(request) {
            writeln!(stdout, "{resp}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

// --------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal JSON-RPC request value with the given method.
    fn req(id: i64, method: &str, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
    }

    fn dispatch(req_value: Value) -> Value {
        let r: RpcRequest = serde_json::from_value(req_value).unwrap();
        handle_request(r).expect("request expects a response")
    }

    #[test]
    fn initialize_advertises_tools_capability() {
        let resp = dispatch(req(1, "initialize", json!({})));
        let result = &resp["result"];
        assert_eq!(result["protocolVersion"], "2025-03-26");
        assert_eq!(result["serverInfo"]["name"], "splatforge-mcp");
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_returns_public_tier_tools() {
        let resp = dispatch(req(2, "tools/list", json!({})));
        let tools = resp["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"splatforge.analyze"));
        assert!(names.contains(&"splatforge.list_presets"));
        assert!(names.contains(&"splatforge.optimize"));
    }

    #[test]
    fn list_presets_includes_paid_preset_marker() {
        let resp = dispatch(req(
            3,
            "tools/call",
            json!({ "name": "splatforge.list_presets", "arguments": {} }),
        ));
        let presets = resp["result"]["structuredContent"]["presets"]
            .as_array()
            .unwrap();
        assert_eq!(presets.len(), PRESET_INFO.len() + 1);
        let paid = presets.iter().find(|p| p["tier"] == "paid").unwrap();
        assert_eq!(paid["name"], "differentiable-repack");
    }

    #[test]
    fn analyze_rejects_non_path_splatref() {
        let resp = dispatch(req(
            4,
            "tools/call",
            json!({
                "name": "splatforge.analyze",
                "arguments": {
                    "input": { "kind": "url", "url": "https://example.com/x.ply" }
                }
            }),
        ));
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("only supports"));
    }

    #[test]
    fn optimize_rejects_paid_preset() {
        let resp = dispatch(req(
            5,
            "tools/call",
            json!({
                "name": "splatforge.optimize",
                "arguments": {
                    "input": { "kind": "path", "path": "/dev/null" },
                    "preset": "differentiable-repack"
                }
            }),
        ));
        assert_eq!(resp["result"]["isError"], true);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("paid"));
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let resp = dispatch(req(6, "splatforge/teleport", json!({})));
        assert_eq!(resp["error"]["code"], METHOD_NOT_FOUND);
    }

    #[test]
    fn generate_lod_pyramid_emits_correct_splat_counts() {
        // Use a synthetic SplatBench scene that's small enough for the
        // test suite — 8K-splat product-scan PLY ships with the repo.
        let tmp = tempfile::tempdir().unwrap();
        let out_dir = tmp.path().to_path_buf();
        let resp = dispatch(req(
            13,
            "tools/call",
            json!({
                "name": "splatforge.generate_lod_pyramid",
                "arguments": {
                    "input": {
                        "kind": "path",
                        "path": "../../benches/scenes/splatbench_specular_proxy.ply"
                    },
                    "ratios": [1.0, 0.5, 0.25],
                    "out_dir": out_dir.display().to_string(),
                }
            }),
        ));
        assert_eq!(resp["result"]["isError"], false, "tool errored: {:?}", resp);
        let pyramid = resp["result"]["structuredContent"]["pyramid"]
            .as_array()
            .unwrap();
        assert_eq!(pyramid.len(), 3);
        let splats_in =
            resp["result"]["structuredContent"]["splatsIn"].as_u64().unwrap();
        // r=1.0 must emit exactly the input count; r=0.5 ≈ half; r=0.25 ≈ quarter.
        let counts: Vec<u64> = pyramid
            .iter()
            .map(|p| p["splatCount"].as_u64().unwrap())
            .collect();
        assert_eq!(counts[0], splats_in);
        let half = splats_in / 2;
        assert!(
            (counts[1] as i64 - half as i64).abs() <= 1,
            "r=0.5 emitted {} vs expected ~{}", counts[1], half
        );
        let quarter = splats_in / 4;
        assert!(
            (counts[2] as i64 - quarter as i64).abs() <= 1,
            "r=0.25 emitted {} vs expected ~{}", counts[2], quarter
        );
        // Every emitted file exists and is non-empty.
        for level in pyramid {
            let p = level["path"].as_str().unwrap();
            let m = std::fs::metadata(p).unwrap();
            assert!(m.len() > 1024, "{p} is suspiciously small ({} bytes)", m.len());
        }
    }

    #[test]
    fn generate_lod_pyramid_rejects_out_of_range_ratio() {
        let resp = dispatch(req(
            14,
            "tools/call",
            json!({
                "name": "splatforge.generate_lod_pyramid",
                "arguments": {
                    "input": { "kind": "path", "path": "/dev/null" },
                    "ratios": [1.5],
                }
            }),
        ));
        assert_eq!(resp["result"]["isError"], true);
    }

    #[test]
    fn find_similar_scenes_returns_neighbors_in_distance_order() {
        let resp = dispatch(req(
            10,
            "tools/call",
            json!({
                "name": "splatforge.find_similar_scenes",
                "arguments": {
                    "splatCount": 1_200_000,
                    "shDegree": 3,
                    "contentClass": "indoor-real-estate",
                    "k": 3
                }
            }),
        ));
        assert_eq!(resp["result"]["isError"], false);
        let neighbors = resp["result"]["structuredContent"]["neighbors"]
            .as_array()
            .unwrap();
        assert_eq!(neighbors.len(), 3);
        // Distance is monotonically non-decreasing.
        let dists: Vec<f64> = neighbors
            .iter()
            .map(|n| n["distance"].as_f64().unwrap())
            .collect();
        for w in dists.windows(2) {
            assert!(w[0] <= w[1] + 1e-9, "distances not sorted: {dists:?}");
        }
        // Top match should be bonsai (1.16M splats, indoor-real-estate, sh 3).
        assert_eq!(neighbors[0]["id"], "bonsai_mipnerf360_iter7k");
    }

    #[test]
    fn resources_list_includes_splatbench() {
        let resp = dispatch(req(11, "resources/list", json!({})));
        let resources = resp["result"]["resources"].as_array().unwrap();
        assert!(resources.iter().any(|r| r["uri"] == "splatbench://v0"));
    }

    #[test]
    fn resources_read_returns_corpus_json() {
        let resp = dispatch(req(
            12,
            "resources/read",
            json!({ "uri": "splatbench://v0" }),
        ));
        let contents = resp["result"]["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["mimeType"], "application/json");
        let text = contents[0]["text"].as_str().unwrap();
        // Sanity: this should at least parse and contain 16 scenes.
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["scenes"].as_array().unwrap().len(), 16);
    }

    #[test]
    fn notifications_dont_produce_responses() {
        let r: RpcRequest = serde_json::from_value(
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized", "params": {} }),
        )
        .unwrap();
        assert!(handle_request(r).is_none());
    }
}
