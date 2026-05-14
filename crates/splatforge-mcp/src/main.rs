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
use splatforge_ply::read_ply;
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
            "tools": { "listChanged": false }
        }
    })
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
        other => tool_error(format!("unknown tool: {other}")),
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
        // Optional MCP methods we don't support — answer empty so clients
        // don't error out on capability negotiation mismatches.
        "resources/list" => Ok(json!({ "resources": [] })),
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
    fn notifications_dont_produce_responses() {
        let r: RpcRequest = serde_json::from_value(
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized", "params": {} }),
        )
        .unwrap();
        assert!(handle_request(r).is_none());
    }
}
