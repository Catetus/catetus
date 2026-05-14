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
use thiserror::Error;

/// Alias to keep both naming conventions working.
pub type WriteOptions = WriteOpts;

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
}

impl Default for WriteOpts {
    fn default() -> Self {
        Self {
            chunked: false,
            chunk_target_splats: 100_000,
            lod_fractions: vec![1.0],
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
    component_type: u32, // 5126 = FLOAT
    count: usize,
    #[serde(rename = "type")]
    accessor_type: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GltfRoot {
    asset: GltfAsset,
    #[serde(rename = "extensionsUsed", skip_serializing_if = "Vec::is_empty", default)]
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
const SF_INDEX: &str = "SF_spatial_streaming_index";
const FLOAT: u32 = 5126;

fn add_view_acc(
    root: &mut GltfRoot,
    buffer_idx: usize,
    offset: &mut usize,
    count: usize,
    byte_len: usize,
    accessor_ty: &str,
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
        component_type: FLOAT,
        count,
        accessor_type: accessor_ty.to_string(),
    });
    *offset += byte_len;
    acc
}

/// Write a scene as `<dir>/scene.gltf` plus one or more `.bin` files under
/// `<dir>/buffers/`. `path` is the output `.gltf` path.
pub fn write_gltf(scene: &SplatScene, path: &Path, opts: &WriteOpts) -> Result<(), GltfError> {
    let dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
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

    let mut chunk_records: Vec<serde_json::Value> = Vec::new();
    let mut primitives: Vec<serde_json::Value> = Vec::new();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let buf_bytes = pack_chunk(chunk);
        let buf_name = format!("buffers/chunk_{chunk_idx:04}.bin");
        let buf_path = dir.join(&buf_name);
        fs::write(&buf_path, &buf_bytes)?;
        let buffer_idx = root.buffers.len();
        root.buffers.push(GltfBuffer {
            byte_length: buf_bytes.len(),
            uri: Some(buf_name.clone()),
        });

        // accessors: POSITION, _ROTATION, _SCALE, _OPACITY, _COLOR_DC, optional _COLOR_SH
        let n = chunk.len();
        let mut offset = 0usize;
        let pos_acc = add_view_acc(&mut root, buffer_idx, &mut offset, n, n * 12, "VEC3");
        let rot_acc = add_view_acc(&mut root, buffer_idx, &mut offset, n, n * 16, "VEC4");
        let scale_acc = add_view_acc(&mut root, buffer_idx, &mut offset, n, n * 12, "VEC3");
        let op_acc = add_view_acc(&mut root, buffer_idx, &mut offset, n, n * 4, "SCALAR");
        let dc_acc = add_view_acc(&mut root, buffer_idx, &mut offset, n, n * 12, "VEC3");

        // SH: emit only if any splat carries SH.
        let has_sh = chunk.iter().any(|s| matches!(s.color, Color::Sh { .. }));
        let mut khr_attrs = serde_json::json!({
            "POSITION": pos_acc,
            "_ROTATION": rot_acc,
            "_SCALE": scale_acc,
            "_OPACITY": op_acc,
            "_COLOR_DC": dc_acc,
        });
        let mut sh_degree = 0u8;
        if has_sh {
            let coeffs_per_splat = 45;
            let sh_acc = add_view_acc(
                &mut root,
                buffer_idx,
                &mut offset,
                n,
                n * coeffs_per_splat * 4,
                "SCALAR",
            );
            khr_attrs
                .as_object_mut()
                .unwrap()
                .insert("_COLOR_SH".to_string(), serde_json::json!(sh_acc));
            sh_degree = chunk.iter().map(|s| s.color.degree()).max().unwrap_or(0);
        }

        primitives.push(serde_json::json!({
            "extensions": {
                KHR: {
                    "attributes": khr_attrs,
                    "shDegree": sh_degree,
                }
            }
        }));

        if opts.chunked {
            // bbox
            let (mn, mx) = chunk_bbox(chunk);
            let hash = blake3::hash(&buf_bytes).to_hex().to_string();
            chunk_records.push(serde_json::json!({
                "buffer": buffer_idx,
                "byteOffset": 0,
                "byteLength": buf_bytes.len(),
                "splatCount": chunk.len(),
                "bbox": { "min": mn, "max": mx },
                "lod": 0,
                "checksum": format!("blake3:{hash}"),
                "loadPriority": chunk_idx,
            }));
        }
    }

    root.meshes.push(serde_json::json!({ "primitives": primitives }));

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

fn pack_chunk(chunk: &[Splat]) -> Vec<u8> {
    let n = chunk.len();
    let has_sh = chunk.iter().any(|s| matches!(s.color, Color::Sh { .. }));
    let coeffs_per_splat = if has_sh { 45 } else { 0 };
    let cap = n * (12 + 16 + 12 + 4 + 12 + coeffs_per_splat * 4);
    let mut out = Vec::with_capacity(cap);
    // POSITION VEC3
    for s in chunk {
        for v in s.position {
            out.write_f32::<LittleEndian>(v).unwrap();
        }
    }
    // _ROTATION VEC4
    for s in chunk {
        for v in s.rotation {
            out.write_f32::<LittleEndian>(v).unwrap();
        }
    }
    // _SCALE VEC3
    for s in chunk {
        for v in s.scale {
            out.write_f32::<LittleEndian>(v).unwrap();
        }
    }
    // _OPACITY scalar
    for s in chunk {
        out.write_f32::<LittleEndian>(s.opacity).unwrap();
    }
    // _COLOR_DC VEC3
    for s in chunk {
        let dc = match &s.color {
            Color::Rgb(c) => *c,
            Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
        };
        for v in dc {
            out.write_f32::<LittleEndian>(v).unwrap();
        }
    }
    // optional SH
    if has_sh {
        for s in chunk {
            match &s.color {
                Color::Sh { coeffs, .. } => {
                    for i in 0..45 {
                        let v = if i + 3 < coeffs.len() {
                            coeffs[i + 3]
                        } else {
                            0.0
                        };
                        out.write_f32::<LittleEndian>(v).unwrap();
                    }
                }
                Color::Rgb(_) => {
                    for _ in 0..45 {
                        out.write_f32::<LittleEndian>(0.0).unwrap();
                    }
                }
            }
        }
    }
    out
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
    let dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    read_gltf_str(&raw, &dir)
}

fn read_gltf_str(raw: &str, base_dir: &Path) -> Result<SplatScene, GltfError> {
    let root: GltfRoot = serde_json::from_str(raw)?;
    if !root.extensions_used.iter().any(|e| e == KHR) {
        return Err(GltfError::MissingExtension);
    }
    let mesh = root
        .meshes
        .first()
        .ok_or_else(|| GltfError::Malformed("no meshes".to_string()))?;
    let prim = mesh
        .get("primitives")
        .and_then(|p| p.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| GltfError::Malformed("no primitives".to_string()))?;
    let ext = prim
        .get("extensions")
        .and_then(|e| e.get(KHR))
        .ok_or(GltfError::MissingExtension)?;
    let attrs = ext
        .get("attributes")
        .ok_or_else(|| GltfError::Malformed("no attributes".to_string()))?;

    let get_idx = |name: &str| -> Option<usize> {
        attrs.get(name).and_then(|v| v.as_u64()).map(|n| n as usize)
    };
    let pos_acc = get_idx("POSITION").ok_or(GltfError::MissingExtension)?;
    let rot_acc = get_idx("_ROTATION").ok_or(GltfError::MissingExtension)?;
    let scale_acc = get_idx("_SCALE").ok_or(GltfError::MissingExtension)?;
    let op_acc = get_idx("_OPACITY").ok_or(GltfError::MissingExtension)?;
    let dc_acc = get_idx("_COLOR_DC").ok_or(GltfError::MissingExtension)?;
    let sh_acc = get_idx("_COLOR_SH");

    // Load buffer bytes.
    let mut buffers_bytes: Vec<Vec<u8>> = Vec::with_capacity(root.buffers.len());
    for b in &root.buffers {
        let uri = b
            .uri
            .as_ref()
            .ok_or_else(|| GltfError::BufferNotFound("no uri".to_string()))?;
        let bp = base_dir.join(uri);
        let data = fs::read(&bp)
            .map_err(|_| GltfError::BufferNotFound(bp.display().to_string()))?;
        buffers_bytes.push(data);
    }

    let read_floats = |acc_idx: usize, comps: usize| -> Result<Vec<f32>, GltfError> {
        let acc = &root.accessors[acc_idx];
        let bv = &root.buffer_views[acc.buffer_view];
        let data = &buffers_bytes[bv.buffer];
        let bytes = &data[bv.byte_offset..bv.byte_offset + bv.byte_length];
        let n = acc.count * comps;
        if bytes.len() < n * 4 {
            return Err(GltfError::Malformed("accessor out of range".to_string()));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let chunk = &bytes[i * 4..i * 4 + 4];
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    };

    let positions = read_floats(pos_acc, 3)?;
    let rotations = read_floats(rot_acc, 4)?;
    let scales = read_floats(scale_acc, 3)?;
    let opacities = read_floats(op_acc, 1)?;
    let dc = read_floats(dc_acc, 3)?;
    let n = opacities.len();
    let sh = if let Some(idx) = sh_acc {
        Some(read_floats(idx, 45)?)
    } else {
        None
    };

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let color = if let Some(ref sh) = sh {
            let mut coeffs = Vec::with_capacity(48);
            coeffs.extend_from_slice(&dc[i * 3..i * 3 + 3]);
            coeffs.extend_from_slice(&sh[i * 45..i * 45 + 45]);
            Color::Sh {
                degree: 3,
                coeffs,
            }
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
    let dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let used: Vec<String> = value
        .get("extensionsUsed")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
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
    let mut checksum_ok = true;
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
            let expected = chunk
                .get("checksum")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            let buf = buffers.get(buffer_idx).ok_or_else(|| {
                GltfError::Malformed(format!("chunk {i} references missing buffer"))
            })?;
            let uri = buf
                .get("uri")
                .and_then(|u| u.as_str())
                .ok_or_else(|| GltfError::Malformed("buffer has no uri".to_string()))?;
            let data = fs::read(dir.join(uri))
                .map_err(|_| GltfError::BufferNotFound(uri.to_string()))?;
            let slice = &data[byte_offset..byte_offset + byte_length];
            let actual = format!("blake3:{}", blake3::hash(slice).to_hex());
            if actual != expected {
                checksum_ok = false;
                return Err(GltfError::ChecksumMismatch(i));
            }
            splat_count += chunk
                .get("splatCount")
                .and_then(|s| s.as_u64())
                .unwrap_or(0) as usize;
        }
    } else {
        // Walk accessors to find _OPACITY count as splat count.
        if let Some(prim) = value
            .get("meshes")
            .and_then(|m| m.as_array())
            .and_then(|m| m.first())
            .and_then(|m| m.get("primitives"))
            .and_then(|p| p.as_array())
            .and_then(|p| p.first())
        {
            if let Some(idx) = prim
                .get("extensions")
                .and_then(|e| e.get(KHR))
                .and_then(|e| e.get("attributes"))
                .and_then(|a| a.get("_OPACITY"))
                .and_then(|v| v.as_u64())
            {
                if let Some(acc) = value
                    .get("accessors")
                    .and_then(|a| a.as_array())
                    .and_then(|a| a.get(idx as usize))
                {
                    splat_count = acc.get("count").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
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
fn build_single_buffer_gltf(scene: &SplatScene) -> Result<(GltfRoot, Vec<u8>), GltfError> {
    let chunk: &[Splat] = scene.splats.as_slice();
    let buf_bytes = pack_chunk(chunk);

    let mut root = GltfRoot {
        asset: GltfAsset {
            version: "2.0".to_string(),
            generator: Some("splatforge-gltf".to_string()),
        },
        extensions_used: vec![KHR.to_string()],
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
    let mut offset = 0usize;
    let pos_acc = add_view_acc(&mut root, 0, &mut offset, n, n * 12, "VEC3");
    let rot_acc = add_view_acc(&mut root, 0, &mut offset, n, n * 16, "VEC4");
    let scale_acc = add_view_acc(&mut root, 0, &mut offset, n, n * 12, "VEC3");
    let op_acc = add_view_acc(&mut root, 0, &mut offset, n, n * 4, "SCALAR");
    let dc_acc = add_view_acc(&mut root, 0, &mut offset, n, n * 12, "VEC3");

    let has_sh = chunk.iter().any(|s| matches!(s.color, Color::Sh { .. }));
    let mut khr_attrs = serde_json::json!({
        "POSITION": pos_acc,
        "_ROTATION": rot_acc,
        "_SCALE": scale_acc,
        "_OPACITY": op_acc,
        "_COLOR_DC": dc_acc,
    });
    let mut sh_degree = 0u8;
    if has_sh {
        let coeffs_per_splat = 45;
        let sh_acc = add_view_acc(
            &mut root,
            0,
            &mut offset,
            n,
            n * coeffs_per_splat * 4,
            "SCALAR",
        );
        khr_attrs
            .as_object_mut()
            .ok_or_else(|| GltfError::Malformed("attrs not object".to_string()))?
            .insert("_COLOR_SH".to_string(), serde_json::json!(sh_acc));
        sh_degree = chunk.iter().map(|s| s.color.degree()).max().unwrap_or(0);
    }

    root.meshes.push(serde_json::json!({
        "primitives": [{
            "extensions": {
                KHR: {
                    "attributes": khr_attrs,
                    "shDegree": sh_degree,
                }
            }
        }]
    }));

    Ok((root, buf_bytes))
}

fn pad_to_4(buf: &mut Vec<u8>, pad_byte: u8) {
    while buf.len() % 4 != 0 {
        buf.push(pad_byte);
    }
}

/// Write a `SplatScene` as a binary glTF (`.glb`) container with the JSON and
/// the splat data embedded as a single chunk. Chunked output is not supported
/// for GLB; pass `opts.chunked == false`.
pub fn write_glb(scene: &SplatScene, path: &Path, opts: &WriteOptions) -> Result<(), GltfError> {
    if opts.chunked {
        return Err(GltfError::GlbChunkedUnsupported);
    }
    let (root, bin) = build_single_buffer_gltf(scene)?;
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
        return Err(GltfError::Malformed(format!("unsupported GLB version {version}")));
    }
    let total = read_u32_le(bytes, 8)? as usize;
    if bytes.len() < total {
        return Err(GltfError::Malformed("GLB length exceeds buffer".to_string()));
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
    let json_bytes = json_bytes.ok_or_else(|| GltfError::Malformed("missing JSON chunk".to_string()))?;
    let bin_bytes = bin_bytes.ok_or_else(|| GltfError::Malformed("missing BIN chunk".to_string()))?;

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
    let mesh = root
        .meshes
        .first()
        .ok_or_else(|| GltfError::Malformed("no meshes".to_string()))?;
    let prim = mesh
        .get("primitives")
        .and_then(|p| p.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| GltfError::Malformed("no primitives".to_string()))?;
    let ext = prim
        .get("extensions")
        .and_then(|e| e.get(KHR))
        .ok_or(GltfError::MissingExtension)?;
    let attrs = ext
        .get("attributes")
        .ok_or_else(|| GltfError::Malformed("no attributes".to_string()))?;

    let get_idx = |name: &str| -> Option<usize> {
        attrs.get(name).and_then(|v| v.as_u64()).map(|n| n as usize)
    };
    let pos_acc = get_idx("POSITION").ok_or(GltfError::MissingExtension)?;
    let rot_acc = get_idx("_ROTATION").ok_or(GltfError::MissingExtension)?;
    let scale_acc = get_idx("_SCALE").ok_or(GltfError::MissingExtension)?;
    let op_acc = get_idx("_OPACITY").ok_or(GltfError::MissingExtension)?;
    let dc_acc = get_idx("_COLOR_DC").ok_or(GltfError::MissingExtension)?;
    let sh_acc = get_idx("_COLOR_SH");

    let read_floats = |acc_idx: usize, comps: usize| -> Result<Vec<f32>, GltfError> {
        let acc = &root.accessors[acc_idx];
        let bv = &root.buffer_views[acc.buffer_view];
        if bv.buffer != 0 {
            return Err(GltfError::Malformed("GLB only supports buffer 0".to_string()));
        }
        if bin.len() < bv.byte_offset + bv.byte_length {
            return Err(GltfError::Malformed("accessor out of range".to_string()));
        }
        let bytes = &bin[bv.byte_offset..bv.byte_offset + bv.byte_length];
        let n = acc.count * comps;
        if bytes.len() < n * 4 {
            return Err(GltfError::Malformed("accessor under-sized".to_string()));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let c = &bytes[i * 4..i * 4 + 4];
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        Ok(out)
    };

    let positions = read_floats(pos_acc, 3)?;
    let rotations = read_floats(rot_acc, 4)?;
    let scales = read_floats(scale_acc, 3)?;
    let opacities = read_floats(op_acc, 1)?;
    let dc = read_floats(dc_acc, 3)?;
    let n = opacities.len();
    let sh = if let Some(idx) = sh_acc {
        Some(read_floats(idx, 45)?)
    } else {
        None
    };

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let color = if let Some(ref sh) = sh {
            let mut coeffs = Vec::with_capacity(48);
            coeffs.extend_from_slice(&dc[i * 3..i * 3 + 3]);
            coeffs.extend_from_slice(&sh[i * 45..i * 45 + 45]);
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
