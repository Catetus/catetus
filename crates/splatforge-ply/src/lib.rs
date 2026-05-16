#![deny(clippy::all)]
//! Inria-style 3DGS PLY → SplatForge IR.
//!
//! See `specs/0002-ply-ingest.md`.
//!
//! The binary-body decoder uses a single hoisted field-offset table plus a
//! flat byte-slice walk (no per-scalar `Cursor` allocation, no per-splat
//! `Vec<f32>` scratch buffer, no O(P²) property-name lookups). The file
//! itself is read via `memmap2::Mmap` to avoid materialising a 30 GiB
//! `Vec<u8>` on Sweet-Corals-class PLYs.

use std::fs;
use std::io::{BufRead, Cursor, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};
use splatforge_core::{Color, CoordinateSystem, Splat, SplatScene, TemporalMode};
use thiserror::Error;

pub mod progressive;
pub use progressive::{
    decode_progressive, decode_progressive_file, encode_progressive, encode_progressive_file,
    importance_scores_from_ply, read_mgs2_header, Mgs2Header, MGS2_MAGIC, MGS2_PREFIX_LEN,
    MGS2_VERSION,
};

/// All errors produced by PLY ingest.
#[derive(Debug, Error)]
pub enum PlyError {
    /// Underlying IO failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// File did not begin with the `ply` magic header.
    #[error("not a PLY file (missing 'ply' magic)")]
    NotAPly,
    /// Big-endian PLYs are not currently supported.
    #[error("unsupported_endian: big-endian PLYs are not supported")]
    UnsupportedEndian,
    /// Header could not be parsed.
    #[error("malformed header: {0}")]
    MalformedHeader(String),
    /// A required Gaussian-splat field (e.g. `x`, `rot_0`, `f_dc_0`) was absent.
    #[error("missing_required_field: {0}")]
    MissingRequiredField(String),
    /// The body ended before all declared splats could be read.
    #[error("truncated_payload: input ended mid-record")]
    TruncatedPayload,
    /// An ASCII value could not be parsed as a float.
    #[error("invalid ascii float: {0}")]
    InvalidAsciiFloat(String),
}

/// PLY storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    BinaryLE,
    Ascii,
}

#[derive(Debug, Clone, Copy)]
enum ScalarTy {
    F32,
    F64,
    I32,
    U32,
    I16,
    U16,
    I8,
    U8,
}

impl ScalarTy {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            _ => return None,
        })
    }

    fn size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 => 8,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
struct Property {
    name: String,
    ty: ScalarTy,
}

#[derive(Debug, Clone)]
struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

#[derive(Debug)]
struct Header {
    format: Format,
    elements: Vec<Element>,
    body_offset: usize,
}

fn parse_header(bytes: &[u8]) -> Result<Header, PlyError> {
    let mut cursor = Cursor::new(bytes);
    let mut line = String::new();
    cursor.read_line(&mut line)?;
    if line.trim_end() != "ply" {
        return Err(PlyError::NotAPly);
    }

    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    let mut current: Option<Element> = None;

    loop {
        line.clear();
        let n = cursor.read_line(&mut line)?;
        if n == 0 {
            return Err(PlyError::MalformedHeader("unexpected eof".to_string()));
        }
        let trimmed = line.trim_end();
        if trimmed == "end_header" {
            break;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts.as_slice() {
            ["format", fmt, _ver] => {
                format = Some(match *fmt {
                    "binary_little_endian" => Format::BinaryLE,
                    "ascii" => Format::Ascii,
                    "binary_big_endian" => return Err(PlyError::UnsupportedEndian),
                    other => {
                        return Err(PlyError::MalformedHeader(format!(
                            "unknown format: {other}"
                        )))
                    }
                });
            }
            ["comment", ..] => {}
            ["element", name, count] => {
                if let Some(elem) = current.take() {
                    elements.push(elem);
                }
                let count: usize = count
                    .parse()
                    .map_err(|_| PlyError::MalformedHeader("bad element count".to_string()))?;
                current = Some(Element {
                    name: (*name).to_string(),
                    count,
                    properties: Vec::new(),
                });
            }
            ["property", "list", ..] => {
                return Err(PlyError::MalformedHeader(
                    "list properties not supported".to_string(),
                ));
            }
            ["property", ty, name] => {
                let ty = ScalarTy::parse(ty)
                    .ok_or_else(|| PlyError::MalformedHeader(format!("bad property type {ty}")))?;
                if let Some(elem) = current.as_mut() {
                    elem.properties.push(Property {
                        name: (*name).to_string(),
                        ty,
                    });
                } else {
                    return Err(PlyError::MalformedHeader(
                        "property outside element".to_string(),
                    ));
                }
            }
            [] => {}
            _ => {} // ignore unknown directives
        }
    }
    if let Some(elem) = current.take() {
        elements.push(elem);
    }
    let format = format.ok_or_else(|| PlyError::MalformedHeader("no format line".to_string()))?;
    let body_offset = cursor.position() as usize;
    Ok(Header {
        format,
        elements,
        body_offset,
    })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n == 0.0 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// Required PLY field names for an Inria-style 3DGS file.
const REQUIRED: &[&str] = &[
    "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
];

fn parse_vertex_element(header: &Header) -> Result<&Element, PlyError> {
    header
        .elements
        .iter()
        .find(|e| e.name == "vertex")
        .ok_or_else(|| PlyError::MalformedHeader("no 'vertex' element".to_string()))
}

fn check_required(elem: &Element) -> Result<(), PlyError> {
    for req in REQUIRED {
        if !elem.properties.iter().any(|p| p.name == *req) {
            // Friendly message for rotation
            if req.starts_with("rot_") {
                return Err(PlyError::MissingRequiredField(format!(
                    "missing required rotation fields ({req})"
                )));
            }
            return Err(PlyError::MissingRequiredField(format!(
                "missing required field {req}"
            )));
        }
    }
    Ok(())
}

/// Precomputed location of one scalar field within the per-vertex record.
#[derive(Debug, Clone, Copy)]
struct FieldLoc {
    /// Byte offset within a single vertex record.
    offset: usize,
    /// Storage type on disk.
    ty: ScalarTy,
}

/// Hoisted addressing table for an Inria vertex element. Each `Option<FieldLoc>`
/// resolves once at parse-start; the inner loop just indexes.
#[derive(Debug, Clone)]
struct VertexLayout {
    stride: usize,
    /// Required: x, y, z.
    pos: [FieldLoc; 3],
    /// Required: rot_0 (w), rot_1 (x), rot_2 (y), rot_3 (z).
    rot: [FieldLoc; 4],
    /// Required: scale_0..2 (log space).
    scale: [FieldLoc; 3],
    /// Required: opacity (logit space).
    opacity: FieldLoc,
    /// Required: f_dc_0..2.
    dc: [FieldLoc; 3],
    /// Optional: f_rest_* in property order.
    f_rest: Vec<FieldLoc>,
}

impl VertexLayout {
    fn build(vertex: &Element) -> Result<Self, PlyError> {
        // Property byte offsets within the vertex record.
        let mut offsets = Vec::with_capacity(vertex.properties.len());
        let mut acc = 0usize;
        for p in &vertex.properties {
            offsets.push(acc);
            acc += p.ty.size();
        }
        let stride = acc;

        let find = |name: &str| -> Result<FieldLoc, PlyError> {
            vertex
                .properties
                .iter()
                .enumerate()
                .find(|(_, p)| p.name == name)
                .map(|(i, p)| FieldLoc {
                    offset: offsets[i],
                    ty: p.ty,
                })
                .ok_or_else(|| PlyError::MissingRequiredField(name.to_string()))
        };

        let pos = [find("x")?, find("y")?, find("z")?];
        let rot = [find("rot_0")?, find("rot_1")?, find("rot_2")?, find("rot_3")?];
        let scale = [find("scale_0")?, find("scale_1")?, find("scale_2")?];
        let opacity = find("opacity")?;
        let dc = [find("f_dc_0")?, find("f_dc_1")?, find("f_dc_2")?];

        // f_rest in property-declaration order.
        let f_rest: Vec<FieldLoc> = vertex
            .properties
            .iter()
            .enumerate()
            .filter(|(_, p)| p.name.starts_with("f_rest_"))
            .map(|(i, p)| FieldLoc {
                offset: offsets[i],
                ty: p.ty,
            })
            .collect();

        Ok(Self {
            stride,
            pos,
            rot,
            scale,
            opacity,
            dc,
            f_rest,
        })
    }
}

/// Read a single scalar from a fixed-size byte slice.
/// All branches are inlined; the per-row loop has no function-call overhead.
#[inline(always)]
fn load_scalar(row: &[u8], loc: FieldLoc) -> f32 {
    let off = loc.offset;
    // Safety / correctness: callers guarantee `row.len() >= stride` so
    // `row[off..off+size]` is in bounds. `try_into` + `from_le_bytes` is
    // unaligned-safe on every target; the optimiser collapses the array copy
    // to a single mov for f32/i32 and four bytes for f64.
    match loc.ty {
        ScalarTy::F32 => {
            let b: [u8; 4] = row[off..off + 4].try_into().unwrap();
            f32::from_le_bytes(b)
        }
        ScalarTy::F64 => {
            let b: [u8; 8] = row[off..off + 8].try_into().unwrap();
            f64::from_le_bytes(b) as f32
        }
        ScalarTy::I32 => {
            let b: [u8; 4] = row[off..off + 4].try_into().unwrap();
            i32::from_le_bytes(b) as f32
        }
        ScalarTy::U32 => {
            let b: [u8; 4] = row[off..off + 4].try_into().unwrap();
            u32::from_le_bytes(b) as f32
        }
        ScalarTy::I16 => {
            let b: [u8; 2] = row[off..off + 2].try_into().unwrap();
            i16::from_le_bytes(b) as f32
        }
        ScalarTy::U16 => {
            let b: [u8; 2] = row[off..off + 2].try_into().unwrap();
            u16::from_le_bytes(b) as f32
        }
        ScalarTy::I8 => (row[off] as i8) as f32,
        ScalarTy::U8 => row[off] as f32,
    }
}

/// Decode one vertex record into a `Splat`. Hot path; everything constant
/// across the file (offsets, SH-degree, f_rest table) is resolved into
/// `layout`/`sh_degree` before this is called.
#[inline(always)]
fn decode_row(
    row: &[u8],
    layout: &VertexLayout,
    sh_degree: u8,
    total_coeffs_per_channel: usize,
) -> Splat {
    let position = [
        load_scalar(row, layout.pos[0]),
        load_scalar(row, layout.pos[1]),
        load_scalar(row, layout.pos[2]),
    ];

    // PLY rotation order is (w, x, y, z); IR is (x, y, z, w).
    let rw = load_scalar(row, layout.rot[0]);
    let rx = load_scalar(row, layout.rot[1]);
    let ry = load_scalar(row, layout.rot[2]);
    let rz = load_scalar(row, layout.rot[3]);
    let rotation = normalize_quat([rx, ry, rz, rw]);

    let scale = [
        load_scalar(row, layout.scale[0]).exp(),
        load_scalar(row, layout.scale[1]).exp(),
        load_scalar(row, layout.scale[2]).exp(),
    ];
    let opacity = sigmoid(load_scalar(row, layout.opacity));

    let dc = [
        load_scalar(row, layout.dc[0]),
        load_scalar(row, layout.dc[1]),
        load_scalar(row, layout.dc[2]),
    ];

    let color = if layout.f_rest.is_empty() {
        Color::Rgb(dc)
    } else {
        let mut coeffs = Vec::with_capacity(3 * total_coeffs_per_channel);
        coeffs.extend_from_slice(&dc);
        for loc in &layout.f_rest {
            coeffs.push(load_scalar(row, *loc));
        }
        Color::Sh {
            degree: sh_degree,
            coeffs,
        }
    };

    Splat {
        position,
        rotation,
        scale,
        opacity,
        color,
    }
}

/// Threshold above which we shard the body across the global rayon pool.
/// Below this many splats, the thread-launch and join overhead dominates.
const PARALLEL_THRESHOLD: usize = 256 * 1024;

/// Fast binary-PLY decoder. Hoists all property-name lookups and per-scalar
/// `Cursor` allocations out of the inner loop, then walks the body as a
/// contiguous slice of vertex records. For large files (`> PARALLEL_THRESHOLD`
/// splats) the work is sharded across the global rayon pool; ordering is
/// preserved because each shard writes to a pre-sized slice owned by its
/// chunk index.
fn read_binary(bytes: &[u8], header: &Header, vertex_idx: usize) -> Result<Vec<Splat>, PlyError> {
    let vertex = &header.elements[vertex_idx];
    let layout = VertexLayout::build(vertex)?;
    let stride = layout.stride;
    let need = stride
        .checked_mul(vertex.count)
        .ok_or(PlyError::TruncatedPayload)?;
    let body_end = header
        .body_offset
        .checked_add(need)
        .ok_or(PlyError::TruncatedPayload)?;
    if bytes.len() < body_end {
        return Err(PlyError::TruncatedPayload);
    }
    let body = &bytes[header.body_offset..body_end];

    // SH degree resolution is constant across the file.
    let rest_per_channel = layout.f_rest.len() / 3;
    let total_coeffs_per_channel = rest_per_channel + 1;
    let sh_degree: u8 = match total_coeffs_per_channel {
        1 => 0,
        4 => 1,
        9 => 2,
        16 => 3,
        _ => 0,
    };

    let count = vertex.count;

    if count < PARALLEL_THRESHOLD {
        let mut splats: Vec<Splat> = Vec::with_capacity(count);
        for i in 0..count {
            let row_off = i * stride;
            let row = &body[row_off..row_off + stride];
            splats.push(decode_row(row, &layout, sh_degree, total_coeffs_per_channel));
        }
        return Ok(splats);
    }

    // Parallel decode preserving input order: write directly into a
    // pre-allocated `Vec<Splat>` whose slots we partition by chunk.
    use rayon::prelude::*;
    // Aim for ~32 chunks per rayon thread so per-chunk variance evens out
    // without paying the thread-launch overhead too many times.
    let threads = rayon::current_num_threads().max(1);
    let target_chunks = (threads * 32).max(1);
    let chunk_rows = count.div_ceil(target_chunks).max(8 * 1024);

    // We collect chunks in order via `flat_map`-style concatenation of
    // per-chunk Vecs. `IndexedParallelIterator::collect_into_vec` would also
    // work but allocates an intermediate; the per-chunk-Vec approach is
    // simpler and reuses each chunk's contiguous capacity.
    let chunks: Vec<Vec<Splat>> = (0..count)
        .into_par_iter()
        .step_by(chunk_rows)
        .map(|start| {
            let end = (start + chunk_rows).min(count);
            let mut out = Vec::with_capacity(end - start);
            for i in start..end {
                let row_off = i * stride;
                let row = &body[row_off..row_off + stride];
                out.push(decode_row(row, &layout, sh_degree, total_coeffs_per_channel));
            }
            out
        })
        .collect();

    let mut splats: Vec<Splat> = Vec::with_capacity(count);
    for c in chunks {
        splats.extend(c);
    }
    Ok(splats)
}

fn read_ascii(bytes: &[u8], header: &Header, vertex_idx: usize) -> Result<Vec<Splat>, PlyError> {
    let vertex = &header.elements[vertex_idx];
    let f_rest_indices: Vec<usize> = vertex
        .properties
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.name.strip_prefix("f_rest_").map(|_| i))
        .collect();
    let body = &bytes[header.body_offset..];
    let text = std::str::from_utf8(body)
        .map_err(|_| PlyError::MalformedHeader("non-utf8 ascii body".to_string()))?;
    let mut lines = text.lines();
    let mut splats = Vec::with_capacity(vertex.count);
    for _ in 0..vertex.count {
        let line = lines.next().ok_or(PlyError::TruncatedPayload)?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < vertex.properties.len() {
            return Err(PlyError::TruncatedPayload);
        }
        let mut values: Vec<f32> = Vec::with_capacity(vertex.properties.len());
        for (i, _prop) in vertex.properties.iter().enumerate() {
            let v: f32 = toks[i]
                .parse()
                .map_err(|_| PlyError::InvalidAsciiFloat(toks[i].to_string()))?;
            values.push(v);
        }
        splats.push(build_splat(vertex, &values, &f_rest_indices));
    }
    Ok(splats)
}

fn build_splat(vertex: &Element, values: &[f32], f_rest_indices: &[usize]) -> Splat {
    let lookup = |name: &str| -> f32 {
        let i = vertex
            .properties
            .iter()
            .position(|p| p.name == name)
            .unwrap();
        values[i]
    };
    let pos = [lookup("x"), lookup("y"), lookup("z")];
    // PLY rotation order is (w, x, y, z); IR is (x, y, z, w).
    let rw = lookup("rot_0");
    let rx = lookup("rot_1");
    let ry = lookup("rot_2");
    let rz = lookup("rot_3");
    let rot = normalize_quat([rx, ry, rz, rw]);
    let scale = [
        lookup("scale_0").exp(),
        lookup("scale_1").exp(),
        lookup("scale_2").exp(),
    ];
    let opacity = sigmoid(lookup("opacity"));
    let dc = [lookup("f_dc_0"), lookup("f_dc_1"), lookup("f_dc_2")];
    let color = if f_rest_indices.is_empty() {
        Color::Rgb(dc)
    } else {
        // Determine SH degree from f_rest count: total coeffs per splat = 3 * ((deg+1)^2 - 1)
        let rest_per_channel = f_rest_indices.len() / 3;
        let total_coeffs_with_dc = rest_per_channel + 1; // per channel
        let degree = match total_coeffs_with_dc {
            1 => 0,
            4 => 1,
            9 => 2,
            16 => 3,
            _ => 0,
        };
        let mut coeffs = Vec::with_capacity(3 * total_coeffs_with_dc);
        // Pack as DC then rest, interleaved [r, g, b] per band.
        coeffs.extend_from_slice(&dc);
        for &idx in f_rest_indices {
            coeffs.push(values[idx]);
        }
        Color::Sh { degree, coeffs }
    };
    Splat {
        position: pos,
        rotation: rot,
        scale,
        opacity,
        color,
    }
}

/// Read a PLY file from `path`.
///
/// Uses `mmap` to avoid materialising a multi-gigabyte heap buffer on large
/// files. For ASCII PLYs, the body is parsed lazily from the mapped slice.
/// Falls back to a buffered read on platforms where `mmap` fails (e.g. some
/// network filesystems).
pub fn read_ply(path: &Path) -> Result<SplatScene, PlyError> {
    let file = fs::File::open(path)?;
    // Safety: we only read the mapping, never mutate, and the borrow ends
    // before the returned `SplatScene` (which owns its own `Vec<Splat>`).
    match unsafe { memmap2::Mmap::map(&file) } {
        Ok(map) => read_ply_bytes(&map),
        Err(_) => {
            // Rare path — e.g. zero-length file or pipe. Fall back to a copy.
            let bytes = fs::read(path)?;
            read_ply_bytes(&bytes)
        }
    }
}

/// Read a PLY from an in-memory buffer.
pub fn read_ply_bytes(bytes: &[u8]) -> Result<SplatScene, PlyError> {
    let header = parse_header(bytes)?;
    let vertex_idx = header
        .elements
        .iter()
        .position(|e| e.name == "vertex")
        .ok_or_else(|| PlyError::MalformedHeader("no 'vertex' element".to_string()))?;
    let vertex = parse_vertex_element(&header)?;
    check_required(vertex)?;

    let splats = match header.format {
        Format::BinaryLE => read_binary(bytes, &header, vertex_idx)?,
        Format::Ascii => read_ascii(bytes, &header, vertex_idx)?,
    };

    Ok(SplatScene {
        splats,
        coordinate_system: CoordinateSystem::default(),
        semantic_labels: None,
        temporal_mode: TemporalMode::Static,
        lods: None,
    })
}

fn logit(p: f32) -> f32 {
    // Inverse of sigmoid. Clamp slightly inside (0, 1) to avoid -inf/+inf.
    let p = p.clamp(f32::EPSILON, 1.0 - f32::EPSILON);
    (p / (1.0 - p)).ln()
}

fn ln_scale(s: f32) -> f32 {
    // Inverse of `exp` applied on import. Clamp positive to avoid -inf.
    let s = s.max(f32::EPSILON);
    s.ln()
}

/// Write a `SplatScene` to a binary little-endian Inria-3DGS PLY file on disk.
pub fn write_ply(scene: &SplatScene, path: &Path) -> Result<(), PlyError> {
    let bytes = write_ply_bytes(scene)?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Encode a `SplatScene` as a binary little-endian Inria-3DGS PLY byte stream.
pub fn write_ply_bytes(scene: &SplatScene) -> Result<Vec<u8>, PlyError> {
    let mut out = Vec::new();
    // Header: same field order as fixtures/tiny/basic_binary.ply.
    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {}\n", scene.splats.len()));
    let mut props: Vec<&'static str> = Vec::new();
    props.extend(["x", "y", "z", "nx", "ny", "nz"]);
    // f_dc_0..2
    let f_dc = ["f_dc_0", "f_dc_1", "f_dc_2"];
    props.extend(f_dc);
    // f_rest_0..44 (45 entries)
    let f_rest_names: Vec<String> = (0..45).map(|i| format!("f_rest_{i}")).collect();
    // We'll write property lines manually in the same loop to keep alloc free.
    for name in &props {
        header.push_str(&format!("property float {name}\n"));
    }
    for name in &f_rest_names {
        header.push_str(&format!("property float {name}\n"));
    }
    for name in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");
    out.write_all(header.as_bytes())?;

    for s in &scene.splats {
        // x, y, z
        for v in s.position {
            out.write_f32::<LittleEndian>(v)?;
        }
        // nx, ny, nz — not stored in IR, write zeros.
        for _ in 0..3 {
            out.write_f32::<LittleEndian>(0.0)?;
        }
        // f_dc_0..2
        let (dc, rest): ([f32; 3], Vec<f32>) = match &s.color {
            Color::Rgb(c) => (*c, Vec::new()),
            Color::Sh { coeffs, .. } => {
                let dc = [
                    coeffs.first().copied().unwrap_or(0.0),
                    coeffs.get(1).copied().unwrap_or(0.0),
                    coeffs.get(2).copied().unwrap_or(0.0),
                ];
                let rest: Vec<f32> = coeffs.iter().skip(3).copied().collect();
                (dc, rest)
            }
        };
        for v in dc {
            out.write_f32::<LittleEndian>(v)?;
        }
        // f_rest_0..44 — pad to 45 with zeros.
        for i in 0..45 {
            let v = rest.get(i).copied().unwrap_or(0.0);
            out.write_f32::<LittleEndian>(v)?;
        }
        // opacity — logit (inverse of sigmoid applied on import).
        out.write_f32::<LittleEndian>(logit(s.opacity))?;
        // scale_0..2 — ln (inverse of exp applied on import).
        for v in s.scale {
            out.write_f32::<LittleEndian>(ln_scale(v))?;
        }
        // rot_0..3 = (w, x, y, z) on disk; IR holds (x, y, z, w).
        out.write_f32::<LittleEndian>(s.rotation[3])?;
        out.write_f32::<LittleEndian>(s.rotation[0])?;
        out.write_f32::<LittleEndian>(s.rotation[1])?;
        out.write_f32::<LittleEndian>(s.rotation[2])?;
    }
    Ok(out)
}
