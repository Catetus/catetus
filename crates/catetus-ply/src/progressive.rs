//! Progressive MGS2 bitstream — Phase 1.
//!
//! Encodes an Inria-style binary-little-endian 3DGS PLY into a single
//! byte-streamable bitstream where splats are emitted in **descending
//! importance order** (`opacity * det(scale)^{2/3}`). A partial download
//! (`first N bytes`) is itself a valid `.mgs2` payload that decodes to a
//! coarse-but-watchable PLY containing the top-K most-important splats.
//!
//! Phase-1 scope (per `docs/perf/progressive-bitstream-spec.md`):
//!   * Single bitstream, no layer boundaries — every additional byte
//!     contributes to quality at a fixed `record_size` granularity.
//!   * Per-splat records are byte-for-byte the original PLY vertex
//!     records, just reordered. Round-trip at 100 % is therefore the
//!     identity on the splat *multiset* and identical-precision on each
//!     record — the only difference vs the input PLY is splat order.
//!   * Importance score:
//!     opacity     = sigmoid(opacity_logit)          // raw `opacity` field
//!     scale_xyz   = exp(scale_logits)
//!     score       = opacity * det(scale)^{2/3}
//!     This matches novel-3 mixed-CRF, the D1 importance gate, and the
//!     PRoGS / PCGS literature's standard cheap proxy.
//!
//! Out of scope here:
//!   * Layered range-coded residuals (Phase 2+).
//!   * Bit-plane refinement of attribute quantization (Phase 2+).
//!   * Per-layer CDFs / range coder hookup.
//!
//! Format (`.mgs2`, little-endian throughout):
//!
//! ```text
//! [4]  magic           = b"MGS2"
//! [4]  version u32     = 1
//! [4]  flags   u32     = 0       // reserved, must be 0 in v1
//! [8]  n_splats u64
//! [4]  record_size u32           // bytes per splat vertex record
//! [4]  ply_header_len u32        // length of the original ASCII PLY header
//! [H]  ply_header_bytes          // verbatim ASCII PLY header (binary_little_endian)
//! [N * record_size] splat records in descending-importance order
//! ```
//!
//! The first 28 bytes are the fixed prefix; the original PLY header
//! follows (typically ~1.5 KB for Inria 3DGS with f_rest_0..44); after
//! that the payload is the splat records.
//!
//! A partial decode for byte-cutoff `cut` works as:
//!   * Require `cut >= prefix_len + ply_header_len`. If smaller, the
//!     prefix itself isn't fully downloaded — return empty / error.
//!   * Compute `usable_records = (cut - prefix_len - ply_header_len) / record_size`,
//!     clamped to `n_splats`.
//!   * Emit a PLY whose header is the verbatim original header with the
//!     `element vertex <N>` line rewritten to `usable_records`, followed
//!     by the first `usable_records * record_size` bytes of the payload.
//!
//! Because we store the *original* PLY header verbatim, the partial PLY
//! is wire-compatible with any Inria-3DGS PLY reader.

use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::PlyError;

/// `.mgs2` magic — four ASCII bytes at offset 0.
pub const MGS2_MAGIC: &[u8; 4] = b"MGS2";
/// Bitstream format version. Increment on incompatible payload changes.
pub const MGS2_VERSION: u32 = 1;

/// Fixed-size prefix at the head of every `.mgs2` file:
///   magic(4) + version(4) + flags(4) + n_splats(8) + record_size(4) + ply_header_len(4)
pub const MGS2_PREFIX_LEN: usize = 4 + 4 + 4 + 8 + 4 + 4;

/// Header summary returned by [`read_mgs2_header`]. Sufficient to compute
/// any partial-decode byte cutoff without touching the payload.
#[derive(Debug, Clone)]
pub struct Mgs2Header {
    /// Bitstream version (currently always 1).
    pub version: u32,
    /// Reserved flags. Currently always 0.
    pub flags: u32,
    /// Number of splats in the full bitstream (= splats in the source PLY).
    pub n_splats: u64,
    /// Bytes per per-splat vertex record (PLY stride for the `vertex` element).
    pub record_size: u32,
    /// ASCII PLY header (verbatim, ending in `end_header\n`).
    pub ply_header: Vec<u8>,
    /// Offset (bytes from start of file) at which the splat-record payload begins.
    pub payload_offset: u64,
    /// Total payload length in bytes (`n_splats * record_size`).
    pub payload_len: u64,
}

/// Parsed minimal view of the source PLY header. Used to locate `x/y/z`,
/// `scale_*`, `opacity` columns when computing importance.
struct PlyHeaderInfo {
    body_offset: usize,
    record_size: usize,
    n_vertices: usize,
    /// Offset (bytes) into a single record for each field we care about.
    /// `None` if absent — we error out on missing required fields.
    off_opacity: Option<usize>,
    off_scale: [Option<usize>; 3],
}

/// Parse the ASCII PLY header from a binary-little-endian Inria 3DGS file.
///
/// Returns the body offset (= start of binary vertex records), the
/// per-record byte stride, the vertex count, and the byte offsets within
/// a record for `opacity` and `scale_{0,1,2}`. We don't bother walking
/// quaternion / SH columns at this stage — they're not on the importance
/// score's hot path.
fn parse_inria_ply_header(bytes: &[u8]) -> Result<PlyHeaderInfo, PlyError> {
    // The reader in lib.rs is already battle-tested; we don't re-use it
    // here because we want raw byte offsets *inside* each record, which
    // it doesn't surface. The parser below is a tight subset.
    let mut cursor = Cursor::new(bytes);
    let mut line = String::new();
    use std::io::BufRead;
    cursor.read_line(&mut line)?;
    if line.trim_end() != "ply" {
        return Err(PlyError::NotAPly);
    }

    // We only support `vertex` element with all scalar properties. Other
    // elements (e.g. an `end_header` comment, or `face` for non-splat
    // PLYs) cause us to fall back via an early error so the caller can
    // route through the legacy path.
    let mut format_ok = false;
    let mut n_vertices: Option<usize> = None;
    // (name, size_bytes)
    let mut props: Vec<(String, usize)> = Vec::new();
    let mut in_vertex = false;
    let mut other_element = false;

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
            ["format", "binary_little_endian", _] => format_ok = true,
            ["format", "binary_big_endian", _] => return Err(PlyError::UnsupportedEndian),
            ["format", "ascii", _] => {
                return Err(PlyError::MalformedHeader(
                    "ascii PLY not supported by progressive encoder".to_string(),
                ));
            }
            ["element", "vertex", count] => {
                let c: usize = count
                    .parse()
                    .map_err(|_| PlyError::MalformedHeader("bad vertex count".to_string()))?;
                n_vertices = Some(c);
                in_vertex = true;
                other_element = false;
            }
            ["element", _name, _count] => {
                in_vertex = false;
                other_element = true;
            }
            ["property", ty, name] => {
                if other_element {
                    // We don't model other elements; surfaces as truncated
                    // record stride later. Reject up front.
                    return Err(PlyError::MalformedHeader(
                        "progressive encoder only supports vertex-only PLYs".to_string(),
                    ));
                }
                if !in_vertex {
                    return Err(PlyError::MalformedHeader(
                        "property outside element".to_string(),
                    ));
                }
                let size = match *ty {
                    "float" | "float32" | "int" | "int32" | "uint" | "uint32" => 4,
                    "double" | "float64" => 8,
                    "short" | "int16" | "ushort" | "uint16" => 2,
                    "char" | "int8" | "uchar" | "uint8" => 1,
                    other => {
                        return Err(PlyError::MalformedHeader(format!(
                            "unsupported property type {other}"
                        )));
                    }
                };
                props.push(((*name).to_string(), size));
            }
            ["property", "list", ..] => {
                return Err(PlyError::MalformedHeader(
                    "list properties not supported".to_string(),
                ));
            }
            ["comment", ..] | [] => {}
            _ => {} // tolerate forward-compat directives
        }
    }

    if !format_ok {
        return Err(PlyError::MalformedHeader(
            "missing binary_little_endian format line".to_string(),
        ));
    }
    let n_vertices =
        n_vertices.ok_or_else(|| PlyError::MalformedHeader("no vertex element".to_string()))?;
    if props.is_empty() {
        return Err(PlyError::MalformedHeader(
            "vertex element has no properties".to_string(),
        ));
    }

    // Walk props once to compute record stride + needed field offsets.
    // We require every property to be a 4-byte float in practice (Inria
    // dump uses f32 everywhere), but we don't *enforce* that — the
    // progressive payload is byte-for-byte the original record, so any
    // valid binary PLY round-trips. Importance needs f32 for opacity +
    // scale_{0,1,2}; we'll only error out if those four columns aren't
    // f32 (size != 4).
    let mut record_size = 0usize;
    let mut off_opacity = None;
    let mut off_scale: [Option<usize>; 3] = [None, None, None];
    for (name, size) in &props {
        let off = record_size;
        record_size += *size;
        match name.as_str() {
            "opacity" => {
                if *size != 4 {
                    return Err(PlyError::MalformedHeader(
                        "opacity column must be float32 for importance scoring".to_string(),
                    ));
                }
                off_opacity = Some(off);
            }
            "scale_0" => {
                if *size != 4 {
                    return Err(PlyError::MalformedHeader(
                        "scale_0 column must be float32 for importance scoring".to_string(),
                    ));
                }
                off_scale[0] = Some(off);
            }
            "scale_1" => {
                if *size != 4 {
                    return Err(PlyError::MalformedHeader(
                        "scale_1 column must be float32 for importance scoring".to_string(),
                    ));
                }
                off_scale[1] = Some(off);
            }
            "scale_2" => {
                if *size != 4 {
                    return Err(PlyError::MalformedHeader(
                        "scale_2 column must be float32 for importance scoring".to_string(),
                    ));
                }
                off_scale[2] = Some(off);
            }
            _ => {}
        }
    }

    Ok(PlyHeaderInfo {
        body_offset: cursor.position() as usize,
        record_size,
        n_vertices,
        off_opacity,
        off_scale,
    })
}

#[inline]
fn read_f32_at(record: &[u8], off: usize) -> f32 {
    // SAFETY: callers always pass `off+4 <= record.len()` because the
    // offsets are derived from the same header that produced
    // `record_size = record.len()`.
    let bytes: [u8; 4] = record[off..off + 4].try_into().unwrap();
    f32::from_le_bytes(bytes)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Importance score for one vertex record: `opacity * det(scale)^{2/3}`.
///
/// PLY stores `opacity` as a pre-sigmoid logit and `scale_{x,y,z}` as
/// pre-exp log-scales (Inria convention; matches `read_inria_ply`).
/// `det(diag(sx, sy, sz)) = sx*sy*sz` so the cube root drops the
/// `^{1/3}`; we then raise to the second power for the `^{2/3}` exponent.
///
/// `NaN` records (e.g. degenerate splats) get score 0 so they sort to
/// the tail rather than poisoning the comparator.
fn importance_score(record: &[u8], info: &PlyHeaderInfo) -> f32 {
    let off_o = match info.off_opacity {
        Some(o) => o,
        None => return 0.0,
    };
    let off_s = match (info.off_scale[0], info.off_scale[1], info.off_scale[2]) {
        (Some(a), Some(b), Some(c)) => [a, b, c],
        _ => return 0.0,
    };
    let opacity_logit = read_f32_at(record, off_o);
    let opacity = sigmoid(opacity_logit);
    let sx = read_f32_at(record, off_s[0]).exp();
    let sy = read_f32_at(record, off_s[1]).exp();
    let sz = read_f32_at(record, off_s[2]).exp();
    let det = sx * sy * sz;
    if !det.is_finite() || det <= 0.0 || !opacity.is_finite() {
        return 0.0;
    }
    // det^{2/3} = (det^{1/3})^2
    let cube_root = det.cbrt();
    let score = opacity * cube_root * cube_root;
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

/// Compute the per-splat importance scores for a full PLY byte buffer.
/// Exposed for benchmarks and for downstream code (LODGE, novel-3) that
/// wants the same scoring as the progressive encoder.
pub fn importance_scores_from_ply(ply_bytes: &[u8]) -> Result<Vec<f32>, PlyError> {
    let info = parse_inria_ply_header(ply_bytes)?;
    let needed = info.body_offset + info.record_size * info.n_vertices;
    if ply_bytes.len() < needed {
        return Err(PlyError::TruncatedPayload);
    }
    let body = &ply_bytes[info.body_offset..info.body_offset + info.record_size * info.n_vertices];
    let mut scores = Vec::with_capacity(info.n_vertices);
    for i in 0..info.n_vertices {
        let rec = &body[i * info.record_size..(i + 1) * info.record_size];
        scores.push(importance_score(rec, &info));
    }
    Ok(scores)
}

/// Encode an Inria 3DGS PLY (`ply_bytes`) into a progressive `.mgs2`
/// bitstream. The output begins with the fixed prefix and a verbatim
/// copy of the source PLY header, followed by `n_splats * record_size`
/// bytes of splat records ordered by descending importance.
///
/// Memory usage is O(`n_splats`) for the permutation + scores; the
/// per-record bodies are streamed by index from `ply_bytes` so we don't
/// duplicate the 287 MB bonsai scene in RAM. Sort is `sort_unstable_by`
/// on `(NotNan-wrapped, original_idx)`; the secondary key keeps the
/// permutation deterministic across runs.
pub fn encode_progressive(ply_bytes: &[u8]) -> Result<Vec<u8>, PlyError> {
    let info = parse_inria_ply_header(ply_bytes)?;
    let n = info.n_vertices;
    let stride = info.record_size;
    let body_start = info.body_offset;
    let body_end = body_start
        .checked_add(stride.checked_mul(n).ok_or(PlyError::TruncatedPayload)?)
        .ok_or(PlyError::TruncatedPayload)?;
    if ply_bytes.len() < body_end {
        return Err(PlyError::TruncatedPayload);
    }
    let body = &ply_bytes[body_start..body_end];

    // Score every record. We sort *indices* by descending score; the
    // secondary key (the index itself) makes the comparator total even
    // when scores tie.
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut scores: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let rec = &body[i * stride..(i + 1) * stride];
        scores.push(importance_score(rec, &info));
    }
    perm.sort_unstable_by(|&a, &b| {
        let sa = scores[a as usize];
        let sb = scores[b as usize];
        // Descending: largest first. NaN is filtered to 0.0 above, so
        // partial_cmp returns Some(_) for every pair.
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let ply_header_bytes = &ply_bytes[..body_start];
    let ply_header_len = ply_header_bytes.len() as u32;
    let payload_len = (n as u64) * (stride as u64);
    let mut out =
        Vec::with_capacity(MGS2_PREFIX_LEN + ply_header_bytes.len() + payload_len as usize);

    // Prefix.
    out.write_all(MGS2_MAGIC)?;
    out.write_u32::<LittleEndian>(MGS2_VERSION)?;
    out.write_u32::<LittleEndian>(0)?; // flags
    out.write_u64::<LittleEndian>(n as u64)?;
    out.write_u32::<LittleEndian>(stride as u32)?;
    out.write_u32::<LittleEndian>(ply_header_len)?;
    // Verbatim PLY header.
    out.write_all(ply_header_bytes)?;
    // Payload: records in descending-importance order.
    for &idx in &perm {
        let i = idx as usize;
        out.write_all(&body[i * stride..(i + 1) * stride])?;
    }
    Ok(out)
}

/// Read the fixed-size prefix + PLY header of a `.mgs2` bitstream
/// without parsing the payload. Cheap O(1).
pub fn read_mgs2_header(mgs2_bytes: &[u8]) -> Result<Mgs2Header, PlyError> {
    if mgs2_bytes.len() < MGS2_PREFIX_LEN {
        return Err(PlyError::TruncatedPayload);
    }
    let mut cur = Cursor::new(mgs2_bytes);
    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != MGS2_MAGIC {
        return Err(PlyError::MalformedHeader(format!(
            "not a MGS2 file (magic = {:?})",
            magic
        )));
    }
    let version = cur.read_u32::<LittleEndian>()?;
    if version != MGS2_VERSION {
        return Err(PlyError::MalformedHeader(format!(
            "unsupported MGS2 version {version}; expected {MGS2_VERSION}"
        )));
    }
    let flags = cur.read_u32::<LittleEndian>()?;
    if flags != 0 {
        return Err(PlyError::MalformedHeader(format!(
            "unsupported MGS2 flags {flags:#x}; expected 0 in v1"
        )));
    }
    let n_splats = cur.read_u64::<LittleEndian>()?;
    let record_size = cur.read_u32::<LittleEndian>()?;
    let ply_header_len = cur.read_u32::<LittleEndian>()?;

    let header_start = MGS2_PREFIX_LEN;
    let header_end = header_start
        .checked_add(ply_header_len as usize)
        .ok_or_else(|| PlyError::MalformedHeader("ply_header_len overflow".into()))?;
    if mgs2_bytes.len() < header_end {
        return Err(PlyError::TruncatedPayload);
    }
    let ply_header = mgs2_bytes[header_start..header_end].to_vec();

    let payload_offset = header_end as u64;
    let payload_len = n_splats
        .checked_mul(record_size as u64)
        .ok_or_else(|| PlyError::MalformedHeader("payload size overflow".into()))?;

    Ok(Mgs2Header {
        version,
        flags,
        n_splats,
        record_size,
        ply_header,
        payload_offset,
        payload_len,
    })
}

/// Rewrite the `element vertex <N>` line in a PLY header with a new
/// count. Returns the rewritten header bytes.
///
/// We do this textually rather than reparsing because the source header
/// may contain comments / unknown directives we want to preserve
/// verbatim (a future-proof decoder for `KHR_gaussian_splatting` PLYs,
/// for example).
fn rewrite_vertex_count(header_bytes: &[u8], new_count: u64) -> Result<Vec<u8>, PlyError> {
    let text = std::str::from_utf8(header_bytes)
        .map_err(|_| PlyError::MalformedHeader("non-utf8 PLY header in .mgs2".into()))?;
    let mut out = String::with_capacity(text.len() + 16);
    let mut rewrote = false;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_end();
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if !rewrote && matches!(parts.as_slice(), ["element", "vertex", _]) {
            // Preserve newline-ending of the original line.
            let nl = if line.ends_with('\n') { "\n" } else { "" };
            out.push_str(&format!("element vertex {new_count}{nl}"));
            rewrote = true;
        } else {
            out.push_str(line);
        }
    }
    if !rewrote {
        return Err(PlyError::MalformedHeader(
            "no `element vertex` line in PLY header".into(),
        ));
    }
    Ok(out.into_bytes())
}

/// Decode a `.mgs2` bitstream up to `cut_bytes` of input, producing a
/// valid Inria 3DGS PLY containing only the most-important splats that
/// fit in the cut.
///
/// `cut_bytes == None` means "decode the whole stream". Pass
/// `Some(byte_count)` for a partial decode. When `cut_bytes` is smaller
/// than the fixed prefix + PLY header, no splats are emitted but a
/// valid empty-vertex PLY is still returned (header rewritten to
/// `element vertex 0`).
///
/// Strict mode: if `cut_bytes` falls partway through a splat record,
/// the trailing partial record is dropped. The output is always a
/// well-formed PLY whose `vertex` count matches the byte count of the
/// payload region.
pub fn decode_progressive(mgs2_bytes: &[u8], cut_bytes: Option<u64>) -> Result<Vec<u8>, PlyError> {
    let h = read_mgs2_header(mgs2_bytes)?;
    let cut = cut_bytes.unwrap_or(mgs2_bytes.len() as u64);
    // Records that fully fit inside `cut` bytes:
    let usable = if cut <= h.payload_offset {
        0u64
    } else {
        let payload_bytes_available = (cut - h.payload_offset).min(h.payload_len);
        payload_bytes_available / h.record_size as u64
    }
    .min(h.n_splats);

    let new_header = rewrite_vertex_count(&h.ply_header, usable)?;
    let mut out = Vec::with_capacity(new_header.len() + (usable as usize) * h.record_size as usize);
    out.extend_from_slice(&new_header);
    let payload_start = h.payload_offset as usize;
    let payload_take = (usable as usize) * h.record_size as usize;
    let end = payload_start
        .checked_add(payload_take)
        .ok_or(PlyError::TruncatedPayload)?;
    if mgs2_bytes.len() < end {
        // Defensive: if `cut` was capped at the file length but somehow
        // we still computed a usable count past the buffer, refuse.
        return Err(PlyError::TruncatedPayload);
    }
    out.extend_from_slice(&mgs2_bytes[payload_start..end]);
    Ok(out)
}

/// Read a PLY from disk, encode to `.mgs2`, write to disk.
pub fn encode_progressive_file(input: &Path, output: &Path) -> Result<(), PlyError> {
    let bytes = fs::read(input)?;
    let mgs2 = encode_progressive(&bytes)?;
    fs::write(output, mgs2)?;
    Ok(())
}

/// Read a `.mgs2` from disk, decode (optionally truncated), write PLY.
pub fn decode_progressive_file(
    input: &Path,
    output: &Path,
    partial_bytes: Option<u64>,
) -> Result<(), PlyError> {
    let bytes = fs::read(input)?;
    // If the user passed a cut larger than the file, clamp to file size
    // — this matches "downloaded the whole thing" semantics.
    let cut = partial_bytes.map(|c| c.min(bytes.len() as u64));
    let ply = decode_progressive(&bytes, cut)?;
    fs::write(output, ply)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny binary-LE Inria-style PLY with `n` splats.
    /// Splat `i` gets opacity logit = i (so sigmoid is monotone in i) and
    /// log-scale = i * 0.1 (so det(scale)^{2/3} is monotone in i). The
    /// importance score is therefore strictly increasing in `i`, so the
    /// encoder must emit them in reverse order.
    fn synthetic_ply(n: usize) -> Vec<u8> {
        // Mirrors fixtures/tiny/basic_binary.ply's column layout.
        let header = format!(
            "ply\n\
             format binary_little_endian 1.0\n\
             element vertex {n}\n\
             property float x\n\
             property float y\n\
             property float z\n\
             property float scale_0\n\
             property float scale_1\n\
             property float scale_2\n\
             property float rot_0\n\
             property float rot_1\n\
             property float rot_2\n\
             property float rot_3\n\
             property float opacity\n\
             property float f_dc_0\n\
             property float f_dc_1\n\
             property float f_dc_2\n\
             end_header\n"
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(header.as_bytes());
        for i in 0..n {
            let f = i as f32;
            let record = [
                f,       // x
                f * 0.5, // y
                -f,      // z
                f * 0.1, // scale_0 (log)
                f * 0.1, // scale_1
                f * 0.1, // scale_2
                1.0,     // rot_0 (w)
                0.0,     // rot_1 (x)
                0.0,     // rot_2 (y)
                0.0,     // rot_3 (z)
                f,       // opacity logit
                0.1,     // f_dc_0
                0.2,     // f_dc_1
                0.3,     // f_dc_2
            ];
            for v in record {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn importance_score_monotone_in_synthetic() {
        let ply = synthetic_ply(5);
        let scores = importance_scores_from_ply(&ply).unwrap();
        for i in 1..scores.len() {
            assert!(
                scores[i] > scores[i - 1],
                "scores must be strictly increasing in synthetic: {scores:?}",
            );
        }
    }

    #[test]
    fn encode_then_decode_full_is_record_identical() {
        let ply = synthetic_ply(7);
        let mgs2 = encode_progressive(&ply).unwrap();
        let header = read_mgs2_header(&mgs2).unwrap();
        assert_eq!(header.n_splats, 7);
        assert_eq!(header.version, MGS2_VERSION);
        let decoded_ply = decode_progressive(&mgs2, None).unwrap();
        // Same splat *multiset* — same total bytes (header may differ
        // textually only if vertex count was rewritten, which at 100 %
        // it isn't).
        let src_info = parse_inria_ply_header(&ply).unwrap();
        let dec_info = parse_inria_ply_header(&decoded_ply).unwrap();
        assert_eq!(src_info.n_vertices, dec_info.n_vertices);
        assert_eq!(src_info.record_size, dec_info.record_size);

        // The decoded body is the source body permuted by importance.
        // Collect both, sort each by raw bytes, and assert equality.
        let src_body = &ply[src_info.body_offset..];
        let dec_body = &decoded_ply[dec_info.body_offset..];
        let stride = src_info.record_size;
        let mut src_records: Vec<&[u8]> = (0..src_info.n_vertices)
            .map(|i| &src_body[i * stride..(i + 1) * stride])
            .collect();
        let mut dec_records: Vec<&[u8]> = (0..dec_info.n_vertices)
            .map(|i| &dec_body[i * stride..(i + 1) * stride])
            .collect();
        src_records.sort();
        dec_records.sort();
        assert_eq!(src_records, dec_records);

        // And the order in `decoded_ply` is descending importance.
        let scores = importance_scores_from_ply(&decoded_ply).unwrap();
        for i in 1..scores.len() {
            assert!(
                scores[i] <= scores[i - 1] + 1e-6,
                "decoded scene must be in descending-importance order: {scores:?}",
            );
        }
    }

    #[test]
    fn partial_decode_keeps_top_importance_splats() {
        let n = 100usize;
        let ply = synthetic_ply(n);
        let mgs2 = encode_progressive(&ply).unwrap();
        let header = read_mgs2_header(&mgs2).unwrap();

        // Three cuts: 10 %, 25 %, 50 % of the *bitstream* size.
        let total = mgs2.len() as u64;
        let cuts = [total / 10, total / 4, total / 2, total];
        let mut last_count = 0u64;
        let mut last_min_score = f32::INFINITY;
        for cut in cuts {
            let ply_out = decode_progressive(&mgs2, Some(cut)).unwrap();
            let info = parse_inria_ply_header(&ply_out).unwrap();
            // Every cut must produce a valid, fully-emittable PLY.
            assert_eq!(info.record_size, header.record_size as usize);
            // Monotone: larger cut never drops splats.
            assert!(
                info.n_vertices as u64 >= last_count,
                "cut {cut}: count {} regressed from {last_count}",
                info.n_vertices,
            );
            last_count = info.n_vertices as u64;
            // Monotone PSNR proxy: the *minimum* score in the kept set
            // is non-increasing as we admit more splats. Equivalently:
            // every kept score >= last_min_score's predecessor.
            let scores = importance_scores_from_ply(&ply_out).unwrap();
            if !scores.is_empty() {
                let new_min = *scores
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                assert!(
                    new_min <= last_min_score + 1e-6,
                    "min-score didn't decrease as more splats admitted: {new_min} > {last_min_score}",
                );
                last_min_score = new_min;
            }
        }
        assert_eq!(last_count, n as u64);
    }

    #[test]
    fn cut_smaller_than_prefix_yields_empty_ply() {
        let ply = synthetic_ply(4);
        let mgs2 = encode_progressive(&ply).unwrap();
        // 16 bytes is inside the fixed prefix.
        let out = decode_progressive(&mgs2, Some(16)).unwrap();
        let info = parse_inria_ply_header(&out).unwrap();
        assert_eq!(info.n_vertices, 0);
    }

    #[test]
    fn cut_partway_through_record_drops_trailing_partial() {
        let ply = synthetic_ply(10);
        let mgs2 = encode_progressive(&ply).unwrap();
        let h = read_mgs2_header(&mgs2).unwrap();
        // Cut at "prefix + ply_header + 2.5 * record_size".
        let cut = h.payload_offset + (h.record_size as u64) * 5 / 2;
        let out = decode_progressive(&mgs2, Some(cut)).unwrap();
        let info = parse_inria_ply_header(&out).unwrap();
        assert_eq!(info.n_vertices, 2, "must drop the partial third record");
    }

    #[test]
    fn rewrite_vertex_count_preserves_other_lines() {
        let header = b"ply\n\
                       format binary_little_endian 1.0\n\
                       comment hello\n\
                       element vertex 123\n\
                       property float x\n\
                       end_header\n";
        let rewritten = rewrite_vertex_count(header, 7).unwrap();
        let s = std::str::from_utf8(&rewritten).unwrap();
        assert!(s.contains("element vertex 7\n"));
        assert!(s.contains("comment hello\n"));
        assert!(s.contains("property float x\n"));
        assert!(!s.contains("element vertex 123"));
    }
}
