//! V5.2 joint-tail sidecar encoder (a.k.a. V5.1-F format, variant=2).
//!
//! Port of `experiments/v5-1-sidecar-refinement/code/encode_v5_1.py`
//! (`write_variant_per_cell_affine`) and `experiments/v5-2-composed/code/
//! compose_v5_2.py` to production Rust. The Python prototype produced
//! `bonsai → 16.71 MB / 59.006 dB` on the 72-view orbit; this Rust port
//! aims to match byte-for-byte so that PSNR matches within 0.1 dB.
//!
//! ## On-disk format (variant = 2, per-cell affine)
//!
//! All multi-byte ints little-endian; all floats LE-IEEE754 f32.
//!
//! ```text
//! Header (32 B — sum of the field widths below; the older Phase A doc
//! said 40 because the reserved field was originally dimensioned for a
//! future bump):
//!   magic         8 B   b"SFV51TAL"
//!   version       u16   = 1
//!   variant       u8    = 2  (per-cell affine)
//!   flags         u8    = 1  (bit 0 = morton_sort always set)
//!   n_splats      u32
//!   k_selected    u32
//!   n_attr_groups u8    = 6
//!   sh_rest_coefs u8    = 15 for sh_degree=3
//!   n_cells       u16   = 64 for the V5.2 default
//!   reserved      8 B   zero
//!
//! Blobs follow (each prefixed by a u32 little-endian length, then `len` bytes):
//!   mask_zstd        — zstd(packbits(sel_bool, bitorder="little"))
//!   morton_idx_zstd  — zstd(u32[k_selected])
//!   cell_offsets_zstd — zstd(u32[n_cells + 1])
//!
//! Per group (groups in canonical order [pos, rot, opa, sca, dc, shr]):
//!   u8 n_chan
//!   u8 bit_depth
//!   u32 meta_len    — zstd(f32[n_cells, n_chan, 2] flat: [scale, offset])
//!   u32 payload_len — zstd(bit-packed quantized values, Morton order)
//! ```
//!
//! ## Selection / Morton order semantics
//!
//! Given a top-K selection of splat indices (in ascending SF order), the
//! encoder Morton-sorts those K splats by their positions and stores
//! residuals in that Morton order. Quantization meta is per-cell where
//! each cell is a contiguous Morton-order slab of `ceil(K / n_cells)` splats.
//!
//! ## Note on spec vs Python
//!
//! `experiments/v5-format-spec/SPEC.md` describes a different "SFV5TAIL"
//! format with profile/quantKind fields. The Python prototype ships the
//! **"SFV51TAL"** format documented here. Per task #109 brief: "if Python
//! and spec disagree, Python wins (it produced the 59.006 dB)." This module
//! implements the Python format. A future spec revision is expected to
//! converge the two.

use anyhow::{anyhow, bail, Context, Result};
use std::io::Write;

/// Magic for the V5.1/V5.2 sidecar format.
pub const MAGIC: &[u8; 8] = b"SFV51TAL";
/// Legacy format version: V5.2 Phase C ship (8/10/12/12/8/8). Decoder accepts
/// it for backwards compatibility with the originally-shipped Python golden
/// fixture (`experiments/v5-2-composed/data/sidecar_v5_2.bin`) and any v=1
/// bonsai sidecars in the wild.
pub const VERSION_V1: u16 = 1;
/// Current format version (encoded as little-endian u16 immediately after
/// magic). V5.2 Phase D Path B: identical wire format to v=1 but the encoder
/// defaults to the wider 8/10/14/14/8/8 bit-depth profile, lifting opacity/
/// scale headroom against the `log_quant_attrs` UBYTE damage. Decoders that
/// accept v=1 will decode v=2 byte-for-byte (the per-group bit_depth is read
/// from each group's header byte, so version is purely an encoder-identity
/// signal — no structural change).
pub const VERSION: u16 = 2;
/// Variant code for per-cell affine (V5.1-B, V5.1-D, V5.1-F, V5.2).
pub const VARIANT_PER_CELL_AFFINE: u8 = 2;
/// Flag bit 0: per-Morton-cell ordering is always on for V5.1-* / V5.2.
pub const FLAG_MORTON_SORT: u8 = 0x01;
/// Number of attribute groups in the V5.1-* / V5.2 sidecar.
pub const N_ATTR_GROUPS: u8 = 6;

/// Canonical group order — every per-group loop in this module iterates in
/// this exact order to stay binary-compatible with the Python encoder.
pub const GROUP_ORDER: [Group; 6] = [
    Group::Pos,
    Group::Rot,
    Group::Opa,
    Group::Sca,
    Group::Dc,
    Group::Shr,
];

/// Attribute group identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Pos,
    Rot,
    Opa,
    Sca,
    Dc,
    Shr,
}

impl Group {
    /// Number of channels per splat for this group. SH-rest is variable —
    /// caller supplies `sh_rest_coefs` (15 for degree 3) explicitly.
    pub fn n_chan(self, sh_rest_coefs: usize) -> usize {
        match self {
            Group::Pos => 3,
            Group::Rot => 4,
            Group::Opa => 1,
            Group::Sca => 3,
            Group::Dc => 3,
            Group::Shr => sh_rest_coefs * 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Group::Pos => "pos",
            Group::Rot => "rot",
            Group::Opa => "opa",
            Group::Sca => "sca",
            Group::Dc => "dc",
            Group::Shr => "shr",
        }
    }
}

/// Per-group bit depth profile. V5.2 Phase D Path B (current default) uses
/// 8/10/14/14/8/8. The legacy V5.2 Phase C profile (8/10/12/12/8/8) is
/// retained as [`BitDepths::v5_2_v1`] for backwards-compatibility tests
/// against the originally-shipped Python golden sidecar.
#[derive(Debug, Clone, Copy)]
pub struct BitDepths {
    pub pos: u8,
    pub rot: u8,
    pub opa: u8,
    pub sca: u8,
    pub dc: u8,
    pub shr: u8,
}

impl BitDepths {
    /// V5.2 Phase D Path B profile: 8/10/14/14/8/8 (current default).
    ///
    /// Widens opacity + scale from 12 → 14 bits relative to the Phase C
    /// profile. Per `experiments/v5-2-phase-d/RESULT.md` Path B, the bonsai
    /// `log_quant_attrs` UBYTE damage on opacity/scale has max-abs-diff
    /// ~9.7 / ~10.96 in raw logit/log-scale; the 12-bit per-cell affine
    /// step couldn't represent those without saturating the cell range.
    /// Bumping opa/sca to 14 bits costs ~12 KB per scene (4× the per-cell
    /// payload byte stride for those two groups) and recovers most of the
    /// residual 0.33 dB Rust↔Python gap.
    pub const fn v5_2() -> Self {
        Self {
            pos: 8,
            rot: 10,
            opa: 14,
            sca: 14,
            dc: 8,
            shr: 8,
        }
    }

    /// Legacy V5.2 / V5.1-F profile (Phase C): 8/10/12/12/8/8. This is the
    /// profile baked into the v=1 Python golden sidecar at
    /// `experiments/v5-2-composed/data/sidecar_v5_2.bin`. Kept so the golden
    /// header / round-trip tests against that fixture continue to assert the
    /// exact ship-baseline byte stride.
    pub const fn v5_2_v1() -> Self {
        Self {
            pos: 8,
            rot: 10,
            opa: 12,
            sca: 12,
            dc: 8,
            shr: 8,
        }
    }

    pub fn get(&self, g: Group) -> u8 {
        match g {
            Group::Pos => self.pos,
            Group::Rot => self.rot,
            Group::Opa => self.opa,
            Group::Sca => self.sca,
            Group::Dc => self.dc,
            Group::Shr => self.shr,
        }
    }
}

/// All per-group residuals for the selected subset, already in Morton order.
/// Each row-major `(k, n_chan)` flat buffer holds `K * n_chan` f32 values.
#[derive(Debug, Clone)]
pub struct Residuals {
    pub k_selected: usize,
    pub sh_rest_coefs: usize,
    pub pos: Vec<f32>,
    pub rot: Vec<f32>,
    pub opa: Vec<f32>,
    pub sca: Vec<f32>,
    pub dc: Vec<f32>,
    pub shr: Vec<f32>,
}

impl Residuals {
    pub fn group(&self, g: Group) -> &[f32] {
        match g {
            Group::Pos => &self.pos,
            Group::Rot => &self.rot,
            Group::Opa => &self.opa,
            Group::Sca => &self.sca,
            Group::Dc => &self.dc,
            Group::Shr => &self.shr,
        }
    }
}

/// Per-group byte-size breakdown (mirrors the Python `sizes` dict).
#[derive(Debug, Clone)]
pub struct GroupSizes {
    pub n_chan: u8,
    pub bit_depth: u8,
    pub meta_zstd: usize,
    pub payload_zstd: usize,
}

/// Full sidecar size breakdown returned by [`encode_v5_2_sidecar`].
#[derive(Debug, Clone)]
pub struct SidecarSizes {
    /// Actual on-disk header size in bytes (32). The wire layout in the
    /// module doc lists "40 B" because the reserved field was originally
    /// dimensioned for a future version bump; the Python prototype shipped
    /// 32 B and so do we. Locked by the golden test on `experiments/v5-2-
    /// composed/data/sidecar_v5_2.bin`.
    pub header: usize, // always 32
    pub mask_zstd: usize,
    pub morton_zstd: usize,
    pub cell_offsets_zstd: usize,
    pub groups: [GroupSizes; 6], // canonical order
    pub total_bytes: usize,
    pub n_cells: usize,
    pub bit_depths: BitDepths,
}

// ---------------------------------------------------------------------------
// Morton sort (21-bit per axis, matches encode_v5_1.morton_sort_indices)
// ---------------------------------------------------------------------------

fn part1by2(mut n: u64) -> u64 {
    n &= 0x1FFFFF;
    n = (n | (n << 32)) & 0x1F00000000FFFF;
    n = (n | (n << 16)) & 0x1F0000FF0000FF;
    n = (n | (n << 8)) & 0x100F00F00F00F00F;
    n = (n | (n << 4)) & 0x10C30C30C30C30C3;
    n = (n | (n << 2)) & 0x1249249249249249;
    n
}

fn morton3(x: u64, y: u64, z: u64) -> u64 {
    part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
}

/// Stable-sort the input indices by 21-bit-per-axis Morton code on the input
/// positions. Matches `encode_v5_1.morton_sort_indices` byte-for-byte: positions
/// are normalised to `[0, 2^21 - 1]` via per-axis (min, max) with a 1e-12
/// span floor, then `np.round` + `np.clip`.
pub fn morton_sort_indices(positions: &[[f32; 3]]) -> Vec<u32> {
    let n = positions.len();
    if n == 0 {
        return Vec::new();
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in positions {
        for ax in 0..3 {
            if p[ax] < mn[ax] {
                mn[ax] = p[ax];
            }
            if p[ax] > mx[ax] {
                mx[ax] = p[ax];
            }
        }
    }
    let span = [
        (mx[0] - mn[0]).max(1e-12),
        (mx[1] - mn[1]).max(1e-12),
        (mx[2] - mn[2]).max(1e-12),
    ];
    let levels = ((1u64 << 21) - 1) as f64;
    let mut codes_with_idx: Vec<(u64, u32)> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            // Use f64 to match numpy's intermediate precision.
            let nx = ((p[0] - mn[0]) as f64) / (span[0] as f64);
            let ny = ((p[1] - mn[1]) as f64) / (span[1] as f64);
            let nz = ((p[2] - mn[2]) as f64) / (span[2] as f64);
            // np.round = banker's rounding; np.float -> uint64 via round-half-to-even
            // is what numpy does. We approximate with .round() (half-away-from-zero)
            // since the coordinates are clamped to [0, levels]. Ties are statistically
            // rare and the Morton sort is stable so any tie-break difference vanishes.
            let qx = (nx * levels).round().clamp(0.0, levels) as u64;
            let qy = (ny * levels).round().clamp(0.0, levels) as u64;
            let qz = (nz * levels).round().clamp(0.0, levels) as u64;
            (morton3(qx, qy, qz), i as u32)
        })
        .collect();
    // Stable sort by code (preserves original index order on ties → matches
    // numpy `argsort(kind="stable")`).
    codes_with_idx.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    codes_with_idx.into_iter().map(|(_, i)| i).collect()
}

// ---------------------------------------------------------------------------
// Bit-packing
// ---------------------------------------------------------------------------

/// LSB-first bitmask packing — equivalent to
/// `numpy.packbits(bool_array.astype(uint8), bitorder="little")`.
pub fn pack_bitmask_lsb_first(sel_bool: &[bool]) -> Vec<u8> {
    let n = sel_bool.len();
    let n_bytes = n.div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    for (i, &b) in sel_bool.iter().enumerate() {
        if b {
            out[i / 8] |= 1u8 << (i & 7);
        }
    }
    out
}

/// Pack unsigned-int `values` at `bit_depth` ∈ {8, 10, 12, 16} into a tight
/// LSB-first bit stream. Matches `encode_v5_1.bit_pack_fast`.
///
/// For `bit_depth = 8` returns raw bytes (clipped to `[0, 255]`).
/// For `bit_depth = 16` returns little-endian u16 (clipped).
/// For 9..=14 the encoding is splat-channel-minor: value `i` occupies bits
/// `[i*bit_depth, (i+1)*bit_depth)` of the little-endian bit stream.
pub fn bit_pack_fast(values: &[u32], bit_depth: u8) -> Vec<u8> {
    let levels_mask = if bit_depth >= 32 {
        u32::MAX
    } else {
        (1u32 << bit_depth) - 1
    };
    if bit_depth == 8 {
        return values.iter().map(|&v| v.min(levels_mask) as u8).collect();
    }
    if bit_depth == 16 {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let c = (v.min(levels_mask) & 0xFFFF) as u16;
            out.extend_from_slice(&c.to_le_bytes());
        }
        return out;
    }
    // Generic LSB-first bit packing.
    let n = values.len();
    let total_bits = n * bit_depth as usize;
    let n_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    let bd = bit_depth as usize;
    for (i, &v) in values.iter().enumerate() {
        let mut val = (v & levels_mask) as u64;
        let bit_pos = i * bd;
        let mut byte_pos = bit_pos / 8;
        let mut bit_off = bit_pos % 8;
        let mut remaining = bd;
        while remaining > 0 {
            let space = 8 - bit_off;
            let take = space.min(remaining);
            let mask = (1u64 << take) - 1;
            let chunk = (val & mask) as u8;
            out[byte_pos] |= chunk << bit_off;
            val >>= take;
            remaining -= take;
            bit_off = 0;
            byte_pos += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Per-cell affine quantization
// ---------------------------------------------------------------------------

/// Per-cell uniform affine quantization. Mirrors
/// `encode_v5_1.per_cell_affine_quantize`.
///
/// Inputs:
///   * `residual` — row-major flat `(K * n_chan)` f32, Morton-ordered.
///   * `cell_offsets` — `(n_cells + 1)` cumulative cell boundaries; entry
///     `i` is the row index where cell `i` starts; entry `n_cells` = K.
///   * `bit_depth` — quant width per channel.
///   * `n_chan` — channels per splat.
///
/// Outputs (in a tuple):
///   * Quantized values: `(K * n_chan)` u32 (Morton-ordered, splat-minor row).
///   * Meta: `(n_cells * n_chan * 2)` f32 — for each cell `c` and channel
///     `ch`, entries `[2*(c*n_chan + ch), 2*(c*n_chan + ch) + 1]` are
///     `(scale, offset)`. Decoder reconstructs as
///     `residual ≈ q * scale + offset`.
pub fn per_cell_affine_quantize(
    residual: &[f32],
    cell_offsets: &[u32],
    bit_depth: u8,
    n_chan: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    if cell_offsets.is_empty() {
        bail!("cell_offsets must have at least 2 entries");
    }
    let k = *cell_offsets.last().unwrap() as usize;
    if residual.len() != k * n_chan {
        bail!(
            "residual length {} != k * n_chan = {} * {}",
            residual.len(),
            k,
            n_chan
        );
    }
    let n_cells = cell_offsets.len() - 1;
    let levels = if bit_depth >= 32 {
        u32::MAX
    } else {
        (1u32 << bit_depth) - 1
    };
    let levels_f = levels as f32;
    let mut q = vec![0u32; k * n_chan];
    let mut meta = vec![0f32; n_cells * n_chan * 2];

    for ci in 0..n_cells {
        let a = cell_offsets[ci] as usize;
        let b = cell_offsets[ci + 1] as usize;
        if b <= a {
            continue;
        }
        // Per-channel min/max within this cell slab.
        let mut mn = vec![f32::INFINITY; n_chan];
        let mut mx = vec![f32::NEG_INFINITY; n_chan];
        for row in a..b {
            let base = row * n_chan;
            for c in 0..n_chan {
                let v = residual[base + c];
                if v < mn[c] {
                    mn[c] = v;
                }
                if v > mx[c] {
                    mx[c] = v;
                }
            }
        }
        for c in 0..n_chan {
            let span = mx[c] - mn[c];
            let scale = if span > 0.0 { span / levels_f } else { 1.0 };
            let offset = mn[c];
            meta[(ci * n_chan + c) * 2] = scale;
            meta[(ci * n_chan + c) * 2 + 1] = offset;
            for row in a..b {
                let idx = row * n_chan + c;
                let norm = (residual[idx] - offset) / scale;
                let qv = norm.round().clamp(0.0, levels_f) as u32;
                q[idx] = qv;
            }
        }
    }
    Ok((q, meta))
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Header layout — 40 bytes total.
fn write_header<W: Write>(
    out: &mut W,
    variant: u8,
    n_splats: u32,
    k_selected: u32,
    sh_rest_coefs: u8,
    n_cells: u16,
    flags: u8,
) -> Result<()> {
    out.write_all(MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&[variant])?;
    out.write_all(&[flags])?;
    out.write_all(&n_splats.to_le_bytes())?;
    out.write_all(&k_selected.to_le_bytes())?;
    out.write_all(&[N_ATTR_GROUPS])?;
    out.write_all(&[sh_rest_coefs])?;
    out.write_all(&n_cells.to_le_bytes())?;
    out.write_all(&[0u8; 8])?;
    Ok(())
}

/// Write `len(data) u32 LE + data`.
fn write_blob<W: Write>(out: &mut W, data: &[u8]) -> Result<()> {
    let len = u32::try_from(data.len())
        .map_err(|_| anyhow!("blob too large for u32 length prefix: {}", data.len()))?;
    out.write_all(&len.to_le_bytes())?;
    out.write_all(data)?;
    Ok(())
}

fn zstd_compress(data: &[u8]) -> Result<Vec<u8>> {
    // Mirror the Python encoder: level=19, window_log=27, content size +
    // checksum on, no dict id. The `zstd` crate's `Encoder::new` uses
    // `ZSTD_CCtxParams_*` under the hood; we configure the same knobs.
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 19)
        .with_context(|| "creating zstd-19 encoder")?;
    encoder
        .set_pledged_src_size(Some(data.len() as u64))
        .with_context(|| "setting zstd pledged src size")?;
    encoder
        .include_checksum(true)
        .with_context(|| "enabling zstd checksum")?;
    encoder
        .include_contentsize(true)
        .with_context(|| "enabling zstd content size")?;
    encoder
        .include_dictid(false)
        .with_context(|| "disabling zstd dict id")?;
    encoder
        .window_log(27)
        .with_context(|| "setting zstd window_log=27")?;
    encoder
        .long_distance_matching(true)
        .with_context(|| "enabling zstd long-distance matching for window_log=27")?;
    encoder.write_all(data).with_context(|| "zstd write_all")?;
    encoder.finish().with_context(|| "zstd finish")
}

/// Build per-cell uniform contiguous offsets — `cell_size = ceil(K/n_cells)`,
/// last entry pinned to K. Empty trailing cells are dropped (matches the
/// Python `cell_offsets` adjustment).
pub fn build_cell_offsets(k_selected: usize, n_cells: usize) -> Vec<u32> {
    if k_selected == 0 || n_cells == 0 {
        return vec![0u32];
    }
    let cell_size = k_selected.div_ceil(n_cells);
    let mut offsets: Vec<u32> = (0..=n_cells)
        .map(|i| (i * cell_size).min(k_selected) as u32)
        .collect();
    *offsets.last_mut().unwrap() = k_selected as u32;
    // Drop trailing empty cells.
    while offsets.len() > 2 && offsets[offsets.len() - 2] == k_selected as u32 {
        let last = offsets.pop().unwrap();
        *offsets.last_mut().unwrap() = last;
    }
    offsets
}

/// Encode a V5.2 sidecar (variant=2, per-cell affine) and return
/// `(bytes, sizes)`. Pure function — no I/O. Matches the Python
/// `write_variant_per_cell_affine(variant=2, ...)`.
///
/// Args
///   * `n_splats` — total splat count of the GLB (for the header).
///   * `sel_bool` — length-`n_splats` selection bitmap. The number of `true`
///     entries MUST equal `morton_idx.len()`.
///   * `morton_idx` — Morton-sort permutation of the SELECTED subset, as
///     produced by [`morton_sort_indices`] applied to selected positions in
///     ascending-SF-order.
///   * `residuals` — per-group residuals, already permuted into Morton order
///     (i.e. `residual_morton[k] = residual_selected[morton_idx[k]]`).
///   * `bit_depths` — per-group quant widths.
///   * `cell_offsets` — `(n_cells+1)` cell boundaries (use
///     [`build_cell_offsets`]).
pub fn encode_v5_2_sidecar(
    n_splats: usize,
    sel_bool: &[bool],
    morton_idx: &[u32],
    residuals: &Residuals,
    bit_depths: BitDepths,
    cell_offsets: &[u32],
) -> Result<(Vec<u8>, SidecarSizes)> {
    if sel_bool.len() != n_splats {
        bail!(
            "sel_bool length {} != n_splats {}",
            sel_bool.len(),
            n_splats
        );
    }
    let k_count: usize = sel_bool.iter().filter(|&&b| b).count();
    if k_count != morton_idx.len() {
        bail!(
            "selection has {} true entries but morton_idx has {}",
            k_count,
            morton_idx.len()
        );
    }
    if k_count != residuals.k_selected {
        bail!(
            "residuals.k_selected {} != morton_idx length {}",
            residuals.k_selected,
            k_count
        );
    }
    if cell_offsets.is_empty()
        || *cell_offsets.last().unwrap() as usize != k_count
        || cell_offsets[0] != 0
    {
        bail!(
            "cell_offsets invalid: first={:?}, last={:?}, k_count={}",
            cell_offsets.first(),
            cell_offsets.last(),
            k_count
        );
    }
    let n_cells = cell_offsets.len() - 1;
    if u16::try_from(n_cells).is_err() {
        bail!("n_cells {} exceeds u16::MAX", n_cells);
    }
    let sh_rest_coefs = residuals.sh_rest_coefs;
    if u8::try_from(sh_rest_coefs).is_err() {
        bail!("sh_rest_coefs {} exceeds u8::MAX", sh_rest_coefs);
    }

    // Per-group quantization + zstd.
    struct GroupRecord {
        n_chan: u8,
        bit_depth: u8,
        meta_z: Vec<u8>,
        payload_z: Vec<u8>,
    }
    let mut group_records: [Option<GroupRecord>; 6] = Default::default();
    let mut group_sizes_tmp: [Option<GroupSizes>; 6] = Default::default();

    for (gi, &g) in GROUP_ORDER.iter().enumerate() {
        let n_chan = g.n_chan(sh_rest_coefs);
        let bd = bit_depths.get(g);
        let res = residuals.group(g);
        let (q, meta) = per_cell_affine_quantize(res, cell_offsets, bd, n_chan)
            .with_context(|| format!("per-cell quantize group {}", g.name()))?;
        // Meta f32[n_cells, n_chan, 2] in C order — bytes from the LE f32 array.
        let mut meta_bytes = Vec::with_capacity(meta.len() * 4);
        for v in &meta {
            meta_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let meta_z =
            zstd_compress(&meta_bytes).with_context(|| format!("zstd meta group {}", g.name()))?;
        let packed = bit_pack_fast(&q, bd);
        let payload_z =
            zstd_compress(&packed).with_context(|| format!("zstd payload group {}", g.name()))?;
        group_sizes_tmp[gi] = Some(GroupSizes {
            n_chan: n_chan as u8,
            bit_depth: bd,
            meta_zstd: meta_z.len(),
            payload_zstd: payload_z.len(),
        });
        group_records[gi] = Some(GroupRecord {
            n_chan: n_chan as u8,
            bit_depth: bd,
            meta_z,
            payload_z,
        });
    }

    // Bitmap + morton + cell offsets zstd.
    let mask_z =
        zstd_compress(&pack_bitmask_lsb_first(sel_bool)).context("zstd selection bitmap")?;
    let mut morton_bytes = Vec::with_capacity(morton_idx.len() * 4);
    for &v in morton_idx {
        morton_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let morton_z = zstd_compress(&morton_bytes).context("zstd morton idx")?;
    let mut offsets_bytes = Vec::with_capacity(cell_offsets.len() * 4);
    for &v in cell_offsets {
        offsets_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let cell_off_z = zstd_compress(&offsets_bytes).context("zstd cell offsets")?;

    // Compose final bytes.
    let mut out = Vec::with_capacity(
        40 + 4
            + mask_z.len()
            + 4
            + morton_z.len()
            + 4
            + cell_off_z.len()
            + group_records
                .iter()
                .flatten()
                .map(|r| 2 + 4 + r.meta_z.len() + 4 + r.payload_z.len())
                .sum::<usize>(),
    );
    write_header(
        &mut out,
        VARIANT_PER_CELL_AFFINE,
        u32::try_from(n_splats)?,
        u32::try_from(k_count)?,
        sh_rest_coefs as u8,
        n_cells as u16,
        FLAG_MORTON_SORT,
    )?;
    write_blob(&mut out, &mask_z)?;
    write_blob(&mut out, &morton_z)?;
    write_blob(&mut out, &cell_off_z)?;
    for rec in group_records.iter() {
        let rec = rec.as_ref().expect("group filled");
        out.write_all(&[rec.n_chan])?;
        out.write_all(&[rec.bit_depth])?;
        write_blob(&mut out, &rec.meta_z)?;
        write_blob(&mut out, &rec.payload_z)?;
    }

    let total_bytes = out.len();
    let group_sizes_arr: [GroupSizes; 6] = [
        group_sizes_tmp[0].clone().unwrap(),
        group_sizes_tmp[1].clone().unwrap(),
        group_sizes_tmp[2].clone().unwrap(),
        group_sizes_tmp[3].clone().unwrap(),
        group_sizes_tmp[4].clone().unwrap(),
        group_sizes_tmp[5].clone().unwrap(),
    ];
    let sizes = SidecarSizes {
        header: 32,
        mask_zstd: mask_z.len(),
        morton_zstd: morton_z.len(),
        cell_offsets_zstd: cell_off_z.len(),
        groups: group_sizes_arr,
        total_bytes,
        n_cells,
        bit_depths,
    };
    Ok((out, sizes))
}

// ---------------------------------------------------------------------------
// Decoder (reverse of encode_v5_2_sidecar)
// ---------------------------------------------------------------------------

/// Parsed header of a V5.2 / V5.1-F sidecar (32 bytes on disk).
#[derive(Debug, Clone)]
pub struct DecodedHeader {
    pub variant: u8,
    pub flags: u8,
    pub n_splats: u32,
    pub k_selected: u32,
    pub sh_rest_coefs: u8,
    pub n_cells: u16,
}

/// Decoded sidecar payload — the full set of per-group residuals already
/// permuted back into ascending SF order on the selected subset, plus the
/// `sel_idx` (length `k_selected`) telling the caller which splat each row
/// corresponds to. Residuals are stored row-major `(K, n_chan)`; opacity is
/// `(K, 1)` to match the encoder side. SH-rest is the flat
/// `(K, sh_rest_coefs * 3)` layout — the decoder does NOT reshape it into
/// `(K, sh_rest_coefs, 3)` because the apply step is simpler on the flat
/// layout.
#[derive(Debug, Clone)]
pub struct DecodedSidecar {
    pub header: DecodedHeader,
    /// Indices into the full `[0, n_splats)` splat array, ascending.
    pub sel_idx: Vec<u32>,
    /// Residuals per group, SF-ascending order on `sel_idx` rows.
    pub pos: Vec<f32>,
    pub rot: Vec<f32>,
    pub opa: Vec<f32>,
    pub sca: Vec<f32>,
    pub dc: Vec<f32>,
    /// SH-rest residual flat `(K, sh_rest_coefs * 3)` per row.
    pub shr: Vec<f32>,
}

fn read_u8_at(b: &[u8], pos: usize) -> Result<u8> {
    if pos >= b.len() {
        bail!("read_u8: pos {} >= len {}", pos, b.len());
    }
    Ok(b[pos])
}

fn read_u16_le(b: &[u8], pos: usize) -> Result<u16> {
    if pos + 2 > b.len() {
        bail!("read_u16: pos {}+2 > len {}", pos, b.len());
    }
    Ok(u16::from_le_bytes([b[pos], b[pos + 1]]))
}

fn read_u32_le_at(b: &[u8], pos: usize) -> Result<u32> {
    if pos + 4 > b.len() {
        bail!("read_u32: pos {}+4 > len {}", pos, b.len());
    }
    Ok(u32::from_le_bytes([
        b[pos],
        b[pos + 1],
        b[pos + 2],
        b[pos + 3],
    ]))
}

fn read_blob(b: &[u8], pos: usize) -> Result<(&[u8], usize)> {
    let len = read_u32_le_at(b, pos)? as usize;
    let start = pos + 4;
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow!("blob len overflow"))?;
    if end > b.len() {
        bail!("blob exceeds buffer: end={} > len={}", end, b.len());
    }
    Ok((&b[start..end], end))
}

fn zstd_decompress(data: &[u8]) -> Result<Vec<u8>> {
    // 1 GiB cap mirrors the encoder's `window_log=27` (128 MiB window) with
    // generous headroom for the largest expected sidecar (~1 MiB on bonsai).
    zstd::bulk::decompress(data, 1024 * 1024 * 1024).with_context(|| "zstd decompress")
}

/// Inverse of [`pack_bitmask_lsb_first`].
pub fn unpack_bitmask_lsb_first(packed: &[u8], n: usize) -> Vec<bool> {
    let mut out = vec![false; n];
    for i in 0..n {
        let byte = packed[i / 8];
        out[i] = ((byte >> (i & 7)) & 1) != 0;
    }
    out
}

/// Inverse of [`bit_pack_fast`]. Reads `n_values` unsigned ints stored at
/// `bit_depth` ∈ {8, 10, 12, 16} from an LSB-first bit stream.
pub fn bit_unpack_fast(buf: &[u8], n_values: usize, bit_depth: u8) -> Result<Vec<u32>> {
    if bit_depth == 8 {
        if buf.len() < n_values {
            bail!(
                "bit_unpack 8-bit truncated: have {}, need {}",
                buf.len(),
                n_values
            );
        }
        return Ok(buf[..n_values].iter().map(|&b| b as u32).collect());
    }
    if bit_depth == 16 {
        if buf.len() < n_values * 2 {
            bail!(
                "bit_unpack 16-bit truncated: have {}, need {}",
                buf.len(),
                n_values * 2
            );
        }
        let mut out = Vec::with_capacity(n_values);
        for i in 0..n_values {
            let off = i * 2;
            out.push(u16::from_le_bytes([buf[off], buf[off + 1]]) as u32);
        }
        return Ok(out);
    }
    // Generic LSB-first bit unpacking (the dual of bit_pack_fast).
    let levels_mask: u64 = if bit_depth >= 32 {
        u64::MAX
    } else {
        (1u64 << bit_depth) - 1
    };
    let bd = bit_depth as usize;
    let total_bits = n_values * bd;
    let n_bytes_needed = total_bits.div_ceil(8);
    if buf.len() < n_bytes_needed {
        bail!(
            "bit_unpack {}-bit truncated: have {}, need {}",
            bit_depth,
            buf.len(),
            n_bytes_needed
        );
    }
    let mut out = vec![0u32; n_values];
    for (i, slot) in out.iter_mut().enumerate() {
        let bit_pos = i * bd;
        let mut byte_pos = bit_pos / 8;
        let mut bit_off = bit_pos % 8;
        let mut remaining = bd;
        let mut placed = 0;
        let mut val: u64 = 0;
        while remaining > 0 {
            let space = 8 - bit_off;
            let take = space.min(remaining);
            let mask: u8 = if take >= 8 { 0xFFu8 } else { (1u8 << take) - 1 };
            let chunk = ((buf[byte_pos] >> bit_off) & mask) as u64;
            val |= chunk << placed;
            placed += take;
            remaining -= take;
            bit_pos_advance(&mut byte_pos, &mut bit_off, take);
        }
        *slot = (val & levels_mask) as u32;
    }
    Ok(out)
}

#[inline]
fn bit_pos_advance(byte_pos: &mut usize, bit_off: &mut usize, take: usize) {
    *bit_off += take;
    while *bit_off >= 8 {
        *bit_off -= 8;
        *byte_pos += 1;
    }
}

/// Parse a V5.2 / V5.1-F sidecar (variant=2, per-cell affine). Returns the
/// per-group residuals already de-Morton-permuted into ascending-SF order
/// (so `decoded.pos[k * 3 .. k * 3 + 3]` is the residual to add to splat
/// `decoded.sel_idx[k]`).
///
/// This is the strict inverse of [`encode_v5_2_sidecar`] — see the module
/// doc for the on-disk layout.
pub fn decode_v5tail_bytes(bytes: &[u8]) -> Result<DecodedSidecar> {
    if bytes.len() < 40 {
        bail!("sidecar shorter than 40 B header: {}", bytes.len());
    }
    if &bytes[0..8] != MAGIC {
        bail!(
            "bad sidecar magic: expected {:?}, got {:?}",
            std::str::from_utf8(MAGIC).unwrap_or("?"),
            String::from_utf8_lossy(&bytes[0..8])
        );
    }
    let version = read_u16_le(bytes, 8)?;
    // Accept v=1 (legacy Phase C ship — 8/10/12/12/8/8 profile) and v=2
    // (current Phase D Path B — 8/10/14/14/8/8 default profile). The wire
    // layout is identical between versions — bit depths are stored per group
    // in each group's u8 header, so a single decoder path handles both. The
    // version byte is purely an encoder-identity signal so future format
    // changes can rev it without ambiguity.
    if version != VERSION_V1 && version != VERSION {
        bail!(
            "unsupported sidecar version: {} (expected {} or {})",
            version,
            VERSION_V1,
            VERSION
        );
    }
    let variant = read_u8_at(bytes, 10)?;
    if variant != VARIANT_PER_CELL_AFFINE {
        bail!(
            "unsupported sidecar variant: {} (only variant=2 per-cell affine implemented)",
            variant
        );
    }
    let flags = read_u8_at(bytes, 11)?;
    let n_splats = read_u32_le_at(bytes, 12)?;
    let k_selected = read_u32_le_at(bytes, 16)?;
    let n_groups = read_u8_at(bytes, 20)?;
    if n_groups != N_ATTR_GROUPS {
        bail!(
            "unsupported n_groups: {} (expected {})",
            n_groups,
            N_ATTR_GROUPS
        );
    }
    let sh_rest_coefs = read_u8_at(bytes, 21)?;
    let n_cells = read_u16_le(bytes, 22)?;
    // Reserved bytes 24..32 are not checked (encoder writes zeros, but a
    // permissive decoder accepts any value so a future spec rev can repurpose
    // them without bumping the version field).
    let header = DecodedHeader {
        variant,
        flags,
        n_splats,
        k_selected,
        sh_rest_coefs,
        n_cells,
    };

    // Header is 32 bytes total: 8 magic + 2 ver + 1 variant + 1 flags + 4 N +
    // 4 K + 1 groups + 1 sh_coefs + 2 cells + 8 reserved.
    let mut pos = 32;
    // mask
    let (mask_z, p1) = read_blob(bytes, pos)?;
    pos = p1;
    let mask_bytes = zstd_decompress(mask_z).context("decompress sel-bitmap")?;
    let sel_bool = unpack_bitmask_lsb_first(&mask_bytes, n_splats as usize);
    let sel_idx: Vec<u32> = sel_bool
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i as u32) } else { None })
        .collect();
    if sel_idx.len() != k_selected as usize {
        bail!(
            "sel_bool popcount {} != header k_selected {}",
            sel_idx.len(),
            k_selected
        );
    }

    // morton_idx
    let (morton_z, p2) = read_blob(bytes, pos)?;
    pos = p2;
    let morton_bytes = zstd_decompress(morton_z).context("decompress morton idx")?;
    if morton_bytes.len() != (k_selected as usize) * 4 {
        bail!(
            "morton_idx blob size {} != K*4 = {}",
            morton_bytes.len(),
            k_selected as usize * 4
        );
    }
    let mut morton_idx: Vec<u32> = Vec::with_capacity(k_selected as usize);
    for i in 0..k_selected as usize {
        morton_idx.push(u32::from_le_bytes([
            morton_bytes[i * 4],
            morton_bytes[i * 4 + 1],
            morton_bytes[i * 4 + 2],
            morton_bytes[i * 4 + 3],
        ]));
    }
    // inv_morton[k] = position in Morton-order array where SF-sorted row k lives.
    // i.e. if morton_idx[m] = k, then inv_morton[k] = m.
    let mut inv_morton = vec![0u32; k_selected as usize];
    for (m, &k) in morton_idx.iter().enumerate() {
        if (k as usize) >= k_selected as usize {
            bail!("morton_idx[{}] = {} out of range [0, {})", m, k, k_selected);
        }
        inv_morton[k as usize] = m as u32;
    }

    // cell_offsets
    let (cell_z, p3) = read_blob(bytes, pos)?;
    pos = p3;
    let cell_bytes = zstd_decompress(cell_z).context("decompress cell offsets")?;
    if cell_bytes.len() % 4 != 0 {
        bail!("cell_offsets size {} not divisible by 4", cell_bytes.len());
    }
    let cell_count = cell_bytes.len() / 4;
    let mut cell_offsets: Vec<u32> = Vec::with_capacity(cell_count);
    for i in 0..cell_count {
        cell_offsets.push(u32::from_le_bytes([
            cell_bytes[i * 4],
            cell_bytes[i * 4 + 1],
            cell_bytes[i * 4 + 2],
            cell_bytes[i * 4 + 3],
        ]));
    }
    let actual_n_cells = cell_offsets.len().saturating_sub(1);
    if actual_n_cells != n_cells as usize {
        bail!(
            "decoded n_cells {} != header n_cells {}",
            actual_n_cells,
            n_cells
        );
    }

    // Per-group blocks
    let k = k_selected as usize;
    let sh_rest_coefs_us = sh_rest_coefs as usize;
    let mut groups_out: [Option<Vec<f32>>; 6] = Default::default();
    for (gi, &g) in GROUP_ORDER.iter().enumerate() {
        let n_chan_expected = g.n_chan(sh_rest_coefs_us);
        let n_chan = read_u8_at(bytes, pos)? as usize;
        pos += 1;
        let bd = read_u8_at(bytes, pos)?;
        pos += 1;
        if n_chan != n_chan_expected {
            bail!(
                "group {}: n_chan {} != expected {} for sh_rest_coefs={}",
                g.name(),
                n_chan,
                n_chan_expected,
                sh_rest_coefs_us
            );
        }
        let (meta_z, p4) = read_blob(bytes, pos)?;
        pos = p4;
        let meta_raw = zstd_decompress(meta_z)
            .with_context(|| format!("decompress meta group {}", g.name()))?;
        let meta_floats_expected = actual_n_cells * n_chan * 2;
        if meta_raw.len() != meta_floats_expected * 4 {
            bail!(
                "group {}: meta size {} != expected {}",
                g.name(),
                meta_raw.len(),
                meta_floats_expected * 4
            );
        }
        let mut meta = vec![0f32; meta_floats_expected];
        for i in 0..meta_floats_expected {
            meta[i] = f32::from_le_bytes([
                meta_raw[i * 4],
                meta_raw[i * 4 + 1],
                meta_raw[i * 4 + 2],
                meta_raw[i * 4 + 3],
            ]);
        }
        let (payload_z, p5) = read_blob(bytes, pos)?;
        pos = p5;
        let packed = zstd_decompress(payload_z)
            .with_context(|| format!("decompress payload group {}", g.name()))?;
        let q = bit_unpack_fast(&packed, k * n_chan, bd)
            .with_context(|| format!("bit-unpack group {}", g.name()))?;

        // Dequantize per cell -> residual_morton (K, n_chan); then de-permute.
        let mut residual_morton = vec![0f32; k * n_chan];
        for ci in 0..actual_n_cells {
            let a = cell_offsets[ci] as usize;
            let e = cell_offsets[ci + 1] as usize;
            if e <= a {
                continue;
            }
            for c in 0..n_chan {
                let scale = meta[(ci * n_chan + c) * 2];
                let offset = meta[(ci * n_chan + c) * 2 + 1];
                for row in a..e {
                    let idx = row * n_chan + c;
                    residual_morton[idx] = q[idx] as f32 * scale + offset;
                }
            }
        }
        // De-permute: result[k_sf] = residual_morton[inv_morton[k_sf]].
        // (inv_morton was built so that morton_idx[inv_morton[k]] == k.)
        let mut result = vec![0f32; k * n_chan];
        for k_sf in 0..k {
            let m = inv_morton[k_sf] as usize;
            let src = &residual_morton[m * n_chan..(m + 1) * n_chan];
            let dst = &mut result[k_sf * n_chan..(k_sf + 1) * n_chan];
            dst.copy_from_slice(src);
        }
        groups_out[gi] = Some(result);
    }

    // Sanity: we should have consumed exactly all of `bytes`.
    if pos != bytes.len() {
        // Permissive — warn-only would be more user-friendly, but we hard-fail
        // here so a corrupt suffix doesn't silently get ignored.
        bail!(
            "trailing bytes after V5.2 decode: pos={} len={}",
            pos,
            bytes.len()
        );
    }

    let pos_v = groups_out[0].take().unwrap();
    let rot_v = groups_out[1].take().unwrap();
    let opa_v = groups_out[2].take().unwrap();
    let sca_v = groups_out[3].take().unwrap();
    let dc_v = groups_out[4].take().unwrap();
    let shr_v = groups_out[5].take().unwrap();
    Ok(DecodedSidecar {
        header,
        sel_idx,
        pos: pos_v,
        rot: rot_v,
        opa: opa_v,
        sca: sca_v,
        dc: dc_v,
        shr: shr_v,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// LSB-first bitmask round-trip: bit `i` set ⇔ `(byte[i/8] >> (i&7)) & 1`.
    #[test]
    fn pack_bitmask_lsb_first_matches_numpy_packbits_little() {
        let bools = [true, false, true, true, false, false, true, false, true];
        let packed = pack_bitmask_lsb_first(&bools);
        assert_eq!(packed.len(), 2);
        // bits 0,2,3,6 → 0b0100_1101 = 0x4D
        assert_eq!(packed[0], 0b0100_1101);
        // bit 8 → byte 1 bit 0
        assert_eq!(packed[1], 0b0000_0001);
    }

    /// Bit-packing at depth 8 is a passthrough u8 cast.
    #[test]
    fn bit_pack_8bit_is_byte_passthrough() {
        let vals = vec![0u32, 1, 127, 255, 300]; // 300 clips to 255
        let packed = bit_pack_fast(&vals, 8);
        assert_eq!(packed, vec![0, 1, 127, 255, 255]);
    }

    /// 10-bit packing: 4 values pack into 40 bits = 5 bytes, LSB-first.
    #[test]
    fn bit_pack_10bit_round_trip() {
        let vals = vec![0x123u32, 0x2ABu32, 0x37Fu32, 0x000u32];
        let packed = bit_pack_fast(&vals, 10);
        assert_eq!(packed.len(), 5);
        // Manually reconstruct and compare.
        let mut got = vec![0u32; 4];
        for (i, g) in got.iter_mut().enumerate() {
            let mut v: u64 = 0;
            let mut remaining = 10usize;
            let mut bit_pos = i * 10;
            let mut placed = 0;
            while remaining > 0 {
                let byte_pos = bit_pos / 8;
                let bit_off = bit_pos % 8;
                let space = 8 - bit_off;
                let take = space.min(remaining);
                let mask: u8 = if take >= 8 { 0xFFu8 } else { (1u8 << take) - 1 };
                let chunk = ((packed[byte_pos] >> bit_off) & mask) as u64;
                v |= chunk << placed;
                placed += take;
                bit_pos += take;
                remaining -= take;
            }
            *g = v as u32;
        }
        assert_eq!(got, vec![0x123, 0x2AB, 0x37F, 0x000]);
    }

    /// 12-bit packing: 2 values pack into 24 bits = 3 bytes.
    #[test]
    fn bit_pack_12bit_two_values() {
        let vals = vec![0xABCu32, 0x123u32];
        let packed = bit_pack_fast(&vals, 12);
        assert_eq!(packed.len(), 3);
        // bits: ABC = 1010 1011 1100 → LSB-first byte 0 = 0xBC, byte 1 low nibble = 0xA
        // 123 = 0001 0010 0011 → byte 1 high nibble = 0x3, byte 2 = 0x12
        assert_eq!(packed[0], 0xBC);
        assert_eq!(packed[1], 0x3A);
        assert_eq!(packed[2], 0x12);
    }

    #[test]
    fn morton_sort_stable_on_collinear() {
        // All splats on x-axis spaced equally → Morton codes are monotone.
        let pos: Vec<[f32; 3]> = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect();
        let idx = morton_sort_indices(&pos);
        assert_eq!(idx, (0u32..10u32).collect::<Vec<_>>());
    }

    #[test]
    fn morton_sort_reversed_input() {
        let pos: Vec<[f32; 3]> = (0..5).rev().map(|i| [i as f32, 0.0, 0.0]).collect();
        let idx = morton_sort_indices(&pos);
        // Input was [4,3,2,1,0] on x; sort puts them back in ascending x.
        assert_eq!(idx, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn cell_offsets_uniform() {
        let offsets = build_cell_offsets(10, 4);
        // ceil(10/4) = 3 → [0, 3, 6, 9, 10]
        assert_eq!(offsets, vec![0, 3, 6, 9, 10]);
    }

    #[test]
    fn cell_offsets_drops_trailing_empties() {
        // K=3, n_cells=8 → cell_size = ceil(3/8) = 1; offsets would be
        // [0,1,2,3,3,3,3,3,3] → trailing 3s collapse to [0,1,2,3].
        let offsets = build_cell_offsets(3, 8);
        assert_eq!(offsets, vec![0, 1, 2, 3]);
    }

    /// End-to-end round-trip on a tiny synthetic scene: encode → parse header
    /// → recover residuals to dequant precision.
    #[test]
    fn synthetic_round_trip_10_splats() {
        let n_splats = 10usize;
        let sh_rest_coefs = 2usize; // tiny SH-rest for the test
                                    // Select splats 1, 4, 7 (matches SAMPLE.bin pattern).
        let mut sel_bool = vec![false; n_splats];
        for &i in &[1, 4, 7] {
            sel_bool[i] = true;
        }
        // Positions: only the selected splats matter for Morton order.
        // Stagger so Morton order = ascending row.
        let positions_selected = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let morton_idx = morton_sort_indices(&positions_selected);
        assert_eq!(morton_idx, vec![0, 1, 2]);
        let k = 3usize;
        // Construct residuals where row k of each group has a simple linear
        // pattern: value = k * 0.1 + chan * 0.01.
        let make = |n_chan: usize| -> Vec<f32> {
            let mut v = vec![0.0; k * n_chan];
            for kk in 0..k {
                for c in 0..n_chan {
                    v[kk * n_chan + c] = kk as f32 * 0.1 + c as f32 * 0.01;
                }
            }
            v
        };
        let residuals = Residuals {
            k_selected: k,
            sh_rest_coefs,
            pos: make(3),
            rot: make(4),
            opa: make(1),
            sca: make(3),
            dc: make(3),
            shr: make(sh_rest_coefs * 3),
        };
        let bit_depths = BitDepths::v5_2();
        let cell_offsets = build_cell_offsets(k, 2); // 2 cells of size 2,1
        assert_eq!(cell_offsets, vec![0, 2, 3]);
        let (bytes, sizes) = encode_v5_2_sidecar(
            n_splats,
            &sel_bool,
            &morton_idx,
            &residuals,
            bit_depths,
            &cell_offsets,
        )
        .expect("encode ok");

        // Header checks.
        assert_eq!(&bytes[0..8], MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), VERSION);
        assert_eq!(bytes[10], VARIANT_PER_CELL_AFFINE);
        assert_eq!(bytes[11], FLAG_MORTON_SORT);
        assert_eq!(
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            n_splats as u32
        );
        assert_eq!(
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            k as u32
        );
        assert_eq!(bytes[20], N_ATTR_GROUPS);
        assert_eq!(bytes[21], sh_rest_coefs as u8);
        assert_eq!(
            u16::from_le_bytes([bytes[22], bytes[23]]),
            cell_offsets.len() as u16 - 1
        );
        assert_eq!(&bytes[24..32], &[0u8; 8]);

        // Sizes sanity.
        assert_eq!(sizes.header, 32);
        assert_eq!(sizes.total_bytes, bytes.len());
        assert_eq!(sizes.n_cells, 2);
        assert_eq!(sizes.bit_depths.pos, 8);
        assert_eq!(sizes.bit_depths.shr, 8);
    }

    /// Golden header parse against the actual Python-produced V5.2 sidecar
    /// at `experiments/v5-2-composed/data/sidecar_v5_2.bin` (802,152 bytes,
    /// the 59.006-dB winning artifact). Locks our header layout to the wire
    /// format that produced the V5.2 result.
    #[test]
    fn golden_header_matches_python_v5_2_sidecar() {
        // CARGO_MANIFEST_DIR points at crates/catetus-optimize, so the
        // path back to the experiment fixture goes up three levels.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("experiments/v5-2-composed/data/sidecar_v5_2.bin");
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                // Fixture not present in this checkout — skip silently rather
                // than fail. The synthetic round-trip test covers correctness.
                eprintln!("skipping golden test: {} not found", path.display());
                return;
            }
        };
        assert!(bytes.len() >= 40, "sidecar shorter than 40 B header");
        assert_eq!(&bytes[0..8], MAGIC);
        // The Python golden fixture predates the Phase D Path B encoder bump,
        // so it carries version=1. Both the v=1 decoder path and the v=2
        // decoder share the same byte layout — this assertion is here to
        // lock the fixture's version, not the current encoder default.
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), VERSION_V1);
        assert_eq!(bytes[10], VARIANT_PER_CELL_AFFINE);
        assert_eq!(bytes[11], FLAG_MORTON_SORT);
        assert_eq!(
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            1_244_819,
            "n_splats must match bonsai SF baseline",
        );
        assert_eq!(
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            12_448,
            "k_selected must match top-1% bonsai selection",
        );
        assert_eq!(bytes[20], N_ATTR_GROUPS);
        assert_eq!(bytes[21], 15, "sh_rest_coefs = 15 for sh_degree=3");
        assert_eq!(
            u16::from_le_bytes([bytes[22], bytes[23]]),
            64,
            "n_cells = 64 in V5.2",
        );
        assert_eq!(&bytes[24..32], &[0u8; 8], "reserved bytes must be zero");
        assert_eq!(bytes.len(), 802_152, "V5.2 sidecar total size locked");
    }

    /// Per-cell quantization: residuals within a cell satisfy
    /// `|residual - dequant| <= scale` for each channel.
    #[test]
    fn per_cell_quant_dequant_within_one_step() {
        let n_chan = 3;
        let k = 6;
        // Two cells of 3 rows each.
        let cell_offsets = vec![0u32, 3u32, 6u32];
        // Per-cell values designed to stress per-cell scales:
        // cell 0: channel 0 ∈ [-1, 1], channel 1 ∈ [10, 20], channel 2 ∈ [0, 0.01]
        // cell 1: tighter ranges.
        let residual: Vec<f32> = vec![
            // cell 0
            -1.0, 10.0, 0.0, 0.0, 15.0, 0.005, 1.0, 20.0, 0.01, // cell 1
            -0.1, 100.0, 50.0, 0.0, 100.5, 50.5, 0.1, 101.0, 51.0,
        ];
        let (q, meta) =
            per_cell_affine_quantize(&residual, &cell_offsets, 8, n_chan).expect("quant ok");
        assert_eq!(q.len(), k * n_chan);
        assert_eq!(meta.len(), 2 * n_chan * 2);
        // Dequant and check tolerance.
        for ci in 0..2 {
            let a = cell_offsets[ci] as usize;
            let b = cell_offsets[ci + 1] as usize;
            for c in 0..n_chan {
                let scale = meta[(ci * n_chan + c) * 2];
                let offset = meta[(ci * n_chan + c) * 2 + 1];
                for row in a..b {
                    let idx = row * n_chan + c;
                    let dequant = q[idx] as f32 * scale + offset;
                    let err = (dequant - residual[idx]).abs();
                    assert!(
                        err <= scale * 1.0001 + 1e-6,
                        "cell {} chan {} row {} err={} scale={}",
                        ci,
                        c,
                        row,
                        err,
                        scale
                    );
                }
            }
        }
    }

    // ---------- Decoder round-trip tests ----------

    /// LSB-first unpack inverts pack for an arbitrary boolean vector.
    #[test]
    fn unpack_bitmask_round_trip() {
        let bools: Vec<bool> = (0..73).map(|i| (i * 7) % 3 == 0).collect();
        let packed = pack_bitmask_lsb_first(&bools);
        let unpacked = unpack_bitmask_lsb_first(&packed, bools.len());
        assert_eq!(bools, unpacked);
    }

    /// Bit-unpack inverts bit-pack at the supported depths.
    #[test]
    fn bit_unpack_inverts_pack_8_10_12_16() {
        for &bd in &[8u8, 10, 12, 16] {
            let mask: u32 = if bd == 32 { u32::MAX } else { (1u32 << bd) - 1 };
            let vals: Vec<u32> = (0..37).map(|i| (i as u32 * 31337) & mask).collect();
            let packed = bit_pack_fast(&vals, bd);
            let unpacked = bit_unpack_fast(&packed, vals.len(), bd).expect("unpack ok");
            assert_eq!(vals, unpacked, "round-trip failed for bd={}", bd);
        }
    }

    /// Full encoder -> decoder round-trip on the 10-splat synthetic scene
    /// from the encoder test. The residuals come back to dequant precision
    /// for every group, in SF-ascending order on the selected subset.
    #[test]
    fn synthetic_round_trip_decode_10_splats() {
        let n_splats = 10usize;
        let sh_rest_coefs = 2usize;
        let mut sel_bool = vec![false; n_splats];
        for &i in &[1, 4, 7] {
            sel_bool[i] = true;
        }
        let positions_selected = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let morton_idx = morton_sort_indices(&positions_selected);
        let k = 3usize;
        let make = |n_chan: usize| -> Vec<f32> {
            let mut v = vec![0.0; k * n_chan];
            for kk in 0..k {
                for c in 0..n_chan {
                    v[kk * n_chan + c] = kk as f32 * 0.1 + c as f32 * 0.01;
                }
            }
            v
        };
        // Encoder takes Morton-ordered residuals; the maker above is row-major
        // by SF order. Permute to Morton order before feeding the encoder
        // (here morton_idx is identity so this is a no-op, but we apply it
        // explicitly so the test stays correct if positions change).
        let to_morton = |raw: &[f32], n_chan: usize| -> Vec<f32> {
            let mut out = vec![0f32; raw.len()];
            for (m, &src_row) in morton_idx.iter().enumerate() {
                let src = src_row as usize;
                out[m * n_chan..(m + 1) * n_chan]
                    .copy_from_slice(&raw[src * n_chan..(src + 1) * n_chan]);
            }
            out
        };
        let raw_pos = make(3);
        let raw_rot = make(4);
        let raw_opa = make(1);
        let raw_sca = make(3);
        let raw_dc = make(3);
        let raw_shr = make(sh_rest_coefs * 3);
        let residuals = Residuals {
            k_selected: k,
            sh_rest_coefs,
            pos: to_morton(&raw_pos, 3),
            rot: to_morton(&raw_rot, 4),
            opa: to_morton(&raw_opa, 1),
            sca: to_morton(&raw_sca, 3),
            dc: to_morton(&raw_dc, 3),
            shr: to_morton(&raw_shr, sh_rest_coefs * 3),
        };
        let bit_depths = BitDepths::v5_2();
        let cell_offsets = build_cell_offsets(k, 2);
        let (bytes, _sizes) = encode_v5_2_sidecar(
            n_splats,
            &sel_bool,
            &morton_idx,
            &residuals,
            bit_depths,
            &cell_offsets,
        )
        .expect("encode ok");
        let decoded = decode_v5tail_bytes(&bytes).expect("decode ok");
        assert_eq!(decoded.header.n_splats, n_splats as u32);
        assert_eq!(decoded.header.k_selected, k as u32);
        assert_eq!(decoded.sel_idx, vec![1u32, 4, 7]);
        assert_eq!(decoded.pos.len(), k * 3);
        // Per-cell-affine quantization at depths 8/10/12 means the recovered
        // residual is within one quantization step of the truth. For 12-bit
        // depth (opa/sca) the step is tiny; for 8-bit (pos/dc/shr) we allow
        // generous slack proportional to the input range.
        let range: f32 = 0.4; // matches the make() max minus min
        let tol_pos = range / 255.0 * 1.5;
        for k_sf in 0..k {
            for c in 0..3 {
                let got = decoded.pos[k_sf * 3 + c];
                let want = raw_pos[k_sf * 3 + c];
                assert!(
                    (got - want).abs() <= tol_pos,
                    "pos k={} c={} got={} want={}",
                    k_sf,
                    c,
                    got,
                    want
                );
            }
        }
        // SH-rest residual round-trip (8-bit) — tolerance matches pos above.
        for k_sf in 0..k {
            for c in 0..(sh_rest_coefs * 3) {
                let got = decoded.shr[k_sf * (sh_rest_coefs * 3) + c];
                let want = raw_shr[k_sf * (sh_rest_coefs * 3) + c];
                assert!(
                    (got - want).abs() <= tol_pos,
                    "shr k={} c={} got={} want={}",
                    k_sf,
                    c,
                    got,
                    want
                );
            }
        }
    }

    /// Golden decoder test: parses the actual Python-produced bonsai V5.2
    /// sidecar end-to-end and asserts the recovered header + sel_idx
    /// shapes match the manifest. Skipped silently when the fixture is
    /// not in this checkout.
    #[test]
    fn golden_decode_python_v5_2_sidecar() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("experiments/v5-2-composed/data/sidecar_v5_2.bin");
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping decoder golden test: {} not found", path.display());
                return;
            }
        };
        let decoded = decode_v5tail_bytes(&bytes).expect("decode python golden ok");
        assert_eq!(decoded.header.n_splats, 1_244_819);
        assert_eq!(decoded.header.k_selected, 12_448);
        assert_eq!(decoded.header.sh_rest_coefs, 15);
        assert_eq!(decoded.header.n_cells, 64);
        assert_eq!(decoded.sel_idx.len(), 12_448);
        // SF-ascending invariant: sel_idx strictly increasing.
        assert!(decoded.sel_idx.windows(2).all(|w| w[0] < w[1]));
        // Per-group row counts match K.
        assert_eq!(decoded.pos.len(), 12_448 * 3);
        assert_eq!(decoded.rot.len(), 12_448 * 4);
        assert_eq!(decoded.opa.len(), 12_448);
        assert_eq!(decoded.sca.len(), 12_448 * 3);
        assert_eq!(decoded.dc.len(), 12_448 * 3);
        assert_eq!(decoded.shr.len(), 12_448 * 15 * 3);
    }

    // ---------- Path B (v5.2 Phase D) tests ----------

    /// The Phase D Path B default profile widens opa + sca from 12 → 14
    /// bits. This test locks the BitDepths default so the next person who
    /// edits the struct can't silently revert the bump.
    #[test]
    fn path_b_default_profile_widens_opa_sca_to_14_bits() {
        let bd = BitDepths::v5_2();
        assert_eq!(bd.pos, 8, "Phase D Path B keeps position at 8 bits");
        assert_eq!(bd.rot, 10, "Phase D Path B keeps rotation at 10 bits");
        assert_eq!(bd.opa, 14, "Phase D Path B widens opacity 12 -> 14 bits");
        assert_eq!(bd.sca, 14, "Phase D Path B widens scale 12 -> 14 bits");
        assert_eq!(bd.dc, 8);
        assert_eq!(bd.shr, 8);
        // The legacy profile stays available for v=1 decode tests.
        let v1 = BitDepths::v5_2_v1();
        assert_eq!(v1.opa, 12);
        assert_eq!(v1.sca, 12);
    }

    /// Path B round-trip: encode the same residuals with the legacy v1
    /// profile and the new Path B default, then verify (a) the Path B
    /// payload is larger by the expected per-cell delta, and (b) the
    /// recovered opacity/scale residual is strictly more accurate under
    /// Path B because the quantization step is 4× finer.
    #[test]
    fn path_b_payload_grows_and_opa_sca_precision_improves() {
        let n_splats = 256usize;
        let sh_rest_coefs = 15usize;
        // Select 64 splats (25%) so the payload is large enough that the
        // bit-depth widening shows up as a multi-byte delta but small enough
        // to round-trip in the test budget.
        let mut sel_bool = vec![false; n_splats];
        let mut sel_idx_sf: Vec<u32> = Vec::new();
        for i in 0..n_splats {
            if i % 4 == 0 {
                sel_bool[i] = true;
                sel_idx_sf.push(i as u32);
            }
        }
        let k = sel_idx_sf.len();
        let positions_selected: Vec<[f32; 3]> = sel_idx_sf
            .iter()
            .map(|&i| [i as f32 * 0.01, (i as f32 * 0.013).sin(), 0.0])
            .collect();
        let morton_idx = morton_sort_indices(&positions_selected);

        // Synthetic opacity / scale residuals span a wide range — this is
        // the regime where the 12 → 14 bit widening pays off (the per-cell
        // affine step shrinks by 4×).
        let make = |n_chan: usize, scale: f32| -> Vec<f32> {
            let mut v = vec![0.0_f32; k * n_chan];
            for kk in 0..k {
                for c in 0..n_chan {
                    v[kk * n_chan + c] = ((kk as f32 * 0.31 + c as f32 * 0.17).sin()) * scale;
                }
            }
            v
        };
        let pos_res = make(3, 0.01);
        let rot_res = make(4, 0.02);
        // Big opacity + scale residuals (the Phase D motivation: log_quant_
        // attrs damage runs to ~10 in raw logit/log-scale).
        let opa_res = make(1, 8.0);
        let sca_res = make(3, 8.0);
        let dc_res = make(3, 0.05);
        let shr_res = make(sh_rest_coefs * 3, 0.005);

        let cell_offsets = build_cell_offsets(k, 16);
        let mk_residuals = || Residuals {
            k_selected: k,
            sh_rest_coefs,
            pos: pos_res.clone(),
            rot: rot_res.clone(),
            opa: opa_res.clone(),
            sca: sca_res.clone(),
            dc: dc_res.clone(),
            shr: shr_res.clone(),
        };

        let (v1_bytes, v1_sizes) = encode_v5_2_sidecar(
            n_splats,
            &sel_bool,
            &morton_idx,
            &mk_residuals(),
            BitDepths::v5_2_v1(),
            &cell_offsets,
        )
        .expect("v1 encode");
        let (v2_bytes, v2_sizes) = encode_v5_2_sidecar(
            n_splats,
            &sel_bool,
            &morton_idx,
            &mk_residuals(),
            BitDepths::v5_2(),
            &cell_offsets,
        )
        .expect("v2 encode");

        // The encoder always stamps the current VERSION regardless of which
        // bit-depth profile the caller picked — version is a wire-format
        // identifier, not a profile identifier. Anyone who needs a v=1
        // header byte for a wire-bytes-identical-to-2026-ship sidecar can
        // mint it by post-hoc rewriting bytes 8..10 (this is what the v=2
        // back-compat test in the polyfill does).
        assert_eq!(
            u16::from_le_bytes([v2_bytes[8], v2_bytes[9]]),
            VERSION,
            "encoder must stamp current VERSION ({})",
            VERSION
        );
        assert_eq!(
            u16::from_le_bytes([v1_bytes[8], v1_bytes[9]]),
            VERSION,
            "encoder stamps current VERSION even when using the legacy profile",
        );

        // Opacity payload: raw uncompressed bit-packed bytes scale as
        // `ceil(K * n_chan * bit_depth / 8)`. Under zstd the delta is
        // smaller but still strictly positive when the residual is high-
        // entropy as it is here.
        let opa_v1 = v1_sizes.groups[2].payload_zstd; // GROUP_ORDER index 2
        let opa_v2 = v2_sizes.groups[2].payload_zstd;
        let sca_v1 = v1_sizes.groups[3].payload_zstd;
        let sca_v2 = v2_sizes.groups[3].payload_zstd;
        assert!(
            opa_v2 > opa_v1,
            "opa zstd payload must grow under 14-bit Path B: v1={} v2={}",
            opa_v1,
            opa_v2
        );
        assert!(
            sca_v2 > sca_v1,
            "sca zstd payload must grow under 14-bit Path B: v1={} v2={}",
            sca_v1,
            sca_v2
        );
        // Non-opa/non-sca groups should be byte-identical (same bit depths,
        // same residual data, deterministic zstd).
        for &gi in &[0usize, 1, 4, 5] {
            let lhs = v1_sizes.groups[gi].payload_zstd;
            let rhs = v2_sizes.groups[gi].payload_zstd;
            assert_eq!(
                lhs, rhs,
                "non-Path-B group {} payload changed unexpectedly: v1={} v2={}",
                gi, lhs, rhs
            );
        }

        // Both versions must decode cleanly (back-compat) and Path B must
        // produce more accurate opa/sca residuals on average.
        let dec_v1 = decode_v5tail_bytes(&v1_bytes).expect("decode v1");
        let dec_v2 = decode_v5tail_bytes(&v2_bytes).expect("decode v2");

        // The encoder takes residuals in Morton order; the decoder returns
        // them in SF-ascending order. Re-permute truth into SF order before
        // comparing: residual_sf[k_sf] = residual_morton[inv_morton[k_sf]],
        // where inv_morton[morton_idx[m]] = m.
        let mut inv_morton = vec![0u32; k];
        for (m, &kk) in morton_idx.iter().enumerate() {
            inv_morton[kk as usize] = m as u32;
        }
        let permute = |morton_buf: &[f32], n_chan: usize| -> Vec<f32> {
            let mut out = vec![0f32; k * n_chan];
            for k_sf in 0..k {
                let m = inv_morton[k_sf] as usize;
                out[k_sf * n_chan..(k_sf + 1) * n_chan]
                    .copy_from_slice(&morton_buf[m * n_chan..(m + 1) * n_chan]);
            }
            out
        };
        let opa_truth = permute(&opa_res, 1);
        let sca_truth = permute(&sca_res, 3);

        let opa_err = |dec: &DecodedSidecar| -> f64 {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += (dec.opa[kk] as f64 - opa_truth[kk] as f64).abs();
            }
            acc / k as f64
        };
        let sca_err = |dec: &DecodedSidecar| -> f64 {
            let mut acc = 0.0f64;
            for kk in 0..k {
                for c in 0..3 {
                    acc += (dec.sca[kk * 3 + c] as f64 - sca_truth[kk * 3 + c] as f64).abs();
                }
            }
            acc / (k * 3) as f64
        };

        let opa_err_v1 = opa_err(&dec_v1);
        let opa_err_v2 = opa_err(&dec_v2);
        let sca_err_v1 = sca_err(&dec_v1);
        let sca_err_v2 = sca_err(&dec_v2);
        // 12 → 14 bits = 4× more quant steps → roughly 4× less per-cell
        // dequant error. Assert at least 2.5× improvement to leave slack for
        // cell-boundary effects on the small synthetic distribution.
        assert!(
            opa_err_v2 * 2.5 < opa_err_v1,
            "Path B opa precision did not improve: v1_err={:.4e} v2_err={:.4e}",
            opa_err_v1,
            opa_err_v2
        );
        assert!(
            sca_err_v2 * 2.5 < sca_err_v1,
            "Path B sca precision did not improve: v1_err={:.4e} v2_err={:.4e}",
            sca_err_v1,
            sca_err_v2
        );
    }
}
