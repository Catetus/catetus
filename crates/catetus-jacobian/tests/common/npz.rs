//! Minimal stored-mode NPZ reader for tests.
//!
//! Mirrors the logic in `catetus-cli/src/main.rs` (the production
//! `--jacobian-sidecar` loader) so this test crate doesn't depend on
//! `zip` / `npyz`. Only `numpy.savez` (uncompressed) archives are
//! supported; everything else returns `Err`. Only rank-1 `<f4` arrays
//! are decoded.
//!
//! See `catetus-cli/src/main.rs` for the full rationale (stored-mode
//! ZIP local-file-header parsing + an NPY v1/v2 header reader).

use std::path::Path;

use anyhow::{anyhow, Context, Result};

struct NpzEntry {
    name: String,
    data_off: usize,
    data_end: usize,
}

fn list_npz_entries(bytes: &[u8]) -> Result<Vec<NpzEntry>> {
    let mut out: Vec<NpzEntry> = Vec::new();
    let mut cursor: usize = 0;
    while cursor + 30 <= bytes.len() {
        let sig = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        if sig != 0x0403_4b50 {
            break;
        }
        let compression =
            u16::from_le_bytes(bytes[cursor + 8..cursor + 10].try_into().unwrap());
        let csize =
            u32::from_le_bytes(bytes[cursor + 18..cursor + 22].try_into().unwrap()) as usize;
        let usize_ =
            u32::from_le_bytes(bytes[cursor + 22..cursor + 26].try_into().unwrap()) as usize;
        let name_len =
            u16::from_le_bytes(bytes[cursor + 26..cursor + 28].try_into().unwrap()) as usize;
        let extra_len =
            u16::from_le_bytes(bytes[cursor + 28..cursor + 30].try_into().unwrap()) as usize;
        let name_off = cursor + 30;
        let name_end = name_off + name_len;
        let data_off = name_end + extra_len;
        let data_end = data_off + csize;
        if data_end > bytes.len() {
            return Err(anyhow!(
                "NPZ local header at offset {cursor} declares data range \
                 [{data_off}, {data_end}) that exceeds file length {}",
                bytes.len()
            ));
        }
        if compression != 0 {
            return Err(anyhow!(
                "NPZ entry uses compression method {compression}; only stored-mode \
                 (0) NPZ archives are supported. Re-save with `numpy.savez` (NOT \
                 `savez_compressed`)."
            ));
        }
        if csize != usize_ {
            return Err(anyhow!(
                "stored-mode NPZ entry size mismatch: compressed={csize} \
                 uncompressed={usize_}"
            ));
        }
        let name = std::str::from_utf8(&bytes[name_off..name_end])
            .map_err(|e| anyhow!("non-UTF8 NPZ entry name at offset {cursor}: {e}"))?
            .to_string();
        out.push(NpzEntry {
            name,
            data_off,
            data_end,
        });
        cursor = data_end;
    }
    Ok(out)
}

fn parse_npy_f32_1d(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() < 10 || &blob[0..6] != b"\x93NUMPY" {
        return Err(anyhow!("not an NPY file (bad magic)"));
    }
    let major = blob[6];
    let (header_len, header_start) = if major == 1 {
        let l = u16::from_le_bytes(blob[8..10].try_into().unwrap()) as usize;
        (l, 10usize)
    } else if major == 2 {
        if blob.len() < 12 {
            return Err(anyhow!("NPY v2 header truncated"));
        }
        let l = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize;
        (l, 12usize)
    } else {
        return Err(anyhow!("unsupported NPY major version {major}"));
    };
    let header_end = header_start + header_len;
    if header_end > blob.len() {
        return Err(anyhow!("NPY header length exceeds file"));
    }
    let header = std::str::from_utf8(&blob[header_start..header_end])
        .map_err(|e| anyhow!("non-UTF8 NPY header: {e}"))?;

    fn field<'a>(h: &'a str, key: &str) -> Option<&'a str> {
        let needle = format!("'{key}':");
        let i = h.find(&needle)?;
        Some(h[i + needle.len()..].trim_start())
    }

    let descr = field(header, "descr").ok_or_else(|| anyhow!("NPY header missing 'descr'"))?;
    let descr_val = descr
        .trim()
        .trim_start_matches('\'')
        .split('\'')
        .next()
        .unwrap_or("");
    if descr_val != "<f4" && descr_val != "|f4" {
        return Err(anyhow!(
            "NPY array dtype is {descr_val:?}; expected '<f4' (little-endian float32)"
        ));
    }
    let fortran = field(header, "fortran_order").unwrap_or("False");
    if fortran.trim_start().starts_with("True") {
        return Err(anyhow!("NPY array is fortran-order; expected C-order"));
    }
    let shape_raw =
        field(header, "shape").ok_or_else(|| anyhow!("NPY header missing 'shape'"))?;
    let lp = shape_raw
        .find('(')
        .ok_or_else(|| anyhow!("NPY shape missing '('"))?;
    let rp = shape_raw[lp..]
        .find(')')
        .ok_or_else(|| anyhow!("NPY shape missing ')'"))?;
    let inner = &shape_raw[lp + 1..lp + rp];
    let dims: Vec<usize> = inner
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                Some(s.parse::<usize>().ok())
            }
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow!("NPY shape parse failed: {inner:?}"))?;
    if dims.len() != 1 {
        return Err(anyhow!("NPY array has rank {}; expected 1", dims.len()));
    }
    let n = dims[0];
    let data_off = header_end;
    let bytes_needed = n
        .checked_mul(4)
        .ok_or_else(|| anyhow!("NPY size overflow"))?;
    if data_off + bytes_needed > blob.len() {
        return Err(anyhow!(
            "NPY data range exceeds blob: need {bytes_needed} bytes from offset {data_off}, \
             have {}",
            blob.len() - data_off
        ));
    }
    let mut out = Vec::with_capacity(n);
    let mut p = data_off;
    for _ in 0..n {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&blob[p..p + 4]);
        out.push(f32::from_le_bytes(buf));
        p += 4;
    }
    Ok(out)
}

/// Load a rank-1 `<f4` array named `entry_name` from a stored-mode NPZ.
pub fn load_npz_array_f32(path: &Path, entry_name: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading NPZ {}", path.display()))?;
    let entries = list_npz_entries(&bytes)?;
    let entry = entries
        .iter()
        .find(|e| e.name == entry_name)
        .ok_or_else(|| {
            anyhow!(
                "NPZ {} does not contain `{}` (available: {})",
                path.display(),
                entry_name,
                entries
                    .iter()
                    .map(|e| e.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        })?;
    parse_npy_f32_1d(&bytes[entry.data_off..entry.data_end])
        .with_context(|| format!("parsing NPZ entry `{entry_name}`"))
}
