//! FRONT-F helper: mint a per-splat `J_sh_rest` Jacobian for a GT PLY using the
//! REAL closed-form proxy the CLI's `--auto-jacobian` uses
//! (`catetus_jacobian::compute_jacobian`), and write it as a CLI-compatible
//! stored-mode `.npz` (single ZIP entry `J_sh_rest.npy`, no ZIP64). The CLI's
//! joint-Jacobian loader accepts a lone `J_sh_rest` as the selection score.
//!
//! We hand-write the ZIP rather than shelling to numpy because numpy 2.x emits
//! a ZIP64 local header the CLI's minimal parser rejects. The proxy is pure
//! Rust, no GPU, no network.
//!
//! Run:
//!   cargo run -p catetus-optimize --release --example frontf_mint_jacobian -- \
//!       <GT.ply> <out.npz>

use catetus_ply::read_ply;
use std::io::Write;
use std::path::Path;

/// CRC-32 (IEEE), for the ZIP local + central headers.
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (n, slot) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// NPY (v1.0) blob for a 1-D little-endian f32 array.
fn npy_f32_1d(vals: &[f32]) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        vals.len()
    );
    let base = 10 + header.len() + 1;
    let pad = (64 - (base % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Single-entry stored-mode ZIP (no ZIP64) matching the CLI parser:
/// local sig 0x04034b50, compression 0, csize == usize == payload.len().
fn write_stored_npz(path: &Path, entry_name: &str, payload: &[u8]) {
    let crc = crc32(payload);
    let n = payload.len() as u32;
    let name = entry_name.as_bytes();
    let mut f = std::fs::File::create(path).expect("create npz");
    // local file header
    f.write_all(&0x0403_4b50u32.to_le_bytes()).unwrap();
    f.write_all(&20u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&crc.to_le_bytes()).unwrap();
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&(name.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(name).unwrap();
    f.write_all(payload).unwrap();
    // central directory
    let local_off = 0u32;
    let cd_off = 30 + name.len() as u32 + n;
    f.write_all(&0x0201_4b50u32.to_le_bytes()).unwrap();
    f.write_all(&20u16.to_le_bytes()).unwrap();
    f.write_all(&20u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&crc.to_le_bytes()).unwrap();
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&(name.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u32.to_le_bytes()).unwrap();
    f.write_all(&local_off.to_le_bytes()).unwrap();
    f.write_all(name).unwrap();
    let cd_size = 46 + name.len() as u32;
    // EOCD
    f.write_all(&0x0605_4b50u32.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&cd_size.to_le_bytes()).unwrap();
    f.write_all(&cd_off.to_le_bytes()).unwrap();
    f.write_all(&0u16.to_le_bytes()).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gt = read_ply(Path::new(&args[1])).expect("read GT ply");
    let res = catetus_jacobian::compute_jacobian(&gt); // returns JacobianResult
    assert_eq!(res.j_sh_rest.len(), gt.splats.len());
    let mn = res.j_sh_rest.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = res.j_sh_rest.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let npy = npy_f32_1d(&res.j_sh_rest);
    write_stored_npz(Path::new(&args[2]), "J_sh_rest.npy", &npy);
    let total = std::fs::metadata(&args[2]).map(|m| m.len()).unwrap_or(0);
    println!(
        "minted J_sh_rest: n={} min={} max={} npy_bytes={} npz_bytes={} -> {}",
        res.j_sh_rest.len(),
        mn,
        mx,
        npy.len(),
        total,
        args[2]
    );
}
