//! `splatforge-posthac` CLI — encode and decode PostHAC bitstreams.
//!
//! Subcommands:
//!   - `decode`: read `.pthc`, range-decode codes against the embedded
//!     hyperprior, dequantize to f32 attrs, write Inria 3DGS PLY.
//!   - `info`: print the container header in human-readable form.
//!   - `roundtrip-bytes`: given a `.pthc` + a `--raw-codes <file>` of
//!     raw 8-bit codes (n × d), encode the codes against Rust predict_all,
//!     decode, and assert bit-exact recovery.
//!
//! Encoding from a raw PLY is intentionally not yet supported here — the
//! Python training step (`apps/diff-repack/posthac_codec.py encode ...`)
//! is the canonical producer of `.pthc` files for now. The Rust binary
//! handles everything downstream: decode for production, and the
//! roundtrip command for codec validation.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::{Parser, Subcommand};
use splatforge_posthac::*;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "splatforge-posthac", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the container header.
    Info {
        /// Input `.pthc` path.
        input: PathBuf,
    },
    /// Decode `.pthc` to a flat raw-codes binary (header + uint8 stream).
    /// Suitable for piping into a downstream PLY writer.
    Decode {
        /// Input `.pthc`.
        input: PathBuf,
        /// Output raw-codes file (header: u32 n, u32 d; then n*d uint8).
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Validate the codec end-to-end on a `.pthc` + raw-codes pair.
    RoundtripBytes {
        /// Input `.pthc`.
        input: PathBuf,
        /// File with the original 8-bit codes (same layout as `decode --out`).
        #[arg(long)]
        raw_codes: PathBuf,
    },
    /// Re-encode raw 8-bit codes against the hyperprior of an input `.pthc`,
    /// then write a new fully-self-contained `.pthc`. Use this when the
    /// Python training pipeline produced a `.pthc` with a broken (or
    /// platform-divergent) bitstream — the trained weights + positions are
    /// preserved, but the compressed payload is replaced with a Rust-emitted
    /// one that round-trips bit-exact with `splatforge-posthac decode`.
    Encode {
        /// Input `.pthc` (provides weights + positions).
        input: PathBuf,
        /// File with the original 8-bit codes (same layout as `decode --out`).
        #[arg(long)]
        raw_codes: PathBuf,
        /// Output `.pthc`.
        #[arg(short, long)]
        out: PathBuf,
    },
}

fn read_codes_file(path: &PathBuf) -> std::io::Result<(usize, usize, Vec<u8>)> {
    let bytes = fs::read(path)?;
    let mut cur = Cursor::new(bytes);
    let n = cur.read_u32::<LittleEndian>()? as usize;
    let d = cur.read_u32::<LittleEndian>()? as usize;
    let mut codes = vec![0u8; n * d];
    cur.read_exact(&mut codes)?;
    Ok((n, d, codes))
}

fn cmd_info(p: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(&p)?;
    let c = read_container(&bytes)?;
    println!("PostHAC container: {}", p.display());
    println!("  splats:    {}", c.header.n);
    println!("  attrs:     {}", c.header.d);
    println!("  sh_degree: {}", c.header.sh_degree);
    println!("  hyperprior:");
    println!("    grid_levels = {}", c.header.config.grid_levels);
    println!("    features = {}", c.header.config.features_per_level);
    println!(
        "    log2_hashmap_size = {}",
        c.header.config.log2_hashmap_size
    );
    println!("    mlp_hidden = {}", c.header.config.mlp_hidden);
    println!("    n_attrs = {}", c.header.config.n_attrs);
    println!(
        "  hyperprior bytes (table+MLP): ~{:.2} MB",
        (c.weights.grid_tables.len()
            + c.weights.fc1_w.len()
            + c.weights.fc1_b.len()
            + c.weights.fc2_w.len()
            + c.weights.fc2_b.len()) as f64
            * 4.0
            / 1e6
    );
    println!(
        "  positions bytes: {:.2} MB",
        c.positions.len() as f64 * 4.0 / 1e6
    );
    println!(
        "  compressed bytes: {:.2} MB",
        c.compressed.len() as f64 * 4.0 / 1e6
    );
    let total = bytes.len() as f64;
    let raw_attrs = (c.header.n * c.header.d) as f64;
    println!(
        "  total container: {:.2} MB ({:.2}× over raw 8-bit attrs of {:.2} MB)",
        total / 1e6,
        raw_attrs / total,
        raw_attrs / 1e6,
    );
    Ok(())
}

fn cmd_decode(input: PathBuf, out: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(&input)?;
    let c = read_container(&bytes)?;
    let n = c.header.n as usize;
    let d = c.header.d as usize;

    eprintln!(
        "[decode] N={} D={} — running predict_all() in Rust",
        n, d
    );
    let predictions = predict_all(
        &c.positions,
        c.header.pos_mn,
        c.header.pos_mx,
        &c.header.config,
        &c.weights,
    );
    eprintln!("[decode] range-decoding {} symbols", n * d);
    let codes = decode_codes(&c.compressed, n, d, &predictions)?;
    eprintln!("[decode] writing {} bytes to {}", codes.len(), out.display());
    let mut f = fs::File::create(&out)?;
    f.write_u32::<LittleEndian>(n as u32)?;
    f.write_u32::<LittleEndian>(d as u32)?;
    f.write_all(&codes)?;
    Ok(())
}

fn cmd_roundtrip(
    input: PathBuf,
    raw_codes: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(&input)?;
    let c = read_container(&bytes)?;
    let n = c.header.n as usize;
    let d = c.header.d as usize;
    let (n_raw, d_raw, raw) = read_codes_file(&raw_codes)?;
    if (n_raw, d_raw) != (n, d) {
        return Err(format!(
            "shape mismatch: container is N={} D={}, codes file is N={} D={}",
            n, d, n_raw, d_raw
        )
        .into());
    }
    let predictions = predict_all(
        &c.positions,
        c.header.pos_mn,
        c.header.pos_mx,
        &c.header.config,
        &c.weights,
    );

    let compressed = encode_codes(&raw, n, d, &predictions)?;
    let raw_bytes = (n * d) as f64;
    let comp_bytes = (compressed.len() * 4) as f64;
    println!(
        "encoded {:.2} MB → {:.2} MB ({:.2}× over 8-bit, {:.3} bits/symbol)",
        raw_bytes / 1e6,
        comp_bytes / 1e6,
        raw_bytes / comp_bytes,
        comp_bytes * 8.0 / (n * d) as f64,
    );

    let decoded = decode_codes(&compressed, n, d, &predictions)?;
    let mut mismatches = 0;
    for i in 0..raw.len() {
        if decoded[i] != raw[i] {
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        println!("✓ round-trip bit-exact across {} symbols", raw.len());
    } else {
        return Err(format!("{} mismatches out of {}", mismatches, raw.len()).into());
    }
    Ok(())
}

fn cmd_encode(
    input: PathBuf,
    raw_codes: PathBuf,
    out: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(&input)?;
    let c = read_container(&bytes)?;
    let n = c.header.n as usize;
    let d = c.header.d as usize;
    let (n_raw, d_raw, raw) = read_codes_file(&raw_codes)?;
    if (n_raw, d_raw) != (n, d) {
        return Err(format!(
            "shape mismatch: container N={} D={} vs codes N={} D={}",
            n, d, n_raw, d_raw
        )
        .into());
    }
    eprintln!("[encode] N={n} D={d} — predict_all() in Rust");
    let predictions = predict_all(
        &c.positions,
        c.header.pos_mn,
        c.header.pos_mx,
        &c.header.config,
        &c.weights,
    );
    eprintln!("[encode] range-coding {} symbols", n * d);
    let compressed = encode_codes(&raw, n, d, &predictions)?;
    let mut buf = Vec::new();
    write_container(&mut buf, &c.header, &c.weights, &c.positions, &compressed)?;
    fs::write(&out, &buf)?;
    let raw_bytes = (n * d) as f64;
    let comp_bytes = (compressed.len() * 4) as f64;
    println!(
        "wrote {} ({:.2} MB total). compressed {:.2} MB → {:.2} MB ({:.2}× over 8-bit)",
        out.display(),
        buf.len() as f64 / 1e6,
        raw_bytes / 1e6,
        comp_bytes / 1e6,
        raw_bytes / comp_bytes,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info { input } => cmd_info(input),
        Cmd::Decode { input, out } => cmd_decode(input, out),
        Cmd::RoundtripBytes { input, raw_codes } => cmd_roundtrip(input, raw_codes),
        Cmd::Encode {
            input,
            raw_codes,
            out,
        } => cmd_encode(input, raw_codes, out),
    }
}
