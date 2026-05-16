//! `mesonpp` — CLI for the MesonGS++ codec.
//!
//! ```text
//! mesonpp encode <input.ply> <output.meson>
//! mesonpp decode <input.meson> <output.ply>
//! mesonpp info   <input.meson>
//! ```
//!
//! No flags by default — the `EncodeConfig` defaults are what ship as
//! the production `mgs-balanced` preset. Knobs are exposed under
//! `--k-low`, `--k-color`, `--xyz-bits`, `--iters` for benchmarking.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use splatforge_meson::{decode_features, decode_scene, encode_scene, EncodeConfig, MesonError};

#[derive(Parser, Debug)]
#[command(
    name = "mesonpp",
    about = "MesonGS++ post-training 3DGS codec",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Encode a .ply into a .meson container.
    Encode {
        input: PathBuf,
        output: PathBuf,
        /// K-means K for scale/rot/opacity groups.
        #[arg(long, default_value_t = 256)]
        k_low: u16,
        /// K-means K for color (f_dc / f_rest) groups.
        #[arg(long, default_value_t = 256)]
        k_color: u16,
        /// Position-quantization bits per axis.
        #[arg(long, default_value_t = 14)]
        xyz_bits: u8,
        /// K-means Lloyd's iterations.
        #[arg(long, default_value_t = 10)]
        iters: u32,
        /// Deterministic seed for K-means init.
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
        /// Preserve original PLY splat order (adds ~32 bpp of perm data).
        /// Default is Morton-ordered output — viewers and the smoke test
        /// don't depend on input ordering.
        #[arg(long, default_value_t = false)]
        preserve_order: bool,
    },
    /// Decode a .meson container back into a .ply.
    Decode {
        input: PathBuf,
        output: PathBuf,
    },
    /// Print container metadata without doing the full decode.
    Info { input: PathBuf },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("mesonpp: error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), MesonError> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Encode {
            input,
            output,
            k_low,
            k_color,
            xyz_bits,
            iters,
            seed,
            preserve_order,
        } => {
            let cfg = EncodeConfig {
                kmeans_k_low: k_low,
                kmeans_k_color: k_color,
                xyz_bits,
                kmeans_iters: iters,
                seed,
                preserve_order,
            };
            let t0 = Instant::now();
            let scene = splatforge_ply::read_ply(&input)?;
            let read_secs = t0.elapsed().as_secs_f64();
            let n = scene.splats.len();
            let in_size = fs::metadata(&input).map(|m| m.len()).unwrap_or(0);

            let t1 = Instant::now();
            let bytes = encode_scene(&scene, &cfg)?;
            let enc_secs = t1.elapsed().as_secs_f64();
            fs::write(&output, &bytes)?;
            let out_size = bytes.len() as u64;
            let ratio = in_size as f64 / out_size as f64;
            println!(
                "encoded {} splats in {:.2}s (read {:.2}s) — {:.1} MB → {:.2} MB ({:.2}× ratio)",
                n,
                enc_secs,
                read_secs,
                in_size as f64 / (1024.0 * 1024.0),
                out_size as f64 / (1024.0 * 1024.0),
                ratio,
            );
            Ok(())
        }
        Cmd::Decode { input, output } => {
            let t0 = Instant::now();
            let bytes = fs::read(&input)?;
            let scene = decode_scene(&bytes)?;
            let dec_secs = t0.elapsed().as_secs_f64();
            splatforge_ply::write_ply(&scene, &output)?;
            println!(
                "decoded {} splats in {:.2}s",
                scene.splats.len(),
                dec_secs
            );
            Ok(())
        }
        Cmd::Info { input } => {
            let bytes = fs::read(&input)?;
            let layout = splatforge_meson::container_layout(&bytes)?;
            let feats = decode_features(&bytes)?;
            println!("anchors: {}", feats.n);
            println!("sh_degree: {}", feats.sh_degree);
            println!("file_size: {} bytes", bytes.len());
            println!("meta_json: {} bytes", layout.meta_bytes);
            for (i, sz) in layout.stream_sizes.iter().enumerate() {
                let name = match i {
                    0 => "xyz",
                    1 => "scale",
                    2 => "rot",
                    3 => "opacity",
                    4 => "f_dc",
                    5 => "f_rest",
                    6 => "perm",
                    7 => "codebooks",
                    _ => "?",
                };
                println!("stream[{}] {}: {} bytes ({:.2} bpp)",
                    i,
                    name,
                    sz,
                    8.0 * (*sz as f64) / (feats.n as f64),
                );
            }
            Ok(())
        }
    }
}
