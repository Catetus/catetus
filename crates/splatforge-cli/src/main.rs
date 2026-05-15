#![deny(clippy::all)]
//! `splatforge` — the SplatForge command-line tool.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use splatforge_core::{format_from_extension, format_from_magic, AnalyzeReport, Splat, SplatScene};
use splatforge_gltf::{inspect_gltf, read_glb, read_gltf, write_glb, write_gltf, WriteOpts};
use splatforge_optimize::{preset, write_tileset, TilesetOpts};
use splatforge_ply::{
    decode_progressive_file, encode_progressive_file, read_mgs2_header, read_ply, write_ply,
};
use splatforge_spz::{read_spz, write_spz};
use splatforge_usd::{read_usda, read_usdc, write_usda, write_usdc, UsdWriteOpts};

mod license;
use license::{cmd_license_install, cmd_license_refresh, cmd_license_status, cmd_serve};

#[derive(Parser, Debug)]
#[command(
    name = "splatforge",
    version,
    about = "Gaussian Splat optimization CLI"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Analyze a splat file and print a JSON report.
    Analyze {
        /// Input file.
        input: PathBuf,
        /// Pretty-print the JSON output.
        #[arg(long)]
        pretty: bool,
    },
    /// Inspect a splat file; non-zero exit on validation failure.
    Inspect {
        /// Input file.
        input: PathBuf,
    },
    /// Convert between supported formats.
    Convert {
        /// Input file.
        input: PathBuf,
        /// Target format: ply, spz, gltf, glb, usda, usdc.
        #[arg(long, value_name = "FORMAT")]
        to: String,
        /// Output path.
        #[arg(long, short = 'o')]
        out: PathBuf,
    },
    /// Run an optimization pipeline.
    Optimize {
        /// Input file.
        input: PathBuf,
        /// Preset name (see SPEC-0006).
        #[arg(long)]
        preset: String,
        /// Emit chunked glTF.
        #[arg(long)]
        chunked: bool,
        /// Compression mode for the output buffers.
        ///   `zstd` — also emit zstd-compressed `.bin.zst` sidecars (legacy
        ///   behavior of the bare `--compress` flag; halves the on-disk size
        ///   of quantized presets at zero quality cost, served via HTTP with
        ///   `Content-Encoding: zstd`).
        ///   `spz`  — pack the scene as a single SPZ blob inside the GLB
        ///   under the `KHR_gaussian_splatting_compression_spz` extension.
        ///   Requires `--target glb`.
        #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "zstd")]
        compress: Option<String>,
        /// Output container: `gltf` (default) or `glb`. `glb` produces a
        /// self-contained binary glTF; required for `--compress spz`.
        #[arg(long, value_name = "FORMAT")]
        target: Option<String>,
        /// Output path; defaults to "<input>.optimized.gltf" (or `.glb` when
        /// `--target glb` is set).
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        /// Output DIRECTORY for multi-tile presets (`geospatial`). The
        /// directory is created if missing and will contain `tileset.json`
        /// plus one `.glb` per LOD level. Mutually exclusive with `--out`.
        #[arg(long, value_name = "DIR")]
        output_dir: Option<PathBuf>,
        /// Emit machine-readable `PROGRESS frac=<0..1> stage=<name>` lines
        /// to stdout before each pipeline pass, plus one terminal `frac=1.0`.
        /// Consumed by the Modal worker to stream live progress to the UI.
        /// Off by default so interactive CLI output stays clean.
        #[arg(long)]
        progress: bool,
    },
    /// Serve a tiny static preview viewer.
    Preview {
        /// Input file.
        input: PathBuf,
        /// HTTP port.
        #[arg(long, default_value = "5170")]
        port: u16,
    },
    /// Compare two splat files.
    Diff {
        /// Before file.
        before: PathBuf,
        /// After file.
        after: PathBuf,
        /// Output directory.
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        /// Mean pixel-difference threshold (0..1) for pass/fail.
        #[arg(long, default_value_t = 0.03_f32)]
        threshold: f32,
    },
    /// Microbenchmark analyze on a file.
    Benchmark {
        /// Input file.
        input: PathBuf,
        /// Device profile (currently unused).
        #[arg(long)]
        device_profile: Option<String>,
    },
    /// Run a benchmark corpus.
    Corpus {
        #[command(subcommand)]
        cmd: CorpusCmd,
    },
    /// Upload a splat to SplatForge Cloud, poll until done, return the URL.
    ///
    /// Reads `SPLATFORGE_API_KEY` and `SPLATFORGE_API_URL`
    /// (default `https://splatforge-api.fly.dev`) from the environment.
    /// On success prints the public output URL to stdout. On failure exits
    /// non-zero with a one-line error to stderr.
    Submit {
        /// Local splat file (.ply / .spz / .glb).
        input: PathBuf,
        /// Optimize preset (web-mobile / size-min / web-desktop / etc).
        #[arg(long, default_value = "web-mobile")]
        preset: String,
        /// Optional human-readable label stamped on the job for audit.
        #[arg(long)]
        label: Option<String>,
        /// Optional webhook URL the Cloud fires on terminal state.
        #[arg(long)]
        webhook_url: Option<String>,
        /// Poll interval in seconds.
        #[arg(long, default_value_t = 5)]
        poll_secs: u64,
        /// Hard timeout in seconds. 0 = no timeout.
        #[arg(long, default_value_t = 900)]
        timeout_secs: u64,
        /// Don't poll — print the job id and return immediately.
        #[arg(long)]
        no_wait: bool,
    },
    /// Render a fidelity report (8-orbit ΔE94 / SSIM / pixelmatch) for a
    /// candidate against a baseline. Writes a `report.json` next to the
    /// candidate; exits non-zero if mean ΔE94 exceeds `--threshold`.
    Fidelity {
        /// Optimized candidate file.
        candidate: PathBuf,
        /// Baseline file (typically the original / lossless-repack output).
        #[arg(long, short = 'b')]
        baseline: PathBuf,
        /// Output directory for `report.json` and the rendered frames.
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        /// Mean ΔE94 threshold (0..1). Default matches SplatBench `pass`.
        #[arg(long, default_value_t = 0.03_f32)]
        threshold: f32,
    },
    /// Score a single PLY with the v0.4 fidelity MLP. Predict-only —
    /// the baseline PLY is optional (when omitted, the candidate is
    /// compared against the canonical lossless-repack identity profile
    /// baked into `splatforge-fidelity`). Emits a JSON `ScoreReport`.
    FidelityScore {
        /// Candidate PLY.
        candidate: std::path::PathBuf,
        /// Optional baseline PLY.
        #[arg(long, short = 'b')]
        baseline: Option<std::path::PathBuf>,
        /// Pretty-print the JSON output.
        #[arg(long)]
        pretty: bool,
    },
    /// Manage the Pro on-prem license (install, status, refresh). The
    /// license file is `splatforge.lic` — Ed25519-signed JSON minted by
    /// the SplatForge API. Required to run `splatforge serve`.
    License {
        #[command(subcommand)]
        cmd: LicenseCmd,
    },
    /// Run the on-prem SplatForge Pro server. Reads the license at
    /// `~/.splatforge/license.lic` (override with `--license`), verifies
    /// the embedded Ed25519 signature + offline-grace window, then binds
    /// the optimize/serve HTTP surface inside the customer's VPC.
    Serve {
        /// Bind address.
        #[arg(long, default_value = "0.0.0.0:8080")]
        bind: String,
        /// Path to the license file.
        #[arg(long)]
        license: Option<PathBuf>,
        /// Heartbeat endpoint base URL (the SplatForge API). Heartbeats
        /// are skipped entirely when `SPLATFORGE_NO_TELEMETRY=1` is set
        /// in the environment.
        #[arg(long, default_value = "https://splatforge-api.fly.dev")]
        api_base: String,
        /// Number of active seats reported in the heartbeat. Required by
        /// the license enforcement path; the server refuses to start if
        /// this exceeds the license's `seats` claim.
        #[arg(long, default_value_t = 1)]
        active_seats: u32,
    },
    /// Reorder splats in a scene by 3D Morton code of their position,
    /// then re-emit the same file format. The intent is to AMPLIFY the
    /// natural spatial clustering that trained-3DGS PLYs already have,
    /// so the runtime depth-key radix sort gets cache-friendly access
    /// patterns. Render output is byte-identical (the renderer sorts by
    /// depth anyway), but the per-frame sort runs measurably faster on
    /// real scenes (see novel-5 in the research execution log).
    ///
    /// Quantizes position to 16 bits per axis over the scene bounding
    /// box, interleaves x/y/z bits into a 48-bit Morton code packed in
    /// u64, then `sort_by_key` stable. Pass `--in <file>` and
    /// `--out <file>`; both must be the same format. Currently
    /// supports PLY in/out; other formats are loaded via `load_scene`
    /// and written by their respective writer crate.
    MortonPermute {
        /// Input scene file. Auto-detects format from extension/magic.
        #[arg(long, short = 'i')]
        input: PathBuf,
        /// Output scene file. Format inferred from extension; must be
        /// a writable format (`ply` today).
        #[arg(long, short = 'o')]
        out: PathBuf,
    },
    /// Progressive `.mgs2` bitstream codec (Phase 1).
    ///
    /// Encodes an Inria-style 3DGS PLY into a byte-streamable bitstream
    /// where the first bytes hold the highest-importance splats
    /// (`opacity × det(scale)^(2/3)`), and the remaining bytes the rest
    /// in descending importance order. A partial download is a valid
    /// (lower-quality) PLY. See `docs/perf/progressive-bitstream-spec.md`
    /// for the format and the D1 streaming preset for the product surface.
    Progressive {
        #[command(subcommand)]
        cmd: ProgressiveCmd,
    },
    /// Validate an asset against a SplatForge-supported standard
    /// (KHR_gaussian_splatting today; OpenUSD when the USDC reader path
    /// is wired). Wraps `splatforge-khr-validate` for the glTF case.
    SpecCheck {
        /// Input file (.gltf / .glb / .usdc / .usda).
        input: PathBuf,
        /// Spec to check against. Default: auto-detect from extension.
        #[arg(long)]
        spec: Option<String>,
        /// Emit a JSON report instead of human-readable lines.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ProgressiveCmd {
    /// Encode an Inria 3DGS PLY into a `.mgs2` progressive bitstream.
    /// Output is byte-streamable: a partial download is itself a valid
    /// (coarse) Inria-3DGS PLY when run through `progressive decode`.
    Encode {
        /// Input PLY (binary little-endian Inria 3DGS).
        #[arg(long, short = 'i')]
        input: PathBuf,
        /// Output `.mgs2` path.
        #[arg(long, short = 'o')]
        output: PathBuf,
    },
    /// Decode a `.mgs2` (possibly truncated) into a valid PLY.
    ///
    /// Pass `--partial-bytes N` to simulate "only the first N bytes of
    /// the bitstream were downloaded" — the writer emits a PLY
    /// containing every splat that fully fits in the cut, in
    /// descending-importance order. Without `--partial-bytes` the
    /// full bitstream is decoded.
    Decode {
        /// Input `.mgs2` path.
        #[arg(long, short = 'i')]
        input: PathBuf,
        /// Output PLY path.
        #[arg(long, short = 'o')]
        output: PathBuf,
        /// Decode only the first N bytes (simulates a partial download).
        #[arg(long)]
        partial_bytes: Option<u64>,
    },
    /// Print the parsed `.mgs2` header summary (no payload decode).
    Info {
        /// Input `.mgs2` path.
        #[arg(long, short = 'i')]
        input: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum CorpusCmd {
    /// Run a named benchmark suite.
    Run {
        /// Suite name (e.g. "smoke").
        suite: String,
    },
}

#[derive(Subcommand, Debug)]
enum LicenseCmd {
    /// Install a license file by copying it to `~/.splatforge/license.lic`.
    /// Verifies the signature before installing — refuses to clobber the
    /// existing license if the new one is invalid.
    Install {
        /// Path to the `splatforge.lic` to install.
        path: PathBuf,
    },
    /// Print the current license status (org, seats, plan, valid_until,
    /// last_refresh, offline grace remaining). Exits 0 when valid (with
    /// or without grace), 1 otherwise — so a customer cron can gate
    /// other automation on `splatforge license status`.
    Status {
        /// Override the license path.
        #[arg(long)]
        license: Option<PathBuf>,
    },
    /// Hit `/v1/license/refresh` on the API, replace the on-disk license
    /// if the API hands back a fresher one, and reset the `last_refresh`
    /// sidecar so the offline-grace clock starts over.
    Refresh {
        /// Override the license path.
        #[arg(long)]
        license: Option<PathBuf>,
        /// Override the API base URL.
        #[arg(long, default_value = "https://splatforge-api.fly.dev")]
        api_base: String,
    },
}

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt::try_init();
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("splatforge: error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn dispatch(cli: Cli) -> Result<()> {
    match cli.cmd {
        Command::Analyze { input, pretty } => cmd_analyze(&input, pretty),
        Command::Inspect { input } => cmd_inspect(&input),
        Command::Convert { input, to, out } => cmd_convert(&input, &to, &out),
        Command::Optimize {
            input,
            preset,
            chunked,
            compress,
            target,
            out,
            output_dir,
            progress,
        } => cmd_optimize(
            &input,
            &preset,
            chunked,
            compress.as_deref(),
            target.as_deref(),
            out.as_deref(),
            output_dir.as_deref(),
            progress,
        ),
        Command::Preview { input, port } => cmd_preview(&input, port),
        Command::Diff {
            before,
            after,
            out,
            threshold,
        } => cmd_diff(&before, &after, out.as_deref(), threshold),
        Command::Benchmark {
            input,
            device_profile,
        } => cmd_benchmark(&input, device_profile.as_deref()),
        Command::Corpus { cmd } => match cmd {
            CorpusCmd::Run { suite } => cmd_corpus_run(&suite),
        },
        Command::Submit {
            input,
            preset,
            label,
            webhook_url,
            poll_secs,
            timeout_secs,
            no_wait,
        } => cmd_submit(
            &input,
            &preset,
            label.as_deref(),
            webhook_url.as_deref(),
            poll_secs,
            timeout_secs,
            no_wait,
        ),
        Command::Fidelity {
            candidate,
            baseline,
            out,
            threshold,
        } => cmd_fidelity(&candidate, &baseline, out.as_deref(), threshold),
        Command::FidelityScore {
            candidate,
            baseline,
            pretty,
        } => cmd_fidelity_score(&candidate, baseline.as_deref(), pretty),
        Command::License { cmd } => match cmd {
            LicenseCmd::Install { path } => cmd_license_install(&path),
            LicenseCmd::Status { license } => cmd_license_status(license.as_deref()),
            LicenseCmd::Refresh { license, api_base } => {
                cmd_license_refresh(license.as_deref(), &api_base)
            }
        },
        Command::Serve {
            bind,
            license,
            api_base,
            active_seats,
        } => cmd_serve(&bind, license.as_deref(), &api_base, active_seats),
        Command::SpecCheck { input, spec, json } => cmd_spec_check(&input, spec.as_deref(), json),
        Command::MortonPermute { input, out } => cmd_morton_permute(&input, &out),
        Command::Progressive { cmd } => match cmd {
            ProgressiveCmd::Encode { input, output } => cmd_progressive_encode(&input, &output),
            ProgressiveCmd::Decode {
                input,
                output,
                partial_bytes,
            } => cmd_progressive_decode(&input, &output, partial_bytes),
            ProgressiveCmd::Info { input } => cmd_progressive_info(&input),
        },
    }
}

fn cmd_progressive_encode(input: &Path, output: &Path) -> Result<()> {
    let t0 = std::time::Instant::now();
    let in_bytes = std::fs::metadata(input)
        .with_context(|| format!("stat {}", input.display()))?
        .len();
    encode_progressive_file(input, output)
        .with_context(|| format!("encoding {} -> {}", input.display(), output.display()))?;
    let out_bytes = std::fs::metadata(output)?.len();
    let elapsed = t0.elapsed();
    println!(
        "progressive encode: {} ({} B) -> {} ({} B) in {:.2} s",
        input.display(),
        in_bytes,
        output.display(),
        out_bytes,
        elapsed.as_secs_f64()
    );
    Ok(())
}

fn cmd_progressive_decode(
    input: &Path,
    output: &Path,
    partial_bytes: Option<u64>,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    decode_progressive_file(input, output, partial_bytes).with_context(|| {
        format!(
            "decoding {} -> {} (partial_bytes={:?})",
            input.display(),
            output.display(),
            partial_bytes,
        )
    })?;
    let out_bytes = std::fs::metadata(output)?.len();
    let elapsed = t0.elapsed();
    println!(
        "progressive decode: {} -> {} ({} B) partial_bytes={:?} in {:.2} s",
        input.display(),
        output.display(),
        out_bytes,
        partial_bytes,
        elapsed.as_secs_f64(),
    );
    Ok(())
}

fn cmd_progressive_info(input: &Path) -> Result<()> {
    let bytes = std::fs::read(input).with_context(|| format!("reading {}", input.display()))?;
    let h = read_mgs2_header(&bytes)
        .map_err(|e| anyhow!("reading mgs2 header from {}: {e}", input.display()))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "file": input.display().to_string(),
            "file_bytes": bytes.len(),
            "version": h.version,
            "flags": h.flags,
            "n_splats": h.n_splats,
            "record_size": h.record_size,
            "ply_header_bytes": h.ply_header.len(),
            "payload_offset": h.payload_offset,
            "payload_len": h.payload_len,
        }))?
    );
    Ok(())
}

/// Reorder splats in `scene` by 16-bit-per-axis 3D Morton code of their
/// world-space position. Stable sort so equal Morton codes preserve input
/// order. Mutates in place.
fn morton_sort_scene(scene: &mut SplatScene) {
    if scene.splats.is_empty() {
        return;
    }
    // Compute bounding box. The IR doesn't carry a precomputed bbox, so we
    // do one pass here. f32::min/max are NaN-propagating; trained 3DGS
    // shouldn't have NaN positions but we clamp into [bbox_min, bbox_max]
    // below regardless, so a stray NaN folds to the corner.
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for s in &scene.splats {
        for i in 0..3 {
            if s.position[i] < mn[i] {
                mn[i] = s.position[i];
            }
            if s.position[i] > mx[i] {
                mx[i] = s.position[i];
            }
        }
    }
    let extent = [
        (mx[0] - mn[0]).max(f32::MIN_POSITIVE),
        (mx[1] - mn[1]).max(f32::MIN_POSITIVE),
        (mx[2] - mn[2]).max(f32::MIN_POSITIVE),
    ];
    // 16-bit-per-axis spread to 48 bits packed in u64. Classic mask-shift
    // sequence: 5 steps to spread 16 bits over 48 positions.
    fn spread16(v: u32) -> u64 {
        let mut x = (v & 0xFFFF) as u64;
        x = (x | (x << 32)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555;
        x
    }
    let n = scene.splats.len();
    let mut keyed: Vec<(u64, u32)> = Vec::with_capacity(n);
    for (idx, s) in scene.splats.iter().enumerate() {
        let qx = (((s.position[0] - mn[0]) / extent[0]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32;
        let qy = (((s.position[1] - mn[1]) / extent[1]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32;
        let qz = (((s.position[2] - mn[2]) / extent[2]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32;
        let qx = qx.min(65535);
        let qy = qy.min(65535);
        let qz = qz.min(65535);
        let code = spread16(qx) | (spread16(qy) << 1) | (spread16(qz) << 2);
        keyed.push((code, idx as u32));
    }
    keyed.sort_by_key(|&(k, _)| k);
    // Permute splats. Also permute semantic_labels if present so they
    // stay aligned. LOD indices reference the splat array; after this
    // permutation those indices would point at the wrong rows, so we
    // drop precomputed LODs (they'd need re-generation downstream).
    let perm: Vec<usize> = keyed.iter().map(|&(_, i)| i as usize).collect();
    let new_splats: Vec<Splat> = perm.iter().map(|&i| scene.splats[i].clone()).collect();
    scene.splats = new_splats;
    if let Some(labels) = scene.semantic_labels.take() {
        let new_labels: Vec<_> = perm.iter().map(|&i| labels[i].clone()).collect();
        scene.semantic_labels = Some(new_labels);
    }
    // Stale LODs would silently reference the wrong rows; clear them.
    scene.lods = None;
}

fn cmd_morton_permute(input: &Path, out: &Path) -> Result<()> {
    let (mut scene, fmt) = load_scene(input)?;
    let n = scene.len();
    let t0 = std::time::Instant::now();
    morton_sort_scene(&mut scene);
    let elapsed = t0.elapsed();
    // Output format: derive from extension.
    let out_fmt = format_from_extension(out)
        .ok_or_else(|| anyhow!("could not infer output format from {}", out.display()))?;
    if out_fmt != fmt {
        // Allowed in principle (load_scene returns IR), but for the morton-
        // permute use-case we want a same-format roundtrip. Permit the
        // conversion but warn.
        eprintln!(
            "splatforge: warning: input is {fmt}, output is {out_fmt} — \
             cross-format conversion proceeds but morton-permute is intended same-format"
        );
    }
    match out_fmt {
        "ply" => write_ply(&scene, out)?,
        "spz" => write_spz(out, &scene)?,
        "gltf" => write_gltf(&scene, out, &WriteOpts::default())?,
        "glb" => write_glb(&scene, out, &WriteOpts::default())?,
        "usda" => write_usda(&scene, out, &UsdWriteOpts::default())?,
        "usdc" => write_usdc(&scene, out, &UsdWriteOpts::default())?,
        other => return Err(anyhow!("unsupported output format: {other}")),
    }
    println!(
        "morton-permute: {} splats reordered in {:.1} ms -> {}",
        n,
        elapsed.as_secs_f64() * 1000.0,
        out.display()
    );
    Ok(())
}

fn detect_format(path: &Path) -> Result<&'static str> {
    if let Some(fmt) = format_from_extension(path) {
        return Ok(fmt);
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    format_from_magic(&bytes)
        .ok_or_else(|| anyhow!("could not detect format of {}", path.display()))
}

fn load_scene(path: &Path) -> Result<(SplatScene, &'static str)> {
    let fmt = detect_format(path)?;
    let scene = match fmt {
        "ply" => read_ply(path).with_context(|| format!("reading PLY {}", path.display()))?,
        "spz" => read_spz(path).with_context(|| format!("reading SPZ {}", path.display()))?,
        "gltf" => read_gltf(path).with_context(|| format!("reading glTF {}", path.display()))?,
        "glb" => read_glb(path).with_context(|| format!("reading GLB {}", path.display()))?,
        "usda" => read_usda(path).with_context(|| format!("reading USDA {}", path.display()))?,
        "usdc" => read_usdc(path).with_context(|| format!("reading USDC {}", path.display()))?,
        other => return Err(anyhow!("unsupported format: {other}")),
    };
    Ok((scene, fmt))
}

fn cmd_analyze(path: &Path, pretty: bool) -> Result<()> {
    let bytes = std::fs::metadata(path)?.len();
    let (scene, fmt) = load_scene(path)?;
    let report = AnalyzeReport::from_scene(&scene, fmt, bytes);
    println!("{}", report.to_json(pretty));
    Ok(())
}

fn cmd_inspect(path: &Path) -> Result<()> {
    let fmt = detect_format(path)?;
    match fmt {
        "gltf" => {
            let report = inspect_gltf(path)?;
            println!(
                "format=gltf splatCount={} chunks={} checksum={} sf_index={}",
                report.splat_count,
                report.chunk_count,
                if report.checksum_ok { "ok" } else { "fail" },
                report.has_spatial_index
            );
        }
        "glb" => {
            // GLB embeds JSON+BIN in a binary container; inspect_gltf assumes
            // a standalone JSON file, so fall back to scene loading.
            let (scene, _) = load_scene(path)?;
            println!("format=glb splatCount={}", scene.len());
        }
        other => {
            let (scene, fmt) = load_scene(path)?;
            println!("format={fmt} splats={} ({other})", scene.len());
        }
    }
    Ok(())
}

fn cmd_convert(input: &Path, to: &str, out: &Path) -> Result<()> {
    let (scene, _) = load_scene(input)?;
    match to {
        "ply" => {
            write_ply(&scene, out)?;
            Ok(())
        }
        "spz" => {
            write_spz(out, &scene)?;
            Ok(())
        }
        "gltf" => {
            write_gltf(&scene, out, &WriteOpts::default())?;
            Ok(())
        }
        "glb" => {
            write_glb(&scene, out, &WriteOpts::default())?;
            Ok(())
        }
        "usda" => {
            write_usda(&scene, out, &UsdWriteOpts::default())?;
            Ok(())
        }
        "usdc" => {
            write_usdc(&scene, out, &UsdWriteOpts::default())?;
            Ok(())
        }
        other => Err(anyhow!("unknown target format: {other}")),
    }
}

/// Write a single line in the machine-readable progress format. Workers parse
/// this with a simple `line.starts_with("PROGRESS ")` test, so the prefix and
/// `frac=` / `stage=` token names must stay stable. Stdout is line-buffered
/// in the Modal worker because we run with `bufsize=1`.
fn emit_progress(frac: f32, stage: &str) {
    let clamped = frac.clamp(0.0, 1.0);
    println!("PROGRESS frac={:.4} stage={}", clamped, stage);
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
}

#[allow(clippy::too_many_arguments)]
fn cmd_optimize(
    input: &Path,
    preset_name: &str,
    chunked: bool,
    compress: Option<&str>,
    target: Option<&str>,
    out: Option<&Path>,
    output_dir: Option<&Path>,
    progress: bool,
) -> Result<()> {
    if out.is_some() && output_dir.is_some() {
        return Err(anyhow!("--out and --output-dir are mutually exclusive"));
    }
    let target_glb = match target {
        None | Some("gltf") => false,
        Some("glb") => true,
        Some(other) => return Err(anyhow!("unknown --target {other:?} (want gltf or glb)")),
    };
    let compress_mode: Option<&str> = match compress {
        None => None,
        Some("zstd") => Some("zstd"),
        Some("spz") => Some("spz"),
        Some("none") => None,
        Some(other) => return Err(anyhow!("unknown --compress {other:?} (want zstd or spz)")),
    };
    if compress_mode == Some("spz") && !target_glb {
        return Err(anyhow!(
            "--compress spz requires --target glb (the extension is glb-embedded)"
        ));
    }
    if preset_name == "geospatial" && output_dir.is_none() {
        return Err(anyhow!(
            "preset 'geospatial' requires --output-dir <DIR> (the Cesium tileset is multi-file)"
        ));
    }
    // Hosted-only presets — the CLI knows the names (so users get pricing
    // and a useful error instead of "unknown preset") but the actual
    // encoder runs in the worker / Modal app. Wiring the worker through
    // here is a separate integration; until then, surface a clear
    // pending-integration message.
    // TODO(codec-gs-mixed): wire novel-3 mixed_crf.py encoder through
    // splatforge-private/apps/diff-repack so this preset runs end-to-end
    // from the CLI. Tracked: ship cull-default + codec-gs-mixed PR.
    if matches!(
        preset_name,
        "codec-gs-stacked" | "codec-gs-mixed" | "codec-gs-mixed-k5"
    ) {
        return Err(anyhow!(
            "preset '{preset_name}' is known but the worker integration is pending — \
             submit the job through the hosted API to use it"
        ));
    }
    if preset_name != "geospatial" && output_dir.is_some() {
        return Err(anyhow!(
            "--output-dir is only supported with --preset geospatial"
        ));
    }
    // The pipeline run is the longest single span; everything else is fast.
    // Reserve frac=0.00..0.90 for the optimize pipeline so we have headroom
    // for the post-write "encoding glTF" step (~0.90..0.98) and a final
    // "done" tick at 1.0. Without that headroom the bar would hit 100% and
    // then sit there while we write 30+ MB of buffer files.
    if progress {
        // Initial tick — surfaces "starting" to the UI immediately, before
        // any pass runs (PLY parse for a giant scene can take 5+ seconds).
        emit_progress(0.00, "loading-input");
    }
    let (mut scene, _) = load_scene(input)?;
    let pipe = preset(preset_name)?;
    let report = if progress {
        pipe.run_with_progress(&mut scene, |i, total, name| {
            // Map pass index to the [0.05, 0.90] band so the bar reaches
            // 5% by the time the first pass starts and 90% when the last
            // pass finishes ("done" emitted by run_with_progress lands at
            // i == total, frac = 0.90).
            let frac = if total == 0 {
                0.90
            } else {
                0.05 + (i as f32 / total as f32) * 0.85
            };
            emit_progress(frac, name);
        })?
    } else {
        pipe.run(&mut scene)?
    };
    // `geospatial` short-circuits to the tileset writer — no single .gltf out.
    if preset_name == "geospatial" {
        let dir = output_dir.expect("checked above");
        if progress {
            emit_progress(0.92, "writing-tileset");
        }
        let report_t = write_tileset(&scene, dir, &TilesetOpts::default())
            .with_context(|| format!("writing 3D Tiles tileset to {}", dir.display()))?;
        if progress {
            emit_progress(0.98, "writing-report");
        }
        let report_path = dir.join("optimize-report.json");
        std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
        println!(
            "optimized {} -> tileset {} ({} tiles, root error={:.4})",
            input.display(),
            report_t.tileset_json.display(),
            report_t.tiles.len(),
            report_t
                .tiles
                .first()
                .map(|t| t.geometric_error)
                .unwrap_or(0.0),
        );
        for t in &report_t.tiles {
            println!(
                "  lod{} fraction={:.4} splats={} geometricError={:.4} glb={}",
                t.lod_index, t.fraction, t.splat_count, t.geometric_error, t.glb,
            );
        }
        if progress {
            emit_progress(1.00, "done");
        }
        return Ok(());
    }
    let default_ext = if target_glb {
        "optimized.glb"
    } else {
        "optimized.gltf"
    };
    let out = out
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension(default_ext));
    // Per SPEC-0013, the web-targeted presets opt in to `KHR_mesh_quantization`
    // integer accessors so the glTF wire size lands close to the SPZ payload.
    // `lossless-repack` and `quality-max` keep f32 accessors so byte-identical
    // round-trips remain possible. SPZ already compresses; we keep quantize=false
    // on the SPZ path so the empty placeholder accessors stay FLOAT.
    let quantize = compress_mode != Some("spz")
        && matches!(
            preset_name,
            "web-mobile"
                | "web-desktop"
                | "quest-browser"
                | "visionos-preview"
                | "thumbnail-preview"
                | "size-min"
                // Hosted neural codec — quantized output; the neural pass
                // itself happens upstream (Modal A100 trainer in
                // splatforge-private/apps/diff-repack), this just encodes
                // the quantized splats into glTF.
                | "hosted-neural-outdoor"
                // MesonGS++ post-training codec presets — quantized
                // by the codec itself, glTF emission stays in int range.
                | "mgs-balanced"
                | "mgs-aggressive"
                // CodecGS presets — feature-plane + standard video codec
                // (HEVC / AV1). Splat attributes are quantized to fit
                // the codec's input range; glTF wrapper carries the
                // metadata pointing at the .hevc / .av1 payload.
                | "codec-gs"
                | "codec-gs-extreme"
                // Stacked: v0.1 neural codec training → CodecGS post-
                // process. A4.1 BUILT 2026-05-15 on bicycle:
                // 152× / 22.37 dB.
                | "codec-gs-stacked"
                // Mixed-CRF stacked: encodes the top-K% of splats by
                // importance (opacity × det(scale)^(2/3)) at CRF 14 and
                // the rest at CRF 28. novel-3 BUILT 2026-05-15:
                // K=2 → 151× / 25.2 dB, K=5 → 59× / 26.3 dB on bicycle.
                | "codec-gs-mixed"
                | "codec-gs-mixed-k5"
        );
    let compress_variant = if compress_mode == Some("spz") {
        Some(splatforge_gltf::SpzVariant::V2)
    } else {
        None
    };
    let opts = WriteOpts {
        chunked,
        chunk_target_splats: 100_000,
        lod_fractions: vec![1.0],
        quantize,
        quantize_rotation: false,
        spec_version: Default::default(),
        compress: compress_variant,
    };
    if progress {
        emit_progress(0.92, "encoding-gltf");
    }
    if target_glb {
        write_glb(&scene, &out, &opts)?;
    } else {
        write_gltf(&scene, &out, &opts)?;
    }
    if progress {
        emit_progress(0.98, "writing-report");
    }
    let report_path = out.with_extension("json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!(
        "optimized {} -> {} (report: {})",
        input.display(),
        out.display(),
        report_path.display()
    );

    if progress {
        emit_progress(1.00, "done");
    }
    if compress_mode == Some("zstd") {
        let (orig, comp) = compress_buffer_files(&out)?;
        let ratio = if comp > 0 {
            orig as f64 / comp as f64
        } else {
            0.0
        };
        println!(
            "compressed {} buffer bytes -> {} zstd bytes ({:.2}x smaller)",
            orig, comp, ratio
        );
    }
    Ok(())
}

/// Walk the glTF's parent directory + any `buffers/` subdir and produce a
/// `.zst` sibling for every `.bin` file. Returns (raw_total, zstd_total)
/// so the caller can print a ratio. Lossless: the original `.bin` files
/// stay in place so existing viewers keep working; only callers who know
/// to fetch the `.zst` version benefit from the smaller bytes (served
/// via HTTP `Content-Encoding: zstd`, modern browsers decode transparently).
fn compress_buffer_files(gltf_path: &Path) -> Result<(u64, u64)> {
    let parent = gltf_path
        .parent()
        .ok_or_else(|| anyhow!("glTF path has no parent: {}", gltf_path.display()))?;
    let mut total_raw: u64 = 0;
    let mut total_zst: u64 = 0;
    // Candidate roots: the glTF's own directory, plus the `buffers/` sibling
    // that `write_gltf` uses for chunked output. We walk each non-recursively
    // because a deeper layout isn't part of the writer's contract.
    let roots = [parent.to_path_buf(), parent.join("buffers")];
    for root in roots.iter().filter(|p| p.is_dir()) {
        for entry in
            std::fs::read_dir(root).with_context(|| format!("reading {}", root.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("bin") {
                continue;
            }
            let raw =
                std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            // Level 19 is zstd's near-max ratio. For ~100 MB of quantized
            // splat data, encoding is ~1-2 seconds — fine for an offline
            // optimize step. Drop to level 3 if benchmarking shows this
            // pegging the optimize wall time.
            let compressed = zstd::encode_all(raw.as_slice(), 19)
                .with_context(|| format!("zstd encode {}", path.display()))?;
            let zst_path = path.with_extension("bin.zst");
            std::fs::write(&zst_path, &compressed)
                .with_context(|| format!("writing {}", zst_path.display()))?;
            total_raw = total_raw.saturating_add(raw.len() as u64);
            total_zst = total_zst.saturating_add(compressed.len() as u64);
        }
    }
    Ok((total_raw, total_zst))
}

fn cmd_preview(input: &Path, port: u16) -> Result<()> {
    let bind = format!("0.0.0.0:{port}");
    let server =
        tiny_http::Server::http(&bind).map_err(|e| anyhow!("failed to bind {bind}: {e}"))?;
    let shell_path = Path::new("packages/viewer/preview-shell.html");
    let shell = std::fs::read_to_string(shell_path).unwrap_or_else(|_| {
        format!(
            "<!doctype html><meta charset=utf-8><title>SplatForge preview</title>\
             <h1>SplatForge preview placeholder</h1>\
             <p>Viewer shell not yet generated. ?src={}</p>",
            input.display()
        )
    });
    let src_path = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());
    println!(
        "serving preview on http://localhost:{port}/ (src={})",
        src_path.display()
    );
    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        if url == "/" || url.starts_with("/?") {
            let mut body = shell.clone();
            body = body.replace("{{SPLATFORGE_SRC}}", &src_path.display().to_string());
            let _ = request.respond(
                tiny_http::Response::from_string(body).with_header(
                    tiny_http::Header::from_bytes(
                        &b"Content-Type"[..],
                        &b"text/html; charset=utf-8"[..],
                    )
                    .unwrap(),
                ),
            );
            continue;
        }
        if url.starts_with("/splat") {
            // serve the splat bytes
            let bytes = std::fs::read(&src_path).unwrap_or_default();
            let _ = request.respond(tiny_http::Response::from_data(bytes));
            continue;
        }
        let mut buf = Vec::new();
        let _ = request.as_reader().read_to_end(&mut buf);
        let _ =
            request.respond(tiny_http::Response::from_string("not found").with_status_code(404));
    }
    Ok(())
}

fn cmd_diff(before: &Path, after: &Path, out: Option<&Path>, threshold: f32) -> Result<()> {
    use std::process::Command as ProcCommand;

    if !before.exists() {
        return Err(anyhow!("before file does not exist: {}", before.display()));
    }
    if !after.exists() {
        return Err(anyhow!("after file does not exist: {}", after.display()));
    }
    let out_dir = out
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("reports/diff"));
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;

    let helper = locate_helper()?;
    let cli_exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("splatforge"));

    let status = ProcCommand::new("node")
        .arg(&helper)
        .arg("--before")
        .arg(before)
        .arg("--after")
        .arg(after)
        .arg("--out")
        .arg(&out_dir)
        .arg("--threshold")
        .arg(threshold.to_string())
        .arg("--cli")
        .arg(&cli_exe)
        .status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(anyhow!("diff helper exited with code {:?}", s.code())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(anyhow!(
            "node not found in PATH. Install Node.js 20+ to use `splatforge diff`."
        )),
        Err(e) => Err(anyhow::Error::from(e).context("spawning diff helper")),
    }
}

/// Locate the Node.js helper script that drives `splatforge diff`.
///
/// Resolution order:
///   1. `$SPLATFORGE_DIFF_HELPER` if set (must exist).
///   2. `tests/visual/scripts/diff-cli.mjs` walking up from the binary.
///   3. `tests/visual/scripts/diff-cli.mjs` walking up from `$CWD`.
fn locate_helper() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("SPLATFORGE_DIFF_HELPER") {
        let path = PathBuf::from(p);
        if !path.exists() {
            return Err(anyhow!(
                "SPLATFORGE_DIFF_HELPER points at non-existent file: {}",
                path.display()
            ));
        }
        return Ok(path);
    }
    let rel = Path::new("tests/visual/scripts/diff-cli.mjs");
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.parent().into_iter().flat_map(|p| p.ancestors()) {
            let candidate = ancestor.join(rel);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            let candidate = ancestor.join(rel);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }
    Err(anyhow!(
        "diff helper not found. Set SPLATFORGE_DIFF_HELPER or run from the SplatForge repo root."
    ))
}

fn cmd_benchmark(path: &Path, _device_profile: Option<&str>) -> Result<()> {
    let start = std::time::Instant::now();
    let (scene, fmt) = load_scene(path)?;
    let parse_ms = start.elapsed().as_millis();
    let bytes = std::fs::metadata(path)?.len();
    let report = AnalyzeReport::from_scene(&scene, fmt, bytes);
    let total_ms = start.elapsed().as_millis();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "format": fmt,
            "splatCount": report.splat_count,
            "parseMs": parse_ms,
            "totalMs": total_ms,
        }))?
    );
    Ok(())
}

fn cmd_corpus_run(name: &str) -> Result<()> {
    let suite = match name {
        "smoke" => splatforge_bench::run_smoke()?,
        other => splatforge_bench::run_named(other, Path::new("fixtures"))?,
    };
    println!("{}", splatforge_bench::to_json(&suite)?);
    Ok(())
}

/* ---------- submit / fidelity / spec-check ---------- */

/// Default SplatForge Cloud endpoint. Override with `SPLATFORGE_API_URL`.
const DEFAULT_API_URL: &str = "https://splatforge-api.fly.dev";

fn cmd_submit(
    input: &Path,
    preset: &str,
    label: Option<&str>,
    webhook_url: Option<&str>,
    poll_secs: u64,
    timeout_secs: u64,
    no_wait: bool,
) -> Result<()> {
    use std::process::Command as Cmd;

    let api_url = std::env::var("SPLATFORGE_API_URL").unwrap_or_else(|_| DEFAULT_API_URL.into());
    let api_url = api_url.trim_end_matches('/').to_string();
    let api_key = std::env::var("SPLATFORGE_API_KEY").ok();

    if !input.exists() {
        anyhow::bail!("input not found: {}", input.display());
    }
    let size_bytes = std::fs::metadata(input)?.len();
    let filename = input
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("scene.bin");

    // Build the create-job request body.
    let mut body = serde_json::json!({
        "preset": preset,
        "filename": filename,
        "size_bytes": size_bytes,
    });
    if let Some(l) = label {
        body["label"] = serde_json::Value::String(l.into());
    }
    if let Some(w) = webhook_url {
        body["webhook_url"] = serde_json::Value::String(w.into());
    }

    let body_str = serde_json::to_string(&body)?;
    eprintln!("→ POST {api_url}/v1/jobs  ({size_bytes} bytes)");
    let mut args = vec![
        "-sS".to_string(),
        "-X".into(),
        "POST".into(),
        format!("{api_url}/v1/jobs"),
        "-H".into(),
        "content-type: application/json".into(),
        "-d".into(),
        body_str,
    ];
    if let Some(k) = api_key.as_deref() {
        args.push("-H".into());
        args.push(format!("Authorization: Bearer {k}"));
    }
    let create = Cmd::new("curl").args(&args).output()?;
    if !create.status.success() {
        anyhow::bail!(
            "POST /v1/jobs failed: {}",
            String::from_utf8_lossy(&create.stderr)
        );
    }
    let create_json: serde_json::Value = serde_json::from_slice(&create.stdout)?;
    let job_id = create_json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("create response missing id: {create_json}"))?
        .to_string();
    eprintln!("✓ job {job_id} created");

    // Upload bytes through the API's proxy endpoint.
    eprintln!("→ PUT  {api_url}/v1/jobs/{job_id}/upload");
    let mut upload_args = vec![
        "-sS".to_string(),
        "-X".into(),
        "POST".into(),
        format!("{api_url}/v1/jobs/{job_id}/upload"),
        "-H".into(),
        "content-type: application/octet-stream".into(),
        "--data-binary".into(),
        format!("@{}", input.display()),
    ];
    if let Some(k) = api_key.as_deref() {
        upload_args.push("-H".into());
        upload_args.push(format!("Authorization: Bearer {k}"));
    }
    let up = Cmd::new("curl").args(&upload_args).output()?;
    if !up.status.success() {
        anyhow::bail!("upload failed: {}", String::from_utf8_lossy(&up.stderr));
    }
    eprintln!("✓ upload complete");

    if no_wait {
        println!("{}", job_id);
        return Ok(());
    }

    // Poll until terminal.
    let deadline = if timeout_secs == 0 {
        None
    } else {
        Some(std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs))
    };
    loop {
        if let Some(d) = deadline {
            if std::time::Instant::now() > d {
                anyhow::bail!("polling deadline reached without terminal state (job {job_id})");
            }
        }
        let mut poll_args = vec!["-sS".to_string(), format!("{api_url}/v1/jobs/{job_id}")];
        if let Some(k) = api_key.as_deref() {
            poll_args.push("-H".into());
            poll_args.push(format!("Authorization: Bearer {k}"));
        }
        let poll = Cmd::new("curl").args(&poll_args).output()?;
        if !poll.status.success() {
            anyhow::bail!("poll failed: {}", String::from_utf8_lossy(&poll.stderr));
        }
        let pj: serde_json::Value = serde_json::from_slice(&poll.stdout)?;
        let status = pj
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        match status {
            "done" | "succeeded" => {
                let out_url = pj
                    .get("output_url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("done state but no output_url"))?;
                eprintln!("✓ done");
                println!("{}", out_url);
                return Ok(());
            }
            "error" | "failed" => {
                let err = pj
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("worker returned error with no message");
                anyhow::bail!("job failed: {err}");
            }
            other => {
                let phase = pj.get("phase").and_then(|v| v.as_str()).unwrap_or("");
                eprintln!("  status={other} phase={phase}");
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(poll_secs));
    }
}

fn cmd_fidelity(
    candidate: &Path,
    baseline: &Path,
    out: Option<&Path>,
    threshold: f32,
) -> Result<()> {
    // Today fidelity scoring is delegated to the diff harness, which
    // already does the 8-orbit deterministic render + ΔE94/SSIM/pixelmatch
    // through @splatforge/viewer. cmd_diff writes report.json to the
    // output dir; we forward the threshold for the pass/fail exit code.
    cmd_diff(baseline, candidate, out, threshold)
}

fn cmd_spec_check(input: &Path, spec: Option<&str>, json: bool) -> Result<()> {
    use std::process::Command as Cmd;

    if !input.exists() {
        anyhow::bail!("input not found: {}", input.display());
    }

    // Default: pick the validator from the file extension.
    let chosen = match spec {
        Some(s) => s.to_string(),
        None => match input
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("gltf") | Some("glb") => "khr_gaussian_splatting".to_string(),
            Some("usdc") | Some("usda") => "openusd_particle_field".to_string(),
            other => anyhow::bail!(
                "could not auto-detect spec for extension {:?}; pass --spec explicitly",
                other
            ),
        },
    };

    match chosen.as_str() {
        "khr_gaussian_splatting" => {
            // The validator binary lives in the workspace as
            // `splatforge-khr-validate`. We invoke it via Cargo's standard
            // binary-resolution: when this CLI was installed via
            // `cargo install splatforge-cli`, both binaries end up in the
            // same Cargo bin dir. For local dev, the user can override via
            // `SPLATFORGE_KHR_VALIDATE` to point at the workspace target.
            let validator = std::env::var("SPLATFORGE_KHR_VALIDATE")
                .unwrap_or_else(|_| "splatforge-khr-validate".to_string());
            let mut args: Vec<String> = vec![input.display().to_string()];
            if json {
                args.push("--json".into());
            }
            let status = Cmd::new(&validator).args(&args).status()?;
            if !status.success() {
                anyhow::bail!("{validator} returned non-zero ({status})");
            }
            Ok(())
        }
        "openusd_particle_field" => {
            // Mirrors the KHR branch: the validator binary lives in the
            // workspace as `splatforge-usd-validate` and ships alongside
            // this CLI under the same Cargo bin dir. For local dev,
            // `SPLATFORGE_USD_VALIDATE` overrides the lookup path.
            let validator = std::env::var("SPLATFORGE_USD_VALIDATE")
                .unwrap_or_else(|_| "splatforge-usd-validate".to_string());
            let mut args: Vec<String> = vec![input.display().to_string()];
            if json {
                args.push("--json".into());
            }
            let status = Cmd::new(&validator).args(&args).status()?;
            if !status.success() {
                anyhow::bail!("{validator} returned non-zero ({status})");
            }
            Ok(())
        }
        other => anyhow::bail!("unknown spec: {other}"),
    }
}

fn cmd_fidelity_score(candidate: &Path, baseline: Option<&Path>, pretty: bool) -> Result<()> {
    let report = splatforge_fidelity::score_ply(candidate, baseline)
        .with_context(|| "splatforge-fidelity v0.4")?;
    let json = if pretty {
        serde_json::to_string_pretty(&report)?
    } else {
        serde_json::to_string(&report)?
    };
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod diff_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn locate_helper_honors_env_var() {
        let dir =
            std::env::temp_dir().join(format!("splatforge-diff-helper-{}.mjs", std::process::id()));
        let mut f = std::fs::File::create(&dir).unwrap();
        writeln!(f, "// stub").unwrap();
        // SAFETY: tests in this crate are not run in parallel with anything
        // else mutating SPLATFORGE_DIFF_HELPER.
        std::env::set_var("SPLATFORGE_DIFF_HELPER", &dir);
        let resolved = locate_helper().expect("env override resolves");
        assert_eq!(resolved, dir);
        std::env::remove_var("SPLATFORGE_DIFF_HELPER");
        let _ = std::fs::remove_file(&dir);
    }
}
