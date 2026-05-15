#![deny(clippy::all)]
//! `splatforge` — the SplatForge command-line tool.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use splatforge_core::{format_from_extension, format_from_magic, AnalyzeReport, SplatScene};
use splatforge_gltf::{inspect_gltf, read_glb, read_gltf, write_glb, write_gltf, WriteOpts};
use splatforge_usd::{read_usda, read_usdc, write_usda, write_usdc, UsdWriteOpts};
use splatforge_optimize::{preset, write_tileset, TilesetOpts};
use splatforge_ply::{read_ply, write_ply};
use splatforge_spz::{read_spz, write_spz};

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
        /// After the regular write, also emit zstd-compressed `.bin.zst`
        /// versions of each buffer file. Halves the on-disk size of
        /// quantized presets at zero quality cost. Served via HTTP with
        /// `Content-Encoding: zstd` modern browsers decode transparently.
        #[arg(long)]
        compress: bool,
        /// Output path; defaults to "<input>.optimized.gltf".
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
}

#[derive(Subcommand, Debug)]
enum CorpusCmd {
    /// Run a named benchmark suite.
    Run {
        /// Suite name (e.g. "smoke").
        suite: String,
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
            out,
            output_dir,
            progress,
        } => cmd_optimize(
            &input,
            &preset,
            chunked,
            compress,
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
    }
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

fn cmd_optimize(
    input: &Path,
    preset_name: &str,
    chunked: bool,
    compress: bool,
    out: Option<&Path>,
    output_dir: Option<&Path>,
    progress: bool,
) -> Result<()> {
    if out.is_some() && output_dir.is_some() {
        return Err(anyhow!("--out and --output-dir are mutually exclusive"));
    }
    if preset_name == "geospatial" && output_dir.is_none() {
        return Err(anyhow!(
            "preset 'geospatial' requires --output-dir <DIR> (the Cesium tileset is multi-file)"
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
            report_t.tiles.first().map(|t| t.geometric_error).unwrap_or(0.0),
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
    let out = out
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("optimized.gltf"));
    // Per SPEC-0013, the web-targeted presets opt in to `KHR_mesh_quantization`
    // integer accessors so the glTF wire size lands close to the SPZ payload.
    // `lossless-repack` and `quality-max` keep f32 accessors so byte-identical
    // round-trips remain possible.
    let quantize = matches!(
        preset_name,
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
    if progress {
        emit_progress(0.92, "encoding-gltf");
    }
    write_gltf(&scene, &out, &opts)?;
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
    if compress {
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
        for entry in std::fs::read_dir(root)
            .with_context(|| format!("reading {}", root.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("bin") {
                continue;
            }
            let raw = std::fs::read(&path)
                .with_context(|| format!("reading {}", path.display()))?;
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
