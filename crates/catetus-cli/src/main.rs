#![deny(clippy::all)]
//! `catetus` — the Catetus command-line tool.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use catetus_core::{
    format_from_extension, format_from_magic, AnalyzeReport, Color, Splat, SplatScene,
};
use catetus_gltf::{
    inspect_gltf, read_glb, read_gltf, write_glb, write_gltf, DcQuantTable, RotationQuantTable,
    RotationSmallest3Table, ShRestPaletteRef, ShRestQuantTable, WriteOpts,
};
use catetus_optimize::{
    preset, take_last_dc_quant_table, take_last_delta_stream, take_last_rotation_quant_table,
    take_last_rotation_smallest3_table, take_last_sh_rest_palette, take_last_sh_rest_quant_table,
    write_tileset, RDPrune, TilesetOpts,
};
use catetus_ply::{
    decode_progressive_file, encode_progressive_file, read_mgs2_header, read_ply, write_ply,
};
use catetus_spz::{read_spz, write_spz};
use catetus_usd::{read_usda, read_usdc, write_usda, write_usdc, UsdWriteOpts};
use clap::{Parser, Subcommand};

mod license;
use license::{cmd_license_install, cmd_license_refresh, cmd_license_status, cmd_serve};

mod encode_api;
use encode_api::{
    apply_v5tail_via_api, resolve_api_url, run_encode_to_disk, EncodeTarget as ApiEncodeTarget,
};

#[derive(Parser, Debug)]
#[command(name = "catetus", version, about = "Gaussian Splat optimization CLI")]
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
        /// Output container: `gltf` (default), `glb`, or `sog`.
        ///   * `gltf` — external `.gltf` + `.bin` pair.
        ///   * `glb`  — self-contained binary glTF; required for `--compress
        ///     spz` and the `--lossless` byte-plane zstd wrap.
        ///   * `sog`  — PlayCanvas SOG container (`zip` of `meta.json` +
        ///     WebP attribute textures). Loads natively in SuperSplat
        ///     (https://superspl.at/editor) and our viewer-app. The same
        ///     `VQPaletteShRest` pass that the GLB path uses parks centroids
        ///     that ride through into `shN_centroids.webp`, so a SOG built
        ///     from a `wmv-vq45k4096-*` preset + `--jacobian-sidecar`
        ///     inherits the render-space Lloyd PSNR lift over PlayCanvas's
        ///     stock encoder. `--compress` / `--lossless` are rejected
        ///     under `--target sog` because the container has no GLB BIN
        ///     to wrap and SOG entries are WebP-compressed already.
        #[arg(long, value_name = "FORMAT")]
        target: Option<String>,
        /// Output path; defaults to "<input>.optimized.gltf" (or `.glb` /
        /// `.sog` depending on `--target`).
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        /// Output DIRECTORY for multi-tile presets (`geospatial`). The
        /// directory is created if missing and will contain `tileset.json`
        /// plus one `.glb` per LOD level. Mutually exclusive with `--out`.
        #[arg(long, value_name = "DIR")]
        output_dir: Option<PathBuf>,
        /// For `--target tileset`: write a SHARED-PALETTE tileset. Builds one
        /// scene-global SH-rest codebook (`palette.shpal` at the tileset root)
        /// and emits each tile's SH-rest as a tiny `.glb.shpalx` index sidecar
        /// instead of inlining FP32 SH coefficients per tile. The total payload
        /// is dramatically smaller (the SH-rest codebook is shared, not
        /// duplicated per tile) at equal octree/LOD config. No effect unless
        /// `--target tileset` is also set; ignored for DC-only scenes (those
        /// transparently fall back to the FP32 tile codec).
        #[arg(long)]
        shared_palette: bool,
        /// Lossless wrapper around the GLB BIN chunk. `brotli11` saves
        /// ~47% on the `quality-max` preset's FP32 SH coefficients and
        /// ~5-7% on quantized presets vs an uncompressed GLB. Decoder is a
        /// single-pass brotli stream over the BIN payload; viewers that
        /// don't understand `SF_brotli_buffer` will hard-fail the load,
        /// which is intentional. Only takes effect with `--target glb`.
        #[arg(long, value_name = "MODE")]
        lossless: Option<String>,
        /// Emit machine-readable `PROGRESS frac=<0..1> stage=<name>` lines
        /// to stdout before each pipeline pass, plus one terminal `frac=1.0`.
        /// Consumed by the Modal worker to stream live progress to the UI.
        /// Off by default so interactive CLI output stays clean.
        #[arg(long)]
        progress: bool,
        /// Replace the preset's `OpacityPrune` step with an `RDPrune`
        /// (rate-distortion prune at the given keep-rate). Experimental;
        /// see `experiments/w2-rd-prune/RESULT.md`. The argument is the
        /// fraction of splats to KEEP (e.g. `--rd-prune 0.7` keeps the
        /// top 70% by distortion proxy). Passing `0` disables.
        #[arg(long, value_name = "KEEP_RATIO")]
        rd_prune: Option<f32>,
        /// Path to a per-splat SH-rest rendering-Jacobian sidecar (.npz)
        /// produced by `experiments/jacobian-census-bonsai-30k/code/
        /// jacobian_census.py` or any compatible tool. The file must be a
        /// stored-mode (uncompressed) `.npz` archive containing a
        /// `J_sh_rest` array of shape `(N,)` and dtype `float32`, where
        /// `N` equals the splat count of the input file BEFORE any
        /// pipeline passes run (the loader keeps the weight array in
        /// lock-step with `RemoveInvalidSplats` and `MortonSort`, the
        /// only structural passes in the canonical-11 VQ45 presets).
        ///
        /// When set, `VQPaletteShRest` runs the render-space weighted
        /// Lloyd-Max algorithm from
        /// `experiments/render-space-lloyd-max/RESULT.md` (V1, +11.94 dB
        /// at the same byte budget on the controlled-baseline bench). No
        /// effect when the preset doesn't include `VQPaletteShRest`.
        #[arg(long, value_name = "NPZ_PATH")]
        jacobian_sidecar: Option<PathBuf>,
        /// Compute the per-splat SH-rest rendering-Jacobian internally
        /// (no external `.npz` required) and feed it into the same
        /// `VQPaletteShRest` render-space weighted Lloyd-Max path as
        /// `--jacobian-sidecar`. This is the **T2.1.R tier** — on bonsai
        /// the canonical `wmv-vq45-no-prune-tight` preset gains +6.24 dB
        /// over SuperSplat at the same byte budget (see
        /// `experiments/3tier-leaderboard/RESULT.md`).
        ///
        /// Uses the closed-form CPU proxy from `catetus-jacobian`
        /// (`α * area_2d(scale) * ||sh_rest||₂`); pure Rust, no GPU,
        /// no Python. Mutually exclusive with `--jacobian-sidecar`.
        #[arg(long, conflicts_with = "jacobian_sidecar")]
        auto_jacobian: bool,
        /// Emit a V5.2 joint-tail residual sidecar next to the output.
        ///
        /// * With `--target glb`: writes `<out>.v5tail`, stamps the GLB's
        ///   `SF_v5_tail_residual` root extension. Bonsai V5.2 prototype:
        ///   16.71 MB / 59.006 dB on the 72-view orbit (see
        ///   `experiments/v5-2-rust-port/RESULT.md`).
        /// * With `--target sog`: writes `<out>.v5tail` next to the SOG.
        ///   SuperSplat / vanilla SOG decoders ignore it; Catetus
        ///   readers (`@catetus/glb-polyfill`'s `decodeV5TailBytes`
        ///   + the viewer-app SOG loader) auto-apply it. Bonsai
        ///   prototype: +6.54 dB at +3.95% bytes vs vanilla SOG (see
        ///   `experiments/sog-v5tail-retune/RESULT.md`).
        ///
        /// The argument is the path to the ground-truth 3DGS PLY (NOT
        /// the post-optimize recon); the encoder computes per-attribute
        /// residuals between GT and the baseline-quantized reconstruction
        /// and writes them through the per-cell affine codec from
        /// `v5-1-sidecar-refinement`. Requires `--jacobian-sidecar` (the
        /// codec needs a per-splat joint-J score to pick the top-K
        /// selection).
        #[arg(long, value_name = "GT_PLY")]
        emit_v5_tail: Option<PathBuf>,
        /// Base URL for the hosted Catetus encode API. Used by
        /// `--target sog` (which short-circuits the local pipeline and
        /// POSTs the raw input PLY to `<api-url>/v1/encode?target=sog`).
        /// Defaults to `https://api.catetus.com` or the `CATETUS_API_URL`
        /// environment variable when set. Only consulted on the SOG path.
        #[arg(long, value_name = "URL")]
        api_url: Option<String>,
        /// Print which quality tier this scene qualifies for (based on its
        /// detected SH degree) and why, then exit WITHOUT writing any output.
        ///
        /// Tiers: SH degree >= 2 -> full quality tiers (`--auto-jacobian`
        /// T2.1.R, `--emit-v5-tail` V5.2 engage the VQ SH-rest palette
        /// meaningfully); SH == 1 -> partial (limited SH-rest budget, muted
        /// gains); SH == 0 -> SF baseline (no view-dependent color, quality
        /// tiers are no-ops) plus a recapture upsell. See
        /// `docs/ops/INGEST1_BLOCKER.md` / the MARKET-1 capture survey.
        #[arg(long)]
        explain_tier: bool,
    },
    /// Emit a V5.2 joint-tail residual sidecar (`.sog.v5tail`) next to an
    /// existing `.sog` file. The sidecar is byte-compatible with the
    /// `SFV51TAL` (variant=2) format the GLB path uses — legacy SOG readers
    /// (SuperSplat, plain viewer) ignore it; Catetus-aware readers add
    /// the residuals and gain ~5 dB. Pair this with the JS polyfill update
    /// in `@catetus/glb-polyfill` (`decodeSogV5Tail`) so the
    /// viewer-app's SOG loader picks the sidecar up automatically.
    SogEmitV5Tail {
        /// Path to the existing `.sog` (the encoder doesn't care WHO wrote
        /// it — works on anything `@playcanvas/splat-transform` produces).
        sog: PathBuf,
        /// Path to the ground-truth 3DGS PLY. Must have the same splat
        /// count as the SOG.
        #[arg(long, value_name = "GT_PLY")]
        gt: PathBuf,
        /// Top fraction of splats (0..1) to encode residuals for. Defaults
        /// to 0.01 (1%, matching the GLB-path V5.2 selection rule).
        #[arg(long, default_value_t = 0.01_f32)]
        k_percent: f32,
        /// Per-group bit-depth profile as `pos/rot/opa/sca/dc/shr`.
        /// Defaults to the V5.2 `8/10/12/12/8/8`. Use `sog-tight` for the
        /// retuned SOG profile (`8/8/8/10/10/12`).
        #[arg(long, value_name = "PROFILE")]
        profile: Option<String>,
        /// Optional per-splat Jacobian-sum NPZ (same format as the GLB
        /// `--jacobian-sidecar`). When set, top-K selection uses the
        /// joint-J score; otherwise falls back to the residual-L1 proxy.
        #[arg(long, value_name = "NPZ")]
        jacobian_sidecar: Option<PathBuf>,
        /// Optional JSON file to dump per-group residual stats
        /// (min/max/std/p99 / p99.9 per channel) BEFORE quantization.
        /// Used by the bit-depth retune experiment.
        #[arg(long, value_name = "JSON")]
        dump_residual_stats: Option<PathBuf>,
        /// Base URL for the hosted Catetus encode API. Defaults to
        /// `https://api.catetus.com` (override via `CATETUS_API_URL`).
        #[arg(long, value_name = "URL")]
        api_url: Option<String>,
    },
    /// Apply a `.sog.v5tail` sidecar to a `.sog` and write the
    /// reconstructed scene as a PLY. Used both for round-trip verification
    /// (the resulting PLY should be MUCH closer to the GT than the vanilla
    /// SOG decode) and for piping into a downstream PSNR bench.
    SogApplyV5Tail {
        /// The base SOG file.
        sog: PathBuf,
        /// The companion sidecar (defaults to `<sog>.v5tail`).
        #[arg(long, value_name = "SIDECAR_PATH")]
        sidecar: Option<PathBuf>,
        /// Output PLY path.
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Optional GT PLY — when set we print per-attribute L1 of
        /// (recon-vanilla vs GT) and (recon-with-sidecar vs GT) so the
        /// caller can see the residual reduction without spinning up a
        /// render-based bench.
        #[arg(long, value_name = "GT_PLY")]
        gt: Option<PathBuf>,
        /// Base URL for the hosted Catetus decode API. Defaults to
        /// `https://api.catetus.com` (override via `CATETUS_API_URL`).
        #[arg(long, value_name = "URL")]
        api_url: Option<String>,
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
    /// Upload a splat to Catetus Cloud, poll until done, return the URL.
    ///
    /// Reads `CATETUS_API_KEY` and `CATETUS_API_URL`
    /// (default `https://catetus-api.fly.dev`) from the environment.
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
    /// Score a single PLY with the fidelity MLP. Predict-only —
    /// the baseline PLY is optional (when omitted, the candidate is
    /// compared against the canonical lossless-repack identity profile
    /// baked into `catetus-fidelity`). Emits a JSON `ScoreReport`.
    ///
    /// `--mlp-version 0.4` (default) uses the original Bradley-Terry +
    /// bootstrap head. `--mlp-version 0.5` uses the real-PSNR-calibrated
    /// head — see `experiments/MLP_V05_RESULT.md` for held-out accuracy.
    FidelityScore {
        /// Candidate PLY.
        candidate: std::path::PathBuf,
        /// Optional baseline PLY.
        #[arg(long, short = 'b')]
        baseline: Option<std::path::PathBuf>,
        /// Which MLP head to use (`0.4` or `0.5`). Default: `0.4`.
        #[arg(long = "mlp-version", default_value = "0.4")]
        mlp_version: String,
        /// Pretty-print the JSON output.
        #[arg(long)]
        pretty: bool,
    },
    /// Manage the Pro on-prem license (install, status, refresh). The
    /// license file is `catetus.lic` — Ed25519-signed JSON minted by
    /// the Catetus API. Required to run `catetus serve`.
    License {
        #[command(subcommand)]
        cmd: LicenseCmd,
    },
    /// Run the on-prem Catetus Pro server. Reads the license at
    /// `~/.catetus/license.lic` (override with `--license`), verifies
    /// the embedded Ed25519 signature + offline-grace window, then binds
    /// the optimize/serve HTTP surface inside the customer's VPC.
    Serve {
        /// Bind address.
        #[arg(long, default_value = "0.0.0.0:8080")]
        bind: String,
        /// Path to the license file.
        #[arg(long)]
        license: Option<PathBuf>,
        /// Heartbeat endpoint base URL (the Catetus API). Heartbeats
        /// are skipped entirely when `CATETUS_NO_TELEMETRY=1` is set
        /// in the environment.
        #[arg(long, default_value = "https://catetus-api.fly.dev")]
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
    /// Build / inspect / unpack a LODGE-style hierarchical LOD pyramid.
    ///
    /// `catetus lodge build` is the offline chunker — it takes a
    /// trained 3DGS PLY and emits a directory containing a
    /// `manifest.json` plus per-level/per-chunk PLY files. The runtime
    /// viewer (Phase A.2/A.3, see `docs/perf/lodge-lod-spec.md`) consumes
    /// the manifest to stream only the LOD chunks visible to the current
    /// camera, dropping per-frame VRAM pressure dramatically on
    /// 10M-splat scenes.
    Lodge {
        #[command(subcommand)]
        cmd: LodgeCmd,
    },
    /// MesonGS++ post-training codec — encode a `.ply` to `.meson`.
    ///
    /// Wraps the standalone `mesonpp` binary's library API. CPU-only,
    /// hits ~23× on Mip-NeRF360-class scenes in the default config
    /// (`--preset mgs-balanced`).
    MesonppEncode {
        /// Input `.ply` (Inria 3DGS format).
        input: PathBuf,
        /// Output `.meson` container.
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Preset name — chooses K-means / xyz-bits / order-preservation
        /// trade-offs. `mgs-balanced` (default) targets 18-23× at near-
        /// lossless quality; `mgs-aggressive` tightens K for ~25×.
        #[arg(long, default_value = "mgs-balanced")]
        preset: String,
    },
    /// MesonGS++ post-training codec — decode a `.meson` to `.ply`.
    MesonppDecode {
        input: PathBuf,
        #[arg(long, short = 'o')]
        out: PathBuf,
    },
    /// Validate an asset against a Catetus-supported standard
    /// (KHR_gaussian_splatting today; OpenUSD when the USDC reader path
    /// is wired). Wraps `catetus-khr-validate` for the glTF case.
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
enum LodgeCmd {
    /// Build a LODGE pyramid from a trained 3DGS PLY.
    Build {
        /// Input PLY.
        #[arg(long, short = 'i')]
        input: PathBuf,
        /// Output directory. Will be created if missing. Contains the
        /// emitted `manifest.json` plus `level_<l>/chunk_<c>.ply`
        /// files.
        #[arg(long, short = 'o')]
        output: PathBuf,
        /// Number of LOD levels (including level 0 = original).
        #[arg(long, default_value_t = 6)]
        levels: usize,
        /// Target splat count at the coarsest level. The pyramid stops
        /// shrinking once a level falls below this.
        #[arg(long, default_value_t = 100_000)]
        target_top: usize,
        /// Geometric coarsening ratio between consecutive levels.
        #[arg(long, default_value_t = 2.0)]
        coarsen_ratio: f32,
        /// Target splat count per spatial chunk. Each level is split
        /// into ceil(level_count / chunk_target_splats) Morton-sorted
        /// chunks.
        #[arg(long, default_value_t = 100_000)]
        chunk_target_splats: usize,
    },
    /// Reassemble one level of a LODGE pyramid back into a single PLY.
    /// Useful for round-trip validation and for diagnostic exports.
    Unpack {
        /// Path to the LODGE output directory (contains `manifest.json`).
        #[arg(long, short = 'i')]
        input: PathBuf,
        /// Level to extract (0 = finest = original).
        #[arg(long, default_value_t = 0)]
        level: u32,
        /// Output PLY path.
        #[arg(long, short = 'o')]
        output: PathBuf,
    },
    /// Print a human-readable summary of a LODGE manifest.
    Info {
        /// Path to the LODGE output directory.
        #[arg(long, short = 'i')]
        input: PathBuf,
        /// Emit the full manifest as JSON instead of a summary.
        #[arg(long)]
        json: bool,
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
    /// Install a license file by copying it to `~/.catetus/license.lic`.
    /// Verifies the signature before installing — refuses to clobber the
    /// existing license if the new one is invalid.
    Install {
        /// Path to the `catetus.lic` to install.
        path: PathBuf,
    },
    /// Print the current license status (org, seats, plan, valid_until,
    /// last_refresh, offline grace remaining). Exits 0 when valid (with
    /// or without grace), 1 otherwise — so a customer cron can gate
    /// other automation on `catetus license status`.
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
        #[arg(long, default_value = "https://catetus-api.fly.dev")]
        api_base: String,
    },
}

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt::try_init();
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("catetus: error: {e:#}");
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
            shared_palette,
            lossless,
            progress,
            rd_prune,
            jacobian_sidecar,
            auto_jacobian,
            emit_v5_tail,
            api_url,
            explain_tier,
        } => cmd_optimize(
            &input,
            &preset,
            chunked,
            compress.as_deref(),
            target.as_deref(),
            out.as_deref(),
            output_dir.as_deref(),
            shared_palette,
            lossless.as_deref(),
            progress,
            rd_prune,
            jacobian_sidecar.as_deref(),
            auto_jacobian,
            emit_v5_tail.as_deref(),
            api_url.as_deref(),
            explain_tier,
        ),
        Command::SogEmitV5Tail {
            sog,
            gt,
            k_percent,
            profile,
            jacobian_sidecar,
            dump_residual_stats,
            api_url,
        } => cmd_sog_emit_v5tail(
            &sog,
            &gt,
            k_percent,
            profile.as_deref(),
            jacobian_sidecar.as_deref(),
            dump_residual_stats.as_deref(),
            api_url.as_deref(),
        ),
        Command::SogApplyV5Tail {
            sog,
            sidecar,
            out,
            gt,
            api_url,
        } => cmd_sog_apply_v5tail(
            &sog,
            sidecar.as_deref(),
            &out,
            gt.as_deref(),
            api_url.as_deref(),
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
            mlp_version,
            pretty,
        } => cmd_fidelity_score(&candidate, baseline.as_deref(), &mlp_version, pretty),
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
        Command::Lodge { cmd } => match cmd {
            LodgeCmd::Build {
                input,
                output,
                levels,
                target_top,
                coarsen_ratio,
                chunk_target_splats,
            } => cmd_lodge_build(
                &input,
                &output,
                levels,
                target_top,
                coarsen_ratio,
                chunk_target_splats,
            ),
            LodgeCmd::Unpack {
                input,
                level,
                output,
            } => cmd_lodge_unpack(&input, level, &output),
            LodgeCmd::Info { input, json } => cmd_lodge_info(&input, json),
        },
        Command::MesonppEncode { input, out, preset } => cmd_mesonpp_encode(&input, &out, &preset),
        Command::MesonppDecode { input, out } => cmd_mesonpp_decode(&input, &out),
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

fn cmd_progressive_decode(input: &Path, output: &Path, partial_bytes: Option<u64>) -> Result<()> {
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

fn cmd_lodge_build(
    input: &Path,
    output: &Path,
    levels: usize,
    target_top: usize,
    coarsen_ratio: f32,
    chunk_target_splats: usize,
) -> Result<()> {
    let opts = catetus_lodge::BuildOpts {
        levels,
        target_top,
        coarsen_ratio,
        chunk_target_splats,
    };
    let t0 = std::time::Instant::now();
    let manifest = catetus_lodge::build_from_ply(input, output, &opts)
        .with_context(|| format!("building LODGE from {}", input.display()))?;
    let elapsed = t0.elapsed();
    let bytes = catetus_lodge::total_on_disk_bytes(output).unwrap_or(0);
    println!(
        "lodge build: {} ({} splats) -> {} ({} levels, {:.2} MiB on disk, {:.1}s)",
        input.display(),
        manifest.original_splat_count,
        output.display(),
        manifest.levels.len(),
        bytes as f64 / (1024.0 * 1024.0),
        elapsed.as_secs_f64(),
    );
    for lvl in &manifest.levels {
        println!(
            "  level {} : {:>9} splats ({:.4}× of L0), {} chunks, d_l={:.3}",
            lvl.level,
            lvl.splat_count,
            lvl.reduction,
            lvl.chunks.len(),
            lvl.depth_threshold,
        );
    }
    Ok(())
}

fn cmd_lodge_unpack(input: &Path, level: u32, output: &Path) -> Result<()> {
    let n = catetus_lodge::unpack_level_to_ply(input, level, output)
        .with_context(|| format!("unpacking level {level} from {}", input.display()))?;
    println!(
        "lodge unpack: level {} ({} splats) -> {}",
        level,
        n,
        output.display()
    );
    Ok(())
}

fn cmd_lodge_info(input: &Path, json: bool) -> Result<()> {
    let manifest = catetus_lodge::read_manifest(input)
        .with_context(|| format!("reading manifest from {}", input.display()))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    println!(
        "source={} splats={} levels={} bbox=[{:?}..{:?}]",
        manifest.source,
        manifest.original_splat_count,
        manifest.levels.len(),
        manifest.bbox[0],
        manifest.bbox[1],
    );
    for lvl in &manifest.levels {
        let chunk_splats: usize = lvl.chunks.iter().map(|c| c.splat_count).sum();
        println!(
            "  level {} : {} splats ({:.4}× of L0), {} chunks ({} splats summed), d_l={:.3}",
            lvl.level,
            lvl.splat_count,
            lvl.reduction,
            lvl.chunks.len(),
            chunk_splats,
            lvl.depth_threshold,
        );
    }
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
            "catetus: warning: input is {fmt}, output is {out_fmt} — \
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

/// MesonGS++ encode wrapper. Maps the public preset names onto
/// `catetus_meson::EncodeConfig`.
fn cmd_mesonpp_encode(input: &Path, out: &Path, preset_name: &str) -> Result<()> {
    let cfg = match preset_name {
        // Production default. Targets ~18-23× on Mip-NeRF360 scenes.
        // Default-EncodeConfig (K=256 for all groups, 14-bit xyz, no
        // perm) is what ships as the worker preset.
        "mgs-balanced" => catetus_meson::EncodeConfig::default(),
        // Aggressive — K=128 for low groups, 12 xyz bits. Cuts another
        // ~15 % off the file at the cost of 0.1 dB.
        "mgs-aggressive" => catetus_meson::EncodeConfig {
            kmeans_k_low: 128,
            kmeans_k_color: 256,
            xyz_bits: 12,
            kmeans_iters: 10,
            seed: 0xC0FFEE,
            preserve_order: false,
        },
        // High-fidelity — keeps the input ordering, K=256 everywhere.
        // For pipelines downstream of a Morton-sensitive consumer.
        "mgs-preserve-order" => catetus_meson::EncodeConfig {
            preserve_order: true,
            ..catetus_meson::EncodeConfig::default()
        },
        other => {
            return Err(anyhow!(
                "unknown mesonpp preset: {other}. valid: mgs-balanced, mgs-aggressive, mgs-preserve-order"
            ))
        }
    };
    let scene = read_ply(input).with_context(|| format!("reading PLY {}", input.display()))?;
    let bytes = catetus_meson::encode_scene(&scene, &cfg)
        .with_context(|| format!("mesonpp encoding {}", input.display()))?;
    std::fs::write(out, &bytes).with_context(|| format!("writing {}", out.display()))?;
    println!(
        "mesonpp encoded {} splats -> {} ({} bytes, preset={})",
        scene.splats.len(),
        out.display(),
        bytes.len(),
        preset_name,
    );
    Ok(())
}

/// MesonGS++ decode wrapper.
fn cmd_mesonpp_decode(input: &Path, out: &Path) -> Result<()> {
    let bytes = std::fs::read(input).with_context(|| format!("reading {}", input.display()))?;
    let scene = catetus_meson::decode_scene(&bytes)
        .with_context(|| format!("mesonpp decoding {}", input.display()))?;
    write_ply(&scene, out).with_context(|| format!("writing PLY {}", out.display()))?;
    println!(
        "mesonpp decoded {} splats -> {}",
        scene.splats.len(),
        out.display(),
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

/// Minimal `.npz` reader for stored-mode (uncompressed) NumPy archives.
///
/// The format is just a ZIP file whose entries are NPY arrays. We only need
/// to support `compression_method = 0` (stored / no compression), which is
/// what `numpy.savez` (uncompressed; NOT `savez_compressed`) emits by
/// default. Each NPY entry has a small ASCII header dict describing dtype,
/// shape, and fortran-order, followed by the raw little-endian array bytes.
///
/// This avoids pulling in `zip` + `npyz` for what is currently a single
/// CLI flag (`--jacobian-sidecar`). If we end up needing compressed NPZ
/// or BIG-endian arrays in the future, swap to `npyz` + `zip`.
fn load_jacobian_sh_rest_from_npz(path: &Path) -> Result<Vec<f32>> {
    let entries = list_npz_entries(path)?;
    load_npz_entry(path, "J_sh_rest.npy", &entries).with_context(|| {
        format!(
            "NPZ {} does not contain a `J_sh_rest.npy` entry (available: {})",
            path.display(),
            entries
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        )
    })
}

/// Locate every entry in a stored-mode `.npz`. Returns (name, [data_off,
/// data_end)) pairs in archive order. Used by the multi-array Jacobian
/// loader which has to scan for several known array names without
/// re-reading the file.
struct NpzEntry {
    name: String,
    data_off: usize,
    data_end: usize,
}

fn list_npz_entries(path: &Path) -> Result<Vec<NpzEntry>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading NPZ {}", path.display()))?;
    let mut out: Vec<NpzEntry> = Vec::new();
    let mut cursor: usize = 0;
    while cursor + 30 <= bytes.len() {
        let sig = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        if sig != 0x0403_4b50 {
            break;
        }
        let compression = u16::from_le_bytes(bytes[cursor + 8..cursor + 10].try_into().unwrap());
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
                "NPZ entry uses compression method {compression}; \
                 only stored-mode (0) NPZ archives are supported. Re-save with \
                 `numpy.savez` (NOT `savez_compressed`)."
            ));
        }
        if csize != usize_ {
            return Err(anyhow!(
                "stored-mode NPZ entry size mismatch: compressed={csize} uncompressed={usize_}"
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

/// Pull one named entry out of a previously-enumerated NPZ and parse it as a
/// rank-1 `<f4` array. Re-reads the file (acceptable for the few-array case).
fn load_npz_entry(path: &Path, name: &str, entries: &[NpzEntry]) -> Result<Vec<f32>> {
    let entry = entries
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| anyhow!("entry `{name}` not in NPZ {}", path.display()))?;
    let bytes =
        std::fs::read(path).with_context(|| format!("re-reading NPZ {}", path.display()))?;
    parse_npy_f32_1d(&bytes[entry.data_off..entry.data_end])
        .with_context(|| format!("parsing NPZ entry `{name}`"))
}

/// Multi-array Jacobian loader for the V5.2 joint-tail codec. Scans the
/// `.npz` for the per-attribute J arrays the oracle-joint-tail-sf-baseline
/// experiment produced (`J_position`, `J_rotation`, `J_opacity`, `J_scale`,
/// `J_dc`, `J_sh_rest`). For each present array we normalise by its max
/// and sum the normalised contributions: `J_joint_sum[i] = Σ_c J_c[i] /
/// max(J_c)`. Returns the joint-sum array. Mirrors the formula the Python
/// prototype used to pick the top-1% selection that ships in
/// `experiments/oracle-joint-tail-sf-baseline/data/J_joint_sum.npy`.
///
/// Auto-detects which arrays are present so a single-array sidecar (legacy
/// T2.1.R) is rejected with a clear error pointing the user at the
/// multi-array file from `jacobian-census-canonical-11`.
fn load_joint_jacobian_sum_from_npz(path: &Path) -> Result<Vec<f32>> {
    let entries = list_npz_entries(path)?;
    // Try the canonical 6-array layout first; fall back to whichever arrays
    // are present. The oracle baseline's `J_joint_sum.npy` is also accepted
    // as a precomputed shortcut.
    if entries.iter().any(|e| e.name == "J_joint_sum.npy") {
        return load_npz_entry(path, "J_joint_sum.npy", &entries)
            .context("loading precomputed J_joint_sum array");
    }
    let candidates = [
        "J_position.npy",
        "J_rotation.npy",
        "J_opacity.npy",
        "J_scale.npy",
        "J_dc.npy",
        "J_sh_rest.npy",
    ];
    let mut found: Vec<(String, Vec<f32>)> = Vec::new();
    for &c in &candidates {
        if entries.iter().any(|e| e.name == c) {
            let v = load_npz_entry(path, c, &entries)?;
            found.push((c.to_string(), v));
        }
    }
    if found.is_empty() {
        return Err(anyhow!(
            "NPZ {} contains no recognised joint-Jacobian arrays. Expected one or more of {} \
             (or a precomputed J_joint_sum.npy). Found: {}",
            path.display(),
            candidates.join(", "),
            entries
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    // Sanity-check all the loaded arrays agree on length.
    let n = found[0].1.len();
    for (name, v) in &found {
        if v.len() != n {
            return Err(anyhow!(
                "joint-J arrays disagree on length: {} has {}, expected {}",
                name,
                v.len(),
                n
            ));
        }
    }
    let mut joint = vec![0.0f32; n];
    let mut joint_arrays: Vec<&str> = Vec::with_capacity(found.len());
    for (name, v) in &found {
        let mx = v.iter().cloned().fold(0.0f32, f32::max);
        if mx <= 0.0 {
            // Skip degenerate arrays — they don't contribute to the ranking.
            continue;
        }
        let inv = 1.0 / mx;
        for i in 0..n {
            joint[i] += v[i] * inv;
        }
        joint_arrays.push(name.as_str());
    }
    println!(
        "joint-Jacobian sum built from {} arrays ({})",
        joint_arrays.len(),
        joint_arrays.join(", ")
    );
    Ok(joint)
}

/// Parse a NumPy NPY (v1.0 / v2.0) array of dtype `<f4` and rank 1.
/// Returns the array as a `Vec<f32>`. Only the dtype / shape / fortran-order
/// fields of the header dict are interpreted; everything else is ignored.
fn parse_npy_f32_1d(blob: &[u8]) -> Result<Vec<f32>> {
    use std::io::Read;
    if blob.len() < 10 || &blob[0..6] != b"\x93NUMPY" {
        return Err(anyhow!("not an NPY file (bad magic)"));
    }
    let major = blob[6];
    let _minor = blob[7];
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
    // Very crude dict parsing — the header is a Python repr dict like
    // `{'descr': '<f4', 'fortran_order': False, 'shape': (1244819,), }`.
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
    let shape_raw = field(header, "shape").ok_or_else(|| anyhow!("NPY header missing 'shape'"))?;
    // shape looks like `(1244819,)` possibly followed by `, }` etc.
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
            "NPY data range exceeds file: need {bytes_needed} bytes from offset {data_off}, \
             have {}",
            blob.len() - data_off
        ));
    }
    let mut out = Vec::with_capacity(n);
    let mut rdr = &blob[data_off..data_off + bytes_needed];
    let mut buf = [0u8; 4];
    for _ in 0..n {
        rdr.read_exact(&mut buf)
            .map_err(|e| anyhow!("NPY read error: {e}"))?;
        out.push(f32::from_le_bytes(buf));
    }
    Ok(out)
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

/// Emit the SH-degree tier-routing report to stderr, plus a no-op WARNING for
/// any quality flag (`--auto-jacobian` / `--emit-v5-tail`) requested on input
/// whose SH degree gives the quality tiers nothing to re-encode.
///
/// This converts the SH=0 weakness into an honest product signal: SH>=2 routes
/// to the full quality tiers, SH==1 is partial, SH==0 gets the SF-baseline note
/// plus the recapture upsell. Without this, an SH=0 capture run with
/// `--auto-jacobian` silently produces output byte-identical to the SF baseline
/// (the KORIYAMA-1 confusion).
fn report_tier(
    decision: &catetus_core::TierDecision,
    auto_jacobian: bool,
    emit_v5_tail: bool,
) {
    use catetus_core::Tier;
    eprintln!(
        "catetus optimize: tier = {} (SH degree {})",
        decision.tier.name(),
        decision.degree_str()
    );
    eprintln!("catetus optimize: {}", decision.reason);
    match decision.tier {
        Tier::Baseline => {
            eprintln!(
                "catetus optimize: NOTE — the SF baseline is the right choice for this capture."
            );
            eprintln!("catetus optimize: {}", catetus_core::RECAPTURE_UPSELL);
        }
        Tier::Partial => {
            eprintln!(
                "catetus optimize: NOTE — quality-tier gains are muted on SH degree 1 input."
            );
        }
        Tier::Full => {}
    }
    if auto_jacobian && !decision.auto_jacobian_effective {
        eprintln!(
            "catetus optimize: WARNING — --auto-jacobian (T2.1.R) has NO EFFECT on this input \
             (no SH-rest coefficients to re-encode); output will match the SF baseline. \
             Recapture at SH=3 to use this tier."
        );
    }
    if emit_v5_tail && !decision.v5_tail_effective {
        eprintln!(
            "catetus optimize: WARNING — --emit-v5-tail (V5.2) has NO EFFECT on this input \
             (no SH-rest coefficients for the VQ palette); the sidecar adds no fidelity."
        );
    }
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
    shared_palette: bool,
    lossless: Option<&str>,
    progress: bool,
    rd_prune: Option<f32>,
    jacobian_sidecar: Option<&Path>,
    auto_jacobian: bool,
    emit_v5_tail: Option<&Path>,
    api_url: Option<&str>,
    explain_tier: bool,
) -> Result<()> {
    if out.is_some() && output_dir.is_some() {
        return Err(anyhow!("--out and --output-dir are mutually exclusive"));
    }
    // SH-degree tier routing. Detect the scene's SH degree on ingest and tell
    // the user which quality tier they qualify for (and warn when the quality
    // flags will be no-ops). We load the scene here for the report; the local
    // optimize path below re-loads it (cheap relative to the pipeline, and the
    // hosted `--target sog` path never loads the scene at all, so a one-time
    // load for the report is the simplest correct wiring). `--explain-tier` is
    // a dry-run that prints the report and exits without writing output.
    if explain_tier || auto_jacobian || emit_v5_tail.is_some() {
        match load_scene(input) {
            Ok((scene, _)) => {
                let decision = catetus_core::route_scene(&scene);
                report_tier(&decision, auto_jacobian, emit_v5_tail.is_some());
                if explain_tier {
                    return Ok(());
                }
            }
            Err(e) => {
                if explain_tier {
                    return Err(e).context("loading input for --explain-tier");
                }
                // Non-explain path: don't fail the whole run on a routing-only
                // load error; the main pipeline below will surface it properly.
                eprintln!("catetus optimize: note: could not detect SH tier ({e:#})");
            }
        }
    }
    // `--target sog` is hosted-only since the 2026-05-19 open-core split:
    // the SOG encoder lives in the private `catetus-sog` crate. Short-
    // circuit here BEFORE running the local optimize pipeline (the
    // pipeline would only produce an in-memory SplatScene that the SOG
    // writer would then need to serialize — and the SOG writer is the
    // part that's gone). Instead we POST the raw input PLY to the
    // hosted `/v1/encode` route and let the private worker run the
    // equivalent pipeline.
    if matches!(target, Some("sog")) {
        if compress.is_some() {
            return Err(anyhow!(
                "--compress is not supported with --target sog \
                 (SOG entries are WebP-compressed already)"
            ));
        }
        if lossless.is_some() {
            return Err(anyhow!(
                "--lossless is not supported with --target sog \
                 (no GLB BIN chunk to wrap)"
            ));
        }
        if output_dir.is_some() {
            return Err(anyhow!(
                "--output-dir is only supported with --preset geospatial"
            ));
        }
        let out_path = out
            .map(PathBuf::from)
            .unwrap_or_else(|| input.with_extension("optimized.sog"));
        let v5tail = emit_v5_tail.is_some();
        // Default the sidecar to `<out>.v5tail` so a user invoking
        // `--emit-v5-tail` doesn't need a separate flag to pick the
        // output path. Mirrors the GLB branch's behaviour.
        let sidecar_path = if v5tail {
            let mut p = out_path.clone();
            let new_ext = match p.extension().and_then(|s| s.to_str()) {
                Some(ext) => format!("{ext}.v5tail"),
                None => "v5tail".to_string(),
            };
            p.set_extension(new_ext);
            Some(p)
        } else {
            None
        };
        if progress {
            emit_progress(0.00, "hosted-sog-encode-upload");
        }
        let resolved_api = resolve_api_url(api_url);
        // Pipeline knobs (--preset, --rd-prune, --jacobian-sidecar) are
        // driven server-side; the hosted route uses its own default
        // profile. Surface a one-line note when a local user supplied
        // non-default knobs so they don't quietly assume those flags
        // ride through.
        if jacobian_sidecar.is_some() || rd_prune.is_some() || preset_name != "web-mobile" {
            eprintln!(
                "catetus: note: --target sog runs on the hosted encoder; \
                 --preset / --rd-prune / --jacobian-sidecar are interpreted \
                 by the server's default profile, not the local flags. \
                 Override the API base with --api-url or CATETUS_API_URL."
            );
        }
        let outcome = run_encode_to_disk(
            &resolved_api,
            ApiEncodeTarget::Sog,
            v5tail,
            input,
            &out_path,
            sidecar_path.as_deref(),
            None,
            std::time::Duration::from_secs(300),
        )?;
        if progress {
            emit_progress(1.00, "hosted-sog-encode-done");
        }
        let sidecar_msg =
            match (v5tail, sidecar_path.as_ref(), outcome.sidecar_bytes.as_ref()) {
                (true, Some(p), Some(b)) => {
                    format!(" + sidecar -> {} ({} bytes)", p.display(), b.len())
                }
                (true, _, _) => " (sidecar requested but server returned none)".to_string(),
                _ => String::new(),
            };
        println!(
            "hosted-sog encode: job_id={} -> {} ({} bytes){}",
            outcome.job_id,
            out_path.display(),
            outcome.output_bytes.len(),
            sidecar_msg,
        );
        let _ = (chunked, auto_jacobian);
        return Ok(());
    }
    let target_glb = match target {
        None | Some("gltf") => false,
        Some("glb") => true,
        // SOG is a separate container; we short-circuit the GLB writer
        // entirely below. For the early flag-validation stage we treat it
        // as "not a GLB" and let the dispatch at write-time decide.
        Some("sog") => false,
        // `tileset` is a multi-file octree LOD output written via
        // `catetus-tileset`. Each tile is itself an SF GLB, so we treat it as
        // "not a single GLB" here and short-circuit to the tileset writer after
        // the optimize pipeline runs (see the `target_tileset` block below).
        Some("tileset") => false,
        Some(other) => {
            return Err(anyhow!(
                "unknown --target {other:?} (want gltf, glb, sog, or tileset)"
            ))
        }
    };
    let target_sog = matches!(target, Some("sog"));
    let target_tileset = matches!(target, Some("tileset"));
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
    let lossless_wrap: Option<catetus_gltf::LosslessWrap> = match lossless {
        None | Some("none") => None,
        Some("brotli11") => Some(catetus_gltf::LosslessWrap::Brotli11),
        Some("zstd19-split") | Some("zstd19split") => Some(catetus_gltf::LosslessWrap::Zstd19Split),
        Some(other) => {
            return Err(anyhow!(
                "unknown --lossless {other:?} (want brotli11, zstd19-split, or none)"
            ))
        }
    };
    if lossless_wrap.is_some() && !target_glb {
        return Err(anyhow!(
            "--lossless requires --target glb (the wrapper sits on the GLB BIN chunk)"
        ));
    }
    if lossless_wrap.is_some() && compress_mode == Some("spz") {
        return Err(anyhow!(
            "--lossless is incompatible with --compress spz (SPZ already compresses)"
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
    // catetus-private/apps/diff-repack so this preset runs end-to-end
    // from the CLI. Tracked: ship cull-default + codec-gs-mixed PR.
    if matches!(
        preset_name,
        "codec-gs-stacked"
            | "codec-gs-mixed"
            | "codec-gs-mixed-k5"
            // `fcgs-instant` runs end-to-end on a Modal A100 via a private
            // Modal app; there's no CPU fallback locally. Surface a clear
            // pending message so users invoking the local CLI don't burn
            // time waiting for an encoder that isn't here.
            | "fcgs-instant"
    ) {
        return Err(anyhow!(
            "preset '{preset_name}' is known but the worker integration is pending — \
             submit the job through the hosted API to use it"
        ));
    }
    if preset_name != "geospatial" && !target_tileset && output_dir.is_some() {
        return Err(anyhow!(
            "--output-dir is only supported with --preset geospatial or --target tileset"
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
    let mut pipe = preset(preset_name)?;
    // Optional: swap the preset's OpacityPrune for an RDPrune at the
    // requested keep-rate. We replace in place to preserve pass order
    // (RDPrune runs at the same pipeline position as OpacityPrune would,
    // so downstream FloaterPrune/quant/sort stay unchanged).
    if let Some(ratio) = rd_prune {
        if ratio > 0.0 {
            let mut swapped = false;
            for slot in pipe.passes.iter_mut() {
                if slot.name() == "OpacityPrune" {
                    *slot = Box::new(RDPrune {
                        target_ratio: ratio,
                    });
                    swapped = true;
                    break;
                }
            }
            if !swapped {
                // Preset didn't include OpacityPrune (e.g. quality-max).
                // Splice RDPrune in after RemoveInvalidSplats so the
                // distortion proxy sees finite-only splats.
                let insert_at = pipe
                    .passes
                    .iter()
                    .position(|p| p.name() == "RemoveInvalidSplats")
                    .map(|i| i + 1)
                    .unwrap_or(0);
                pipe.passes.insert(
                    insert_at,
                    Box::new(RDPrune {
                        target_ratio: ratio,
                    }),
                );
            }
        }
    }
    // Optional per-splat SH-rest rendering-Jacobian sidecar. When present we
    // ingest a Jacobian array from the `.npz`, sanity-check its length
    // against the just-loaded scene, and seed `PassContext.sh_rest_weights`.
    // Downstream `VQPaletteShRest` switches to the render-space weighted
    // Lloyd-Max algorithm (V1 from `experiments/render-space-lloyd-max/
    // RESULT.md`, +11.94 dB at the same byte budget on the controlled bench).
    //
    // Selection rule: when `--emit-v5-tail` is also set we auto-detect the
    // multi-array layout from `jacobian-census-canonical-11` and use the
    // joint-J sum (Σ_c J_c / max(J_c)) — both as VQPaletteShRest weights
    // AND as the V5.2 sidecar's top-K selection score. When `--emit-v5-tail`
    // is not set, the legacy single-array `J_sh_rest` is used to preserve
    // exact byte-stable behaviour for the T2.1.R presets.
    let mut init_ctx = catetus_optimize::PassContext::default();
    let mut joint_jacobian: Option<Vec<f32>> = None;
    if auto_jacobian {
        if emit_v5_tail.is_some() {
            return Err(anyhow!(
                "--auto-jacobian does not yet support --emit-v5-tail (V5.2 needs a joint \
                 multi-array Jacobian — the CPU proxy currently emits only J_sh_rest). Use \
                 --jacobian-sidecar pointing at a joint .npz for V5.2 emission."
            ));
        }
        // Closed-form CPU proxy from `catetus-jacobian` (Apache-2.0). This is
        // the public-CLI equivalent of `--jacobian-sidecar` — no external
        // `.npz`, no CUDA, no Python. T2.1.R tier (~+6.24 dB over SuperSplat
        // on bonsai canonical-11 — see experiments/3tier-leaderboard/RESULT.md).
        let t0 = std::time::Instant::now();
        let result = catetus_jacobian::compute_jacobian(&scene);
        let weights = result.j_sh_rest;
        let elapsed = t0.elapsed();
        println!(
            "auto-jacobian: computed {} SH-rest weights in {:.3}s ({} method)",
            weights.len(),
            elapsed.as_secs_f64(),
            match result.method {
                catetus_jacobian::JacobianMethod::GeometricProxyV1 => "GeometricProxyV1",
            },
        );
        if weights.len() != scene.splats.len() {
            return Err(anyhow!(
                "--auto-jacobian: internal length mismatch ({} weights vs {} splats); \
                 please file a bug",
                weights.len(),
                scene.splats.len()
            ));
        }
        init_ctx.sh_rest_weights = Some(weights);
    } else if let Some(p) = jacobian_sidecar {
        let weights = if emit_v5_tail.is_some() {
            let joint = load_joint_jacobian_sum_from_npz(p).with_context(|| {
                format!(
                    "loading joint Jacobian from {} (expected stored-mode .npz \
                     containing one or more of J_position / J_rotation / \
                     J_opacity / J_scale / J_dc / J_sh_rest, or a precomputed \
                     J_joint_sum array)",
                    p.display()
                )
            })?;
            joint_jacobian = Some(joint.clone());
            joint
        } else {
            load_jacobian_sh_rest_from_npz(p).with_context(|| {
                format!(
                    "loading SH-rest Jacobian sidecar from {} (expected \
                     stored-mode .npz containing a `J_sh_rest` float32 array)",
                    p.display()
                )
            })?
        };
        if weights.len() != scene.splats.len() {
            return Err(anyhow!(
                "--jacobian-sidecar length mismatch: sidecar has {} entries, \
                 input scene has {} splats. The sidecar must be computed on \
                 the SAME PLY (pre-pipeline) that you're optimizing.",
                weights.len(),
                scene.splats.len()
            ));
        }
        println!(
            "loaded Jacobian sidecar: {} weights from {} (mode={})",
            weights.len(),
            p.display(),
            if emit_v5_tail.is_some() {
                "joint-J for V5.2"
            } else {
                "J_sh_rest only"
            }
        );
        init_ctx.sh_rest_weights = Some(weights);
    } else if emit_v5_tail.is_some() {
        return Err(anyhow!(
            "--emit-v5-tail requires --jacobian-sidecar pointing at a multi-array \
             .npz from `jacobian-census-canonical-11` (or a precomputed J_joint_sum)"
        ));
    }
    // Seed splat-origin tracking when we'll need it for V5.2 residuals.
    if emit_v5_tail.is_some() {
        init_ctx.splat_origin_idx = Some((0u32..scene.splats.len() as u32).collect());
    }
    let (report, post_ctx) = if progress {
        pipe.run_with_progress_returning_ctx(&mut scene, init_ctx, |i, total, name| {
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
        pipe.run_with_progress_returning_ctx(&mut scene, init_ctx, |_, _, _| {})?
    };
    let _ = joint_jacobian; // joint-J pre-pipeline kept around for the sidecar
                            // encoder; we actually re-derive from post_ctx below
                            // so the indexing matches the post-pipeline scene.

    // `--target tileset` short-circuits to the streaming octree LOD writer
    // (`catetus-tileset`). The optimize pipeline above has already run on
    // `scene` (Morton sort, prune, etc. per `--preset`), so the tileset's
    // octree is built over the post-optimize splats. Each tile is encoded as a
    // standalone SF GLB; the output directory is HTTP-servable for progressive
    // streaming viewers (root tile first, refine into children by projected
    // size). See `crates/catetus-tileset/STATUS.md`.
    if target_tileset {
        let dir = output_dir.map(PathBuf::from).unwrap_or_else(|| {
            let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("scene");
            input
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!("{stem}-tileset"))
        });
        if progress {
            emit_progress(0.92, "writing-tileset");
        }
        let total_splats = scene.len();
        if total_splats == 0 {
            return Err(anyhow!("input scene has no splats to tile"));
        }
        let ts_config = catetus_tileset::TilesetConfig::default();
        let plan = catetus_tileset::plan_tileset(&scene, &ts_config)
            .map_err(|e| anyhow!("tileset planning failed: {e}"))?;
        let tile_count = plan.payloads.len();
        let lod_levels = plan.lod_meta.lod_levels;
        // Per-node-LOD (tile_count, splat_count) summary across all nodes.
        let mut per_lod: Vec<(usize, usize)> = Vec::new();
        for p in &plan.payloads {
            if per_lod.len() <= p.lod {
                per_lod.resize(p.lod + 1, (0, 0));
            }
            per_lod[p.lod].0 += 1;
            per_lod[p.lod].1 += p.scene.len();
        }
        std::fs::create_dir_all(&dir)?;
        let total_tile_bytes = if shared_palette {
            // Shared-palette tileset: one scene-global SH-rest codebook
            // (`palette.shpal`) + per-tile `.glb.shpalx` index sidecars.
            let written = catetus_tileset::write_tileset_shared(
                &plan,
                &scene,
                &dir,
                &catetus_tileset::SharedPaletteConfig::default(),
            )
            .map_err(|e| anyhow!("writing shared-palette tileset failed: {e}"))?;
            written.total_bytes
        } else {
            let codec = catetus_tileset::GlbTileCodec::from_cli_preset(preset_name);
            catetus_tileset::write_tileset(&plan, &codec, &dir)
                .map_err(|e| anyhow!("writing tileset failed: {e}"))?
        };
        if progress {
            emit_progress(1.00, "done");
        }
        let kind = if shared_palette {
            "shared-palette tileset"
        } else {
            "tileset"
        };
        println!(
            "optimized {} -> {} {} ({} GLB tiles, {} octree LOD levels, {} splats)",
            input.display(),
            kind,
            dir.display(),
            tile_count,
            lod_levels,
            total_splats,
        );
        if shared_palette {
            println!(
                "  shared SH-rest palette: {}",
                dir.join(catetus_tileset::SHARED_PALETTE_FILENAME).display()
            );
        }
        let last_nonempty = per_lod.iter().rposition(|&(t, _)| t > 0).unwrap_or(0);
        for (lod, &(ntiles, nsplats)) in per_lod.iter().enumerate() {
            if ntiles == 0 {
                continue;
            }
            let tag = if lod == 0 {
                "coarsest"
            } else if lod == last_nonempty {
                "finest"
            } else {
                "mid"
            };
            println!("  per-node LOD {lod} ({tag}): {ntiles} tiles, {nsplats} splats");
        }
        println!(
            "  manifests: lod-meta.json (SuperSplat) + tileset.json (3D Tiles 1.1); \
             {:.2} MB tile payloads",
            total_tile_bytes as f64 / (1024.0 * 1024.0),
        );
        return Ok(());
    }

    // `geospatial` short-circuits to the (legacy) Cesium tileset writer — no
    // single .gltf out.
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
    // Force --target glb for presets that ship a GLB-only lossless wrap; the
    // wrap lives on the BIN chunk and has no representation in the external
    // gltf+bin layout. Must run before `default_ext` so the implicit output
    // path lands on `.glb`.
    let target_glb = target_glb
        || matches!(
            preset_name,
            "wmv-sh3-q8-zstd"
                | "web-mobile-vq45"
                | "wmv-vq45-no-prune"
                | "wmv-vq45-no-prune-tight"
                | "wmv-vq45-no-prune-tight-smallest3"
                | "wmv-vq45k1024-no-prune-tight"
                | "wmv-vq45k4096-no-prune-tight"
                | "wmv-vq45k16384-no-prune-tight"
                | "wmv-vq45k4096-posthac-no-prune-tight"
                | "wmv-vq45-posthac-no-prune-tight"
                | "wmv-vq45-tight-v2"
        );
    let default_ext = if target_sog {
        "optimized.sog"
    } else if target_glb {
        "optimized.glb"
    } else {
        "optimized.gltf"
    };
    let out = out
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension(default_ext));
    // SOG short-circuit: when `--target sog` we run the same optimize
    // pipeline (already finished above) but bypass the GLB writer entirely
    // and emit a PlayCanvas SOG container instead. The SH-rest palette
    // produced by `VQPaletteShRest` (in render-space Jacobian-weighted
    // mode when `--jacobian-sidecar` was passed) rides through to
    // `shN_centroids.webp` — that's the +3 to +6 dB lift over PlayCanvas's
    // stock encoder at the same byte budget. See
    // `experiments/sog-render-weighted/RESULT.md`.
    // `--target sog` short-circuits to the hosted `/v1/encode` route at
    // the top of `cmd_optimize` — by the time we reach this point the
    // request has already returned. Keep the dead branch as a defensive
    // guard so a future refactor that drops the short-circuit fails
    // loudly instead of running the GLB writer with target=sog.
    if target_sog {
        unreachable!(
            "--target sog should have been handled by the hosted-encode \
             short-circuit at the top of cmd_optimize"
        );
    }
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
                // catetus-private/apps/diff-repack), this just encodes
                // the quantized splats into glTF.
                | "hosted-neural-outdoor"
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
                // FCGS hosted preset (Chen et al. ICLR'25). Pre-trained
                // feed-forward codec; the encoder runs on a Modal A100
                // and emits a bitstream that the glTF wrapper points
                // at — quantized integer accessors keep the manifest
                // small even when the bitstream is referenced as a
                // sidecar.
                | "fcgs-instant"
                // PRUNE_FIX_BENCH SH-rest quantization presets — pos/scale/rot
                // are already lossy-quantized by the pipeline; turn on the
                // writer's KHR_mesh_quantization integer accessors so they
                // actually shrink to BYTE/SHORT on disk. The SH_DEGREE_l_COEF_n
                // accessors then ride the WriteOpts.sh_rest_quant table for
                // BYTE/SHORT encoding too.
                | "wmv-sh3-q8"
                | "wmv-sh3-q6"
                | "web-mobile-sh3-q8"
                // `wmv-sh3-q8` + the byte-plane zstd-19 lossless wrap. The
                // wrap is dispatched below by the preset → lossless map; we
                // only need to mirror the quantize-flag setting here.
                | "wmv-sh3-q8-zstd"
                // VQPaletteShRest presets — pos/scale/rot are quantized
                // by the pipeline (same as the wmv-sh3-q8* family) and
                // the GLB writer's quantized accessors give us the byte
                // savings on those. SH-rest itself rides the `.shpal`
                // sidecar drained below; the in-GLB SH accessors still
                // round-trip cluster centroids via the quantized path,
                // which keeps the file small while the actual
                // 65,536-entry codebook ships separately.
                | "web-mobile-vq45"
                | "wmv-vq45-no-prune"
                | "wmv-vq45-no-prune-tight"
                | "wmv-vq45-no-prune-tight-smallest3"
                | "wmv-vq45k1024-no-prune-tight"
                | "wmv-vq45k4096-no-prune-tight"
                | "wmv-vq45k16384-no-prune-tight"
                | "wmv-vq45k4096-posthac-no-prune-tight"
                | "wmv-vq45-posthac-no-prune-tight"
                | "wmv-vq45-tight-v2"
        );
    // Preset → lossless-wrap implication. Some presets (e.g. `wmv-sh3-q8-zstd`)
    // bake the wrap into their identity so users don't have to remember a
    // `--lossless` flag. An explicit `--lossless` from the CLI always wins.
    let lossless_wrap = lossless_wrap.or_else(|| match preset_name {
        "wmv-sh3-q8-zstd"
        | "web-mobile-vq45"
        | "wmv-vq45-no-prune"
        | "wmv-vq45-no-prune-tight"
        | "wmv-vq45-no-prune-tight-smallest3"
        | "wmv-vq45k1024-no-prune-tight"
        | "wmv-vq45k4096-no-prune-tight"
        | "wmv-vq45k16384-no-prune-tight"
        | "wmv-vq45k4096-posthac-no-prune-tight"
        | "wmv-vq45-posthac-no-prune-tight"
        | "wmv-vq45-tight-v2" => Some(catetus_gltf::LosslessWrap::Zstd19Split),
        _ => None,
    });
    let compress_variant = if compress_mode == Some("spz") {
        Some(catetus_gltf::SpzVariant::V2)
    } else {
        None
    };
    // Drain any QuantizeSHRest side table from the just-finished pipeline so
    // the GLB writer can emit BYTE/SHORT SH-rest accessors with per-channel
    // min/max instead of FP32. No-op for preset chains that don't include
    // `QuantizeSHRest`.
    let sh_rest_quant = take_last_sh_rest_quant_table().map(|t| ShRestQuantTable {
        bits: t.bits,
        ranges: t.ranges,
    });
    // SOG-style smallest-3 quaternion side table. Drained from the pipeline
    // by the `QuantizeRotationSmallest3` pass; when `Some`, the GLB writer
    // emits ROTATION as a SCALAR UNSIGNED_INT accessor + `SF_quat_smallest3`
    // root extension. See `experiments/SOG_STUDY_RUN/SMALLEST3_QUAT_RESULT.md`.
    let rotation_smallest3 = take_last_rotation_smallest3_table().map(|t| RotationSmallest3Table {
        component_bits: t.component_bits,
    });
    // SOG_STUDY_RUN TIGHT pack — packed-rotation + packed-DC side tables
    // park 8-bit ROTATION (VEC4 UBYTE-normalized, 4 B/splat) and DC
    // (VEC3 UBYTE-normalized, 3 B/splat) with per-component min/max.
    let rotation_quant = take_last_rotation_quant_table().map(|t| RotationQuantTable {
        bits: t.bits,
        mins: t.mins,
        maxs: t.maxs,
    });
    let dc_quant = take_last_dc_quant_table().map(|t| DcQuantTable {
        bits: t.bits,
        mins: t.mins,
        maxs: t.maxs,
    });
    // BEFORE we call the GLB writer so we can (a) write the `.shpal` sidecar
    // next to the output and (b) hand the writer a `ShRestPaletteRef` so it
    // suppresses every `SH_DEGREE_l_COEF_n` accessor in favour of the
    // `SF_gaussian_splatting_palette` root extension. The drain order matters
    // — the prior code path drained the palette AFTER `write_glb`, which left
    // the SH-rest accessors in the GLB and bloated the asset (see VQ45_RESULT
    // for the 68 MB baseline). Doing the drain here means a single end-to-end
    // shot produces the GLB and sidecar in the same coordinate space.
    let pal_drained = take_last_sh_rest_palette();
    let palette_ref = if let Some(pal) = &pal_drained {
        // Sidecar lives next to the GLB, suffixed with `.shpal` appended to
        // whatever extension `out` already has (mirrors the SplatDelta sidecar
        // suffix convention).
        let pal_path = out.with_extension({
            let cur = out.extension().and_then(|s| s.to_str()).unwrap_or("");
            if cur.is_empty() {
                "shpal".to_string()
            } else {
                format!("{cur}.shpal")
            }
        });
        let sidecar_uri = pal_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("scene.shpal")
            .to_string();
        // Probe the scene's max SH degree so the extension carries the same
        // value decoders use to size their per-splat reconstruction.
        let scene_sh_degree: u8 = scene
            .splats
            .iter()
            .map(|s| s.color.degree())
            .max()
            .unwrap_or(0);
        if let Err(e) = std::fs::write(&pal_path, &pal.compressed) {
            eprintln!("warning: failed to write .shpal sidecar: {e}");
            None
        } else {
            println!(
                "wrote VQPaletteShRest sidecar {} ({} bytes, raw {}B, K={}, N={}, mse={:.4e}, kmeans_ms={}, encode_ms={})",
                pal_path.display(),
                pal.compressed.len(),
                pal.raw_len,
                pal.palette_size,
                pal.n_splats,
                pal.stats.mse,
                pal.stats.kmeans_ms,
                pal.stats.encode_ms,
            );
            // PostHAC companion sidecar — only when the VQPaletteShRest pass
            // was configured with `posthac_indices = true`. Written next to
            // the .shpal at `<out>.shpal.pthc`. The JS bench decoder doesn't
            // yet consume this; the composed-bytes accounting in the bench
            // report substitutes `pthc_len - raw_index_bytes_via_zstd_share`
            // for the savings (see experiments/SOG_STUDY_RUN/VQ45_POSTHAC_*).
            if let Some(pthc) = &pal.posthac_indices {
                let pthc_path = pal_path.with_extension({
                    let cur = pal_path.extension().and_then(|s| s.to_str()).unwrap_or("");
                    if cur.is_empty() {
                        "pthc".to_string()
                    } else {
                        format!("{cur}.pthc")
                    }
                });
                if let Err(e) = std::fs::write(&pthc_path, pthc) {
                    eprintln!("warning: failed to write .shpal.pthc sidecar: {e}");
                } else {
                    println!(
                        "wrote VQ45 PostHAC index sidecar {} ({} bytes; raw u16 indices = {} B)",
                        pthc_path.display(),
                        pthc.len(),
                        pal.stats.raw_index_bytes,
                    );
                }
            }
            Some(ShRestPaletteRef {
                sidecar_uri,
                palette_size: pal.palette_size,
                n_splats: pal.n_splats,
                codebook_bits: pal.codebook_bits,
                sh_degree: scene_sh_degree,
            })
        }
    } else {
        None
    };

    // V5.2 joint-tail residual sidecar deferred to AFTER the GLB write.
    // The residual must be computed against the post-GLB-decode recon so
    // it captures the `log_quant_attrs` UBYTE lossy round trip applied
    // inside `write_glb`. Subtracting against the in-memory `scene`
    // (which hasn't been through that quant) drops opa / sca residuals
    // to ~zero and loses 5+ dB of V5.2 headroom — see
    // `experiments/v5-2-rust-port/STATUS.md` for the full chain.
    //
    // To keep the GLB JSON consistent (it must list
    // `SF_v5_tail_residual` so `read_glb` knows to look for the
    // sidecar), we write the GLB twice when `emit_v5_tail` is set:
    //   pass 1: opts.v5_tail = None -> emit clean GLB used as the
    //           residual baseline.
    //   pass 2: opts.v5_tail = Some(V5TailRef) -> overwrite with the
    //           extension stamped on the JSON header.
    // The two GLB passes are ~1 s each on bonsai; cheap relative to the
    // VQ45 pass-time (~2 min) and the disk I/O is local.
    //
    // `post_ctx.splat_origin_idx` (length == scene.splats.len()) maps
    // post-pipeline splat index -> original input PLY row, used to
    // align GT attrs.
    let v5_tail_ref_placeholder: Option<catetus_gltf::V5TailRef> = None;
    // Decoder-conventions fix (#86): the wmv-vq45* family pairs its
    // `QuantizeScale { log_space: true }` pipeline pass with a GLB writer
    // that stores SCALE in log-space and OPACITY in logit-space, so the
    // round-trip preserves the heavy-tailed scale distribution. Uniform
    // 8-bit linear quant otherwise crushes everything below ~8e-4 to
    // zero, then the PLY writer's `ln(EPSILON)` reports it back as
    // -15.94 — visually that's a blurry, over-bright reconstruction in
    // SuperSplat. See experiments/decoder-conventions-fix/RESULT.md.
    let log_quant_attrs = quantize
        && matches!(
            preset_name,
            "web-mobile-vq45"
                | "wmv-vq45-no-prune"
                | "wmv-vq45-no-prune-tight"
                | "wmv-vq45-no-prune-tight-smallest3"
                | "wmv-vq45k1024-no-prune-tight"
                | "wmv-vq45k4096-no-prune-tight"
                | "wmv-vq45k16384-no-prune-tight"
                | "wmv-vq45k4096-posthac-no-prune-tight"
                | "wmv-vq45-posthac-no-prune-tight"
                | "wmv-vq45-tight-v2"
        );
    let mut opts = WriteOpts {
        chunked,
        chunk_target_splats: 100_000,
        lod_fractions: vec![1.0],
        quantize,
        quantize_rotation: false,
        spec_version: Default::default(),
        compress: compress_variant,
        lossless: lossless_wrap,
        sh_rest_quant,
        rotation_smallest3,
        rotation_quant,
        dc_quant,
        palette: palette_ref,
        v5_tail: v5_tail_ref_placeholder,
        log_quant_attrs,
    };
    if progress {
        emit_progress(0.92, "encoding-gltf");
    }
    if target_glb {
        write_glb(&scene, &out, &opts)?;
    } else {
        write_gltf(&scene, &out, &opts)?;
    }

    // V5.2 sidecar pass: read the just-written GLB back to get the
    // post-`write_glb` damaged recon, compute residuals against THAT,
    // write the sidecar, then re-emit the GLB JSON header with the
    // SF_v5_tail_residual extension stamped on. The sidecar `.glb.v5tail`
    // file lives next to the GLB and is loaded by `read_glb` at decode.
    if let Some(gt_path) = emit_v5_tail {
        let origin_idx = post_ctx
            .splat_origin_idx
            .as_ref()
            .ok_or_else(|| anyhow!("internal: splat_origin_idx missing from post-pipeline ctx"))?;
        let joint_score = post_ctx
            .sh_rest_weights
            .as_ref()
            .ok_or_else(|| anyhow!("internal: joint-J weights missing from post-pipeline ctx"))?;
        if !target_glb {
            return Err(anyhow!(
                "--emit-v5-tail requires --target glb (residuals piggyback on the GLB writer)"
            ));
        }
        // Read GLB back via the same path the bench uses. Permissive on
        // missing tail (the GLB we just wrote DOES advertise it via
        // SF_v5_tail_residual? — no, we wrote it WITHOUT v5_tail in opts
        // for pass 1. So no extension is present yet; the read is clean.)
        println!(
            "reading GLB back to compute residuals against the post-write_glb recon: {}",
            out.display()
        );
        let damaged_recon = catetus_gltf::read_glb_with_opts(
            &out,
            &catetus_gltf::ReadOpts {
                allow_missing_palette: false,
                allow_missing_tail: true,
            },
        )
        .with_context(|| {
            format!(
                "re-reading GLB {} for V5.2 residual baseline",
                out.display()
            )
        })?;
        if damaged_recon.splats.len() != scene.splats.len() {
            return Err(anyhow!(
                "GLB round-trip changed splat count: {} -> {}",
                scene.splats.len(),
                damaged_recon.splats.len()
            ));
        }
        let (sidecar_bytes, v5_tail_ref) =
            build_v5_tail_sidecar(&damaged_recon, gt_path, origin_idx, joint_score, &out)?;
        // Pass 2: re-write the GLB with the extension stamped.
        opts.v5_tail = Some(v5_tail_ref.clone());
        write_glb(&scene, &out, &opts)?;
        // Now persist the sidecar.
        let v5tail_path = out.with_extension({
            let cur = out.extension().and_then(|s| s.to_str()).unwrap_or("");
            if cur.is_empty() {
                "v5tail".to_string()
            } else {
                format!("{cur}.v5tail")
            }
        });
        if let Err(e) = std::fs::write(&v5tail_path, &sidecar_bytes) {
            eprintln!("warning: failed to write .v5tail sidecar: {e}");
        } else {
            println!(
                "wrote V5.2 tail-residual sidecar {} ({} bytes, K={}/{}, n_cells={})",
                v5tail_path.display(),
                sidecar_bytes.len(),
                v5_tail_ref.k_selected,
                v5_tail_ref.n_splats,
                v5_tail_ref.n_cells,
            );
        }
    }
    // If the pipeline included `SplatDelta`, drain its sidecar blob and write
    // a `<out>.splatdelta` companion. The blob is the zstd-compressed
    // anchor-stride residual stream — what gets compared against `.sog` in
    // the integration verdict for `web-mobile-delta`.
    if let Some(delta) = take_last_delta_stream() {
        let delta_path = out.with_extension({
            let cur = out.extension().and_then(|s| s.to_str()).unwrap_or("");
            if cur.is_empty() {
                "splatdelta".to_string()
            } else {
                format!("{cur}.splatdelta")
            }
        });
        if let Err(e) = std::fs::write(&delta_path, &delta.compressed) {
            eprintln!("warning: failed to write .splatdelta sidecar: {e}");
        } else {
            println!(
                "wrote SplatDelta sidecar {} ({} bytes, raw {}B, anchors={}/{})",
                delta_path.display(),
                delta.compressed.len(),
                delta.raw_len,
                delta.stats.n_anchor,
                delta.stats.n,
            );
        }
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

/// Build the V5.2 joint-tail residual sidecar for `--emit-v5-tail`.
///
/// `recon_scene` is the post-pipeline (baseline-quantized) scene. We
/// re-load the GT PLY, pick the top-1% selection by `joint_score`, gather
/// pre- and post-residual attrs in SF-ascending order, Morton-sort by
/// recon positions on the selected subset, compute GT - recon residuals
/// in Morton order, and call [`v5_tail::encode_v5_2_sidecar`].
///
/// Returns `(sidecar_bytes, V5TailRef)` so the caller can both write the
/// `.glb.v5tail` file AND stamp the `SF_v5_tail_residual` root extension
/// on the GLB JSON.
fn build_v5_tail_sidecar(
    recon_scene: &SplatScene,
    gt_ply_path: &Path,
    origin_idx: &[u32],
    joint_score: &[f32],
    out_glb_path: &Path,
) -> Result<(Vec<u8>, catetus_gltf::V5TailRef)> {
    use catetus_gltf::v5_tail;
    let n = recon_scene.splats.len();
    if origin_idx.len() != n {
        return Err(anyhow!(
            "splat_origin_idx length {} != recon scene length {}",
            origin_idx.len(),
            n
        ));
    }
    if joint_score.len() != n {
        return Err(anyhow!(
            "joint_score length {} != recon scene length {}",
            joint_score.len(),
            n
        ));
    }

    // 1) Load the GT PLY. Its row count must match the original input PLY
    //    that produced `origin_idx` (the user is expected to pass the same
    //    file as both --input and --emit-v5-tail in the standard case, but
    //    we explicitly support GT = high-precision-source even when the
    //    optimize input is a different snapshot — the structural pipeline
    //    is deterministic and `origin_idx[i]` carries the mapping).
    println!(
        "loading GT PLY for V5.2 residuals: {}",
        gt_ply_path.display()
    );
    let gt_scene = read_ply(gt_ply_path)
        .with_context(|| format!("reading GT PLY {}", gt_ply_path.display()))?;
    // The recon scene's `origin_idx[i]` indexes into the ORIGINAL input PLY
    // (which structurally matches gt_scene 1:1 if same file). We require
    // gt_scene.splats.len() > max(origin_idx). Looser than == because the
    // input PLY might have had invalid splats that RemoveInvalidSplats
    // pruned; origin_idx in the post-pipeline scene refers only to kept
    // rows of the ORIGINAL PLY, so any kept row < gt_scene.len() is fine.
    let max_origin = origin_idx.iter().copied().max().unwrap_or(0) as usize;
    if max_origin >= gt_scene.splats.len() {
        return Err(anyhow!(
            "GT PLY has {} splats but post-pipeline origin_idx references row {} \
             — make sure --emit-v5-tail points at the SAME PLY (or a row-aligned \
             snapshot) used as the optimize input",
            gt_scene.splats.len(),
            max_origin
        ));
    }
    let sh_rest_coefs = recon_sh_rest_coefs(recon_scene);
    // The caller is responsible for passing a `recon_scene` that
    // matches the bench's view of the decoded GLB (i.e. post `write_glb`
    // → `read_glb`). The earlier in-memory subtract path missed the
    // `log_quant_attrs` UBYTE damage and lost 5 dB of V5.2 headroom —
    // see `experiments/v5-2-rust-port/STATUS.md`. As of Phase C v4 the
    // CLI passes the just-`read_glb`-ed scene here, so `recon_scene`
    // already reflects the encoder side's quant losses.
    let recon_for_residual = recon_scene;

    // 2) Selection: top K = 1% of N by joint-J score, indices into the
    //    POST-PIPELINE scene. Sort ascending after picking so the encoder
    //    sees ascending-SF-order indices.
    let k_percent = 1.0f32;
    let k = (n as f32 * k_percent / 100.0).round() as usize;
    if k == 0 {
        return Err(anyhow!(
            "V5.2 selection K = {} (need K > 0; scene has {} splats)",
            k,
            n
        ));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        joint_score[b]
            .partial_cmp(&joint_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sel_idx_sf: Vec<u32> = order.iter().take(k).map(|&i| i as u32).collect();
    sel_idx_sf.sort_unstable();
    let mut sel_bool = vec![false; n];
    for &i in &sel_idx_sf {
        sel_bool[i as usize] = true;
    }
    println!(
        "V5.2 selection: K={} (top {:.2}% of {} splats), sh_rest_coefs={}",
        k, k_percent, n, sh_rest_coefs
    );

    // 3) Morton-sort the selected subset by recon positions, then build
    //    Morton-order GT and recon attribute matrices and subtract.
    let positions_selected: Vec<[f32; 3]> = sel_idx_sf
        .iter()
        .map(|&i| recon_for_residual.splats[i as usize].position)
        .collect();
    let morton_idx = v5_tail::morton_sort_indices(&positions_selected);

    // Pre-allocate Morton-order residual buffers (one per group).
    let mut pos_res = vec![0f32; k * 3];
    let mut rot_res = vec![0f32; k * 4];
    let mut opa_res = vec![0f32; k];
    let mut sca_res = vec![0f32; k * 3];
    let mut dc_res = vec![0f32; k * 3];
    let mut shr_res = vec![0f32; k * sh_rest_coefs * 3];
    for (m_row, &sf_row) in morton_idx.iter().enumerate() {
        let sf_idx = sel_idx_sf[sf_row as usize] as usize;
        let gt_row = origin_idx[sf_idx] as usize;
        // Subtract against the PLY-round-tripped recon, NOT the in-memory
        // one. This is where the tail-clamp loss lives.
        let r = &recon_for_residual.splats[sf_idx];
        let g = &gt_scene.splats[gt_row];
        for c in 0..3 {
            pos_res[m_row * 3 + c] = g.position[c] - r.position[c];
        }
        for c in 0..4 {
            rot_res[m_row * 4 + c] = g.rotation[c] - r.rotation[c];
        }
        // Residuals are computed in **raw 3DGS-PLY space** (log-scale,
        // logit-opacity) to match the Python V5.2 prototype. `read_ply`
        // applies `exp` / `sigmoid` on load, so we round-trip through
        // `ln` / `logit` here before subtracting. Without this the sca /
        // opa residuals collapse to ~1e-7 (linear-space quant-12 of a
        // tiny IR value) and the bench misses V5.2's 5-dB headroom.
        // Root cause + numerics in `experiments/v5-2-rust-port/STATUS.md`.
        opa_res[m_row] = v5_tail_logit(g.opacity) - v5_tail_logit(r.opacity);
        for c in 0..3 {
            sca_res[m_row * 3 + c] = v5_tail_ln(g.scale[c]) - v5_tail_ln(r.scale[c]);
        }
        // DC + SH-rest: pull from Color::Sh::coeffs (DC at [0..3], SH-rest
        // at [3..]). For RGB-only colours we treat SH-rest as zero.
        let (r_dc, r_shr): (&[f32], &[f32]) = match &r.color {
            Color::Sh { coeffs, .. } if coeffs.len() >= 3 => {
                let dc = &coeffs[0..3];
                let shr = if coeffs.len() > 3 {
                    &coeffs[3..]
                } else {
                    &[][..]
                };
                (dc, shr)
            }
            Color::Rgb(rgb) => (rgb, &[][..]),
            _ => (&[0.0, 0.0, 0.0][..], &[][..]),
        };
        let (g_dc, g_shr): (&[f32], &[f32]) = match &g.color {
            Color::Sh { coeffs, .. } if coeffs.len() >= 3 => {
                let dc = &coeffs[0..3];
                let shr = if coeffs.len() > 3 {
                    &coeffs[3..]
                } else {
                    &[][..]
                };
                (dc, shr)
            }
            Color::Rgb(rgb) => (rgb, &[][..]),
            _ => (&[0.0, 0.0, 0.0][..], &[][..]),
        };
        for c in 0..3 {
            dc_res[m_row * 3 + c] = g_dc[c] - r_dc[c];
        }
        let want = sh_rest_coefs * 3;
        for c in 0..want {
            let gv = if c < g_shr.len() { g_shr[c] } else { 0.0 };
            let rv = if c < r_shr.len() { r_shr[c] } else { 0.0 };
            shr_res[m_row * want + c] = gv - rv;
        }
    }

    // 4) Build cell partitions + invoke the encoder.
    let n_cells = 64usize;
    let cell_offsets = v5_tail::build_cell_offsets(k, n_cells);
    let actual_n_cells = cell_offsets.len() - 1;
    let residuals = v5_tail::Residuals {
        k_selected: k,
        sh_rest_coefs,
        pos: pos_res,
        rot: rot_res,
        opa: opa_res,
        sca: sca_res,
        dc: dc_res,
        shr: shr_res,
    };
    let bit_depths = v5_tail::BitDepths::v5_2();
    let (sidecar_bytes, sizes) = v5_tail::encode_v5_2_sidecar(
        n,
        &sel_bool,
        &morton_idx,
        &residuals,
        bit_depths,
        &cell_offsets,
    )
    .context("V5.2 sidecar encode")?;
    println!(
        "V5.2 sidecar encoded: total={} bytes (mask={} morton={} cell_off={} groups=[{}])",
        sizes.total_bytes,
        sizes.mask_zstd,
        sizes.morton_zstd,
        sizes.cell_offsets_zstd,
        sizes
            .groups
            .iter()
            .map(|g| format!("{}b/m={}/p={}", g.bit_depth, g.meta_zstd, g.payload_zstd))
            .collect::<Vec<_>>()
            .join(", "),
    );

    // 5) Build the V5TailRef metadata pointer for the GLB writer.
    let sidecar_uri = {
        let glb_name = out_glb_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("scene.glb");
        format!("{glb_name}.v5tail")
    };
    let v5_tail_ref = catetus_gltf::V5TailRef {
        sidecar_uri,
        n_splats: n,
        k_selected: k,
        sh_rest_coefs: sh_rest_coefs as u8,
        n_cells: actual_n_cells as u16,
        required: false,
    };
    Ok((sidecar_bytes, v5_tail_ref))
}

/// Inspect the recon scene to determine how many SH-rest coefficients per
/// channel each splat carries. Returns the max degree's coefficient count
/// (so sh_degree=3 → 15, sh_degree=2 → 8, etc.). 0 when the scene is all
/// `Color::Rgb`.
/// Inverse of `read_ply`'s `sigmoid` on opacity. Clamped to (1e-7, 1-1e-7).
/// Mirrors `catetus-ply::logit`. Used by `build_v5_tail_sidecar` to
/// convert IR-space opacity back to raw 3DGS-PLY logit space before
/// computing the residual.
#[inline]
fn v5_tail_logit(p: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln()
}

/// Inverse of `read_ply`'s `exp` on scale. Mirrors `catetus-ply::ln_scale`.
#[inline]
fn v5_tail_ln(s: f32) -> f32 {
    s.max(f32::MIN_POSITIVE).ln()
}

fn recon_sh_rest_coefs(scene: &SplatScene) -> usize {
    let mut max_coefs = 0usize;
    for s in &scene.splats {
        if let Color::Sh { coeffs, .. } = &s.color {
            let coefs = coeffs.len().saturating_sub(3) / 3;
            if coefs > max_coefs {
                max_coefs = coefs;
            }
        }
    }
    max_coefs
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

/// `catetus sog-emit-v5-tail <sog> --gt <gt.ply>` — hosted call.
///
/// The SOG writer + `.sog.v5tail` emitter moved to the private
/// `catetus/catetus-private/crates/catetus-sog` crate as part of the
/// 2026-05-19 open-core split. The hosted `/v1/encode?target=sog&v5tail=true`
/// route runs the same encoder on the server side and returns both the
/// SOG container AND the V5.2 sidecar — we discard the SOG (the caller
/// already has one) and write the sidecar next to the provided `.sog`
/// file (default `<sog>.v5tail`, mirroring the GLB-path naming).
///
/// Notes / known limitations:
///   - The hosted route re-encodes from the GT PLY using the server's
///     default profile; `--profile`, `--k-percent`, `--jacobian-sidecar`,
///     and `--dump-residual-stats` are advisory locally and not yet
///     forwarded to the worker. See LAUNCH4_BLOCKER.md for the open
///     plumbing task.
fn cmd_sog_emit_v5tail(
    sog: &Path,
    gt: &Path,
    k_percent: f32,
    profile: Option<&str>,
    jacobian_npz: Option<&Path>,
    dump_residual_stats: Option<&Path>,
    api_url: Option<&str>,
) -> Result<()> {
    if !sog.exists() {
        return Err(anyhow!("SOG not found at {}", sog.display()));
    }
    if !gt.exists() {
        return Err(anyhow!("GT PLY not found at {}", gt.display()));
    }
    let sidecar_path = {
        let mut p = sog.to_path_buf();
        let new_ext = match p.extension().and_then(|s| s.to_str()) {
            Some(ext) => format!("{ext}.v5tail"),
            None => "v5tail".to_string(),
        };
        p.set_extension(new_ext);
        p
    };
    if jacobian_npz.is_some()
        || dump_residual_stats.is_some()
        || profile.is_some()
        || (k_percent - 0.01).abs() > 1e-6
    {
        eprintln!(
            "catetus: note: --jacobian-sidecar / --dump-residual-stats / --profile / \
             --k-percent are not yet forwarded to the hosted encoder. The server \
             will use its default profile (V5.2 8/10/12/12/8/8, top-1% selection). \
             See LAUNCH4_BLOCKER.md for the forwarding task."
        );
    }
    let resolved_api = resolve_api_url(api_url);
    // The hosted encode route's `v5tail=true` response carries both the
    // SOG and the V5.2 sidecar. We use a throwaway temp path for the
    // SOG (caller already has theirs).
    let tmp_dir = std::env::temp_dir();
    let mut tmp_sog = tmp_dir.clone();
    tmp_sog.push(format!(
        "catetus-emit-v5tail-{}.sog",
        std::process::id()
    ));
    let outcome = run_encode_to_disk(
        &resolved_api,
        ApiEncodeTarget::Sog,
        true,
        gt,
        &tmp_sog,
        Some(&sidecar_path),
        Some(&format!(
            "sog-emit-v5-tail for {}",
            sog.file_name().and_then(|s| s.to_str()).unwrap_or("?")
        )),
        std::time::Duration::from_secs(300),
    )?;
    // Best-effort tidy. Failure to remove the temp SOG is not fatal —
    // tempdir sweeps eventually catch it.
    let _ = std::fs::remove_file(&tmp_sog);
    let sidecar_bytes = outcome
        .sidecar_bytes
        .as_ref()
        .map(|b| b.len())
        .unwrap_or(0);
    println!(
        "hosted-sog emit-v5-tail: job_id={} -> {} ({} bytes sidecar; SOG {} bytes discarded)",
        outcome.job_id,
        sidecar_path.display(),
        sidecar_bytes,
        outcome.output_bytes.len(),
    );
    Ok(())
}

/// Parse a `pos/rot/opa/sca/dc/shr` bit-depth profile string. Accepts a
/// handful of named presets in addition to explicit slash-separated lists.
///
/// Currently retained for parity with the private CLI surface (would be wired
/// in if the hosted-only SOG branch were reactivated). Marked dead-code
/// because public CLI no longer reaches it post the 2026-05-19 open-core split.
#[allow(dead_code)]
fn parse_bit_depth_profile(s: &str) -> Result<catetus_gltf::v5_tail::BitDepths> {
    use catetus_gltf::v5_tail::BitDepths;
    match s {
        // V5.2 default profile — currently Phase D Path B (8/10/14/14/8/8).
        "default" | "v5_2" | "v5.2" => return Ok(BitDepths::v5_2()),
        // Legacy V5.2 Phase C ship profile (8/10/12/12/8/8). Use this to
        // reproduce pre-Path-B bench numbers or to mint a sidecar that
        // matches the v=1 wire-format header byte for older readers in
        // the wild.
        "v5_2_v1" | "v5.2.v1" | "phase-c" => return Ok(BitDepths::v5_2_v1()),
        // Retuned SOG profile — see experiments/sog-v5tail-retune/RESULT.md.
        // Trades opacity/dc bits for more SH-rest precision, where the SOG
        // codebook leaves the largest residuals.
        "sog-tight" | "sog_tight" => {
            return Ok(BitDepths {
                pos: 8,
                rot: 8,
                opa: 8,
                sca: 10,
                dc: 10,
                shr: 12,
            });
        }
        _ => {}
    }
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 6 {
        return Err(anyhow!(
            "bit-depth profile must be 6 slash-separated values (pos/rot/opa/sca/dc/shr) or a named preset; got `{}`",
            s
        ));
    }
    let mut bd = [0u8; 6];
    for (i, p) in parts.iter().enumerate() {
        let v: u32 = p
            .trim()
            .parse()
            .with_context(|| format!("parsing bit-depth `{}`", p))?;
        if !(1..=16).contains(&v) {
            return Err(anyhow!("bit-depth {} out of supported range 1..=16", v));
        }
        bd[i] = v as u8;
    }
    Ok(BitDepths {
        pos: bd[0],
        rot: bd[1],
        opa: bd[2],
        sca: bd[3],
        dc: bd[4],
        shr: bd[5],
    })
}

/// `catetus sog-apply-v5-tail <sog> -o <out.ply>` — hosted call.
///
/// The SOG reader + `.sog.v5tail` applier moved to the private
/// `catetus/catetus-private/crates/catetus-sog` crate. Public callers
/// POST `{sog_b64, sidecar_b64}` to `<api>/v1/decode?source=sog&v5tail=true`
/// and receive the reconstructed PLY bytes back.
///
/// The `/v1/decode` route is currently a planned addition to the hosted
/// API (see LAUNCH4_BLOCKER.md); the client is wired against the
/// protocol so the CLI works the moment the server lands it. Until
/// then a real call returns a clear "endpoint not implemented yet"
/// error.
fn cmd_sog_apply_v5tail(
    sog: &Path,
    sidecar: Option<&Path>,
    out: &Path,
    gt: Option<&Path>,
    api_url: Option<&str>,
) -> Result<()> {
    if !sog.exists() {
        return Err(anyhow!("SOG not found at {}", sog.display()));
    }
    let sidecar_path = match sidecar {
        Some(p) => p.to_path_buf(),
        None => {
            let mut p = sog.to_path_buf();
            let new_ext = match p.extension().and_then(|s| s.to_str()) {
                Some(ext) => format!("{ext}.v5tail"),
                None => "v5tail".to_string(),
            };
            p.set_extension(new_ext);
            p
        }
    };
    if !sidecar_path.exists() {
        return Err(anyhow!(
            "sidecar not found at {} (pass --sidecar to override the default `<sog>.v5tail` lookup)",
            sidecar_path.display()
        ));
    }
    let sog_bytes = std::fs::read(sog)
        .with_context(|| format!("reading SOG {}", sog.display()))?;
    let sidecar_bytes = std::fs::read(&sidecar_path)
        .with_context(|| format!("reading sidecar {}", sidecar_path.display()))?;
    let resolved_api = resolve_api_url(api_url);
    let recon_bytes = apply_v5tail_via_api(
        &resolved_api,
        &sog_bytes,
        &sidecar_bytes,
        std::time::Duration::from_secs(300),
    )?;
    std::fs::write(out, &recon_bytes)
        .with_context(|| format!("writing reconstructed PLY to {}", out.display()))?;
    println!(
        "hosted-sog apply-v5-tail: {} ({} bytes) + {} ({} bytes) -> {} ({} bytes)",
        sog.display(),
        sog_bytes.len(),
        sidecar_path.display(),
        sidecar_bytes.len(),
        out.display(),
        recon_bytes.len(),
    );
    if let Some(_gt) = gt {
        eprintln!(
            "catetus: note: --gt (per-attribute L1 diff vs ground truth) is not yet \
             forwarded to the hosted decoder. See LAUNCH4_BLOCKER.md."
        );
    }
    Ok(())
}

#[allow(dead_code)]
#[derive(Default, Debug)]
struct AttrL1 {
    pos: f64,
    rot: f64,
    opa: f64,
    sca: f64,
    dc: f64,
    shr: f64,
}

#[allow(dead_code)]
fn compare_l1(a: &SplatScene, b: &SplatScene) -> AttrL1 {
    let n = a.splats.len().min(b.splats.len());
    let mut out = AttrL1::default();
    for i in 0..n {
        let r = &a.splats[i];
        let g = &b.splats[i];
        for c in 0..3 {
            out.pos += (r.position[c] - g.position[c]).abs() as f64;
        }
        for c in 0..4 {
            out.rot += (r.rotation[c] - g.rotation[c]).abs() as f64;
        }
        let l = |p: f32| {
            let p = p.clamp(1e-7, 1.0 - 1e-7);
            (p / (1.0 - p)).ln()
        };
        out.opa += (l(r.opacity) - l(g.opacity)).abs() as f64;
        for c in 0..3 {
            let ra = r.scale[c].max(f32::MIN_POSITIVE).ln();
            let ga = g.scale[c].max(f32::MIN_POSITIVE).ln();
            out.sca += (ra - ga).abs() as f64;
        }
        let (ra_dc, ra_shr): (&[f32], &[f32]) = match &r.color {
            Color::Sh { coeffs, .. } if coeffs.len() >= 3 => {
                let dc = &coeffs[0..3];
                let rest = if coeffs.len() > 3 {
                    &coeffs[3..]
                } else {
                    &[][..]
                };
                (dc, rest)
            }
            Color::Rgb(rgb) => (rgb, &[][..]),
            _ => (&[0.0, 0.0, 0.0][..], &[][..]),
        };
        let (gb_dc, gb_shr): (&[f32], &[f32]) = match &g.color {
            Color::Sh { coeffs, .. } if coeffs.len() >= 3 => {
                let dc = &coeffs[0..3];
                let rest = if coeffs.len() > 3 {
                    &coeffs[3..]
                } else {
                    &[][..]
                };
                (dc, rest)
            }
            Color::Rgb(rgb) => (rgb, &[][..]),
            _ => (&[0.0, 0.0, 0.0][..], &[][..]),
        };
        for c in 0..3 {
            out.dc += (ra_dc[c] - gb_dc[c]).abs() as f64;
        }
        let m = ra_shr.len().min(gb_shr.len());
        for c in 0..m {
            out.shr += (ra_shr[c] - gb_shr[c]).abs() as f64;
        }
    }
    if n > 0 {
        let nf = n as f64;
        out.pos /= nf;
        out.rot /= nf;
        out.opa /= nf;
        out.sca /= nf;
        out.dc /= nf;
        out.shr /= nf;
    }
    out
}

fn cmd_preview(input: &Path, port: u16) -> Result<()> {
    let bind = format!("0.0.0.0:{port}");
    let server =
        tiny_http::Server::http(&bind).map_err(|e| anyhow!("failed to bind {bind}: {e}"))?;
    let shell_path = Path::new("packages/viewer/preview-shell.html");
    let shell = std::fs::read_to_string(shell_path).unwrap_or_else(|_| {
        format!(
            "<!doctype html><meta charset=utf-8><title>Catetus preview</title>\
             <h1>Catetus preview placeholder</h1>\
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
            body = body.replace("{{CATETUS_SRC}}", &src_path.display().to_string());
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
    let cli_exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("catetus"));

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
            "node not found in PATH. Install Node.js 20+ to use `catetus diff`."
        )),
        Err(e) => Err(anyhow::Error::from(e).context("spawning diff helper")),
    }
}

/// Locate the Node.js helper script that drives `catetus diff`.
///
/// Resolution order:
///   1. `$CATETUS_DIFF_HELPER` if set (must exist).
///   2. `tests/visual/scripts/diff-cli.mjs` walking up from the binary.
///   3. `tests/visual/scripts/diff-cli.mjs` walking up from `$CWD`.
fn locate_helper() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("CATETUS_DIFF_HELPER") {
        let path = PathBuf::from(p);
        if !path.exists() {
            return Err(anyhow!(
                "CATETUS_DIFF_HELPER points at non-existent file: {}",
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
        "diff helper not found. Set CATETUS_DIFF_HELPER or run from the Catetus repo root."
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
        "smoke" => catetus_bench::run_smoke()?,
        other => catetus_bench::run_named(other, Path::new("fixtures"))?,
    };
    println!("{}", catetus_bench::to_json(&suite)?);
    Ok(())
}

/* ---------- submit / fidelity / spec-check ---------- */

/// Default Catetus Cloud endpoint. Override with `CATETUS_API_URL`.
const DEFAULT_API_URL: &str = "https://catetus-api.fly.dev";

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

    let api_url = std::env::var("CATETUS_API_URL").unwrap_or_else(|_| DEFAULT_API_URL.into());
    let api_url = api_url.trim_end_matches('/').to_string();
    let api_key = std::env::var("CATETUS_API_KEY").ok();

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
    // through @catetus/viewer. cmd_diff writes report.json to the
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
            // `catetus-khr-validate`. We invoke it via Cargo's standard
            // binary-resolution: when this CLI was installed via
            // `cargo install catetus-cli`, both binaries end up in the
            // same Cargo bin dir. For local dev, the user can override via
            // `CATETUS_KHR_VALIDATE` to point at the workspace target.
            let validator = std::env::var("CATETUS_KHR_VALIDATE")
                .unwrap_or_else(|_| "catetus-khr-validate".to_string());
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
            // workspace as `catetus-usd-validate` and ships alongside
            // this CLI under the same Cargo bin dir. For local dev,
            // `CATETUS_USD_VALIDATE` overrides the lookup path.
            let validator = std::env::var("CATETUS_USD_VALIDATE")
                .unwrap_or_else(|_| "catetus-usd-validate".to_string());
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

fn cmd_fidelity_score(
    candidate: &Path,
    baseline: Option<&Path>,
    mlp_version: &str,
    pretty: bool,
) -> Result<()> {
    let version = match mlp_version {
        "0.4" | "v0.4" | "0.4.0" | "0.4.0-mlp22" => catetus_fidelity::MlpVersion::V04,
        "0.5" | "v0.5" | "0.5.0" | "0.5.0-mlp22" => catetus_fidelity::MlpVersion::V05,
        other => anyhow::bail!("unknown --mlp-version {other:?} (try 0.4 or 0.5)"),
    };
    // ML scoring is hosted-only post the 2026-05-19 open-core split. We
    // still emit the local 22-feature vector so callers can either ship it
    // to api.catetus.com themselves or run their own model on it; the
    // `score` field is null with a `hosted_endpoint` pointer alongside.
    let features = catetus_fidelity::extract_features(candidate, baseline)
        .with_context(|| format!("catetus-fidelity extract {}", version.version_str()))?;
    let envelope = serde_json::json!({
        "kind": "feature_vector_only",
        "version": version.version_str(),
        "hosted_endpoint": catetus_fidelity::HOSTED_FIDELITY_ENDPOINT,
        "features": features.to_vec(),
        "feature_names": catetus_fidelity::FEATURE_NAMES.to_vec(),
        "score": null,
        "note": "ML scoring is hosted-only; POST features + version to hosted_endpoint",
    });
    let json = if pretty {
        serde_json::to_string_pretty(&envelope)?
    } else {
        serde_json::to_string(&envelope)?
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
            std::env::temp_dir().join(format!("catetus-diff-helper-{}.mjs", std::process::id()));
        let mut f = std::fs::File::create(&dir).unwrap();
        writeln!(f, "// stub").unwrap();
        // SAFETY: tests in this crate are not run in parallel with anything
        // else mutating CATETUS_DIFF_HELPER.
        std::env::set_var("CATETUS_DIFF_HELPER", &dir);
        let resolved = locate_helper().expect("env override resolves");
        assert_eq!(resolved, dir);
        std::env::remove_var("CATETUS_DIFF_HELPER");
        let _ = std::fs::remove_file(&dir);
    }
}
