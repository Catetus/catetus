"""
GPU SplatBench fidelity rerun.

The committed `benches/reports/fidelity-v0.json` was captured with SwiftShader
(CPU-rasterized WebGL2) on macOS. This Modal app reruns the same fidelity
script on a real NVIDIA T4 with Linux Chromium + hardware-accelerated ANGLE,
producing `benches/reports/fidelity-v0-hwaccel.json` for cross-checking.

The result JSON is committed back to the repo via the `--output` flag to
`modal run`. Cost: ~$0.60 per full corpus rerun on T4.

Usage:
    python3 -m modal run apps/fidelity-gpu/run.py::run_corpus

The image bundles:
  - NVIDIA CUDA 12.4 runtime (gives us libEGL_nvidia, libnvidia-vulkan-loader)
  - Chromium + Playwright deps (apt-installed; Playwright's bundled Chromium
    is downloaded at runtime)
  - Rust stable + catetus CLI built from the pinned git tag
  - Node 20 + the fidelity-runner deps (pngjs, pixelmatch, playwright-core)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal

CATETUS_REF = os.environ.get("CATETUS_REF", "v0.1.1")
CATETUS_REPO = os.environ.get(
    "CATETUS_REPO", "https://github.com/Catetus/catetus.git"
)

# NVIDIA CUDA runtime base — gives us libEGL_nvidia.so + libGLX_nvidia.so so
# Chromium's ANGLE → Vulkan/GLES pipeline can talk to a real GPU instead of
# SwiftShader.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        # Build chain for catetus.
        "build-essential", "git", "curl", "pkg-config", "ca-certificates",
        # Chromium runtime deps (Playwright will install Chrome itself).
        "libnss3", "libatk1.0-0", "libatk-bridge2.0-0", "libcups2",
        "libxkbcommon0", "libxcomposite1", "libxdamage1", "libxrandr2",
        "libgbm1", "libxss1", "libasound2", "libxshmfence1", "libdrm2",
        "libpango-1.0-0", "libpangocairo-1.0-0", "libcairo2",
        "fonts-liberation", "xvfb",
        # Vulkan loader + Mesa fallback (for `vulkaninfo` to work as a probe).
        # NOTE: NVIDIA Vulkan ICD needs `libnvidia-gl-XXX` from the host driver
        # bundle; that's only present when Modal mounts the right GPU driver
        # at runtime. We install the userland loader here and trust the GPU
        # mount to supply the ICD JSON pointer at /usr/share/vulkan/icd.d/.
        "libvulkan1", "vulkan-tools", "mesa-vulkan-drivers",
        # Node 20 (used by the fidelity runner).
        "gnupg",
    )
    .run_commands(
        # Node 20 from NodeSource.
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g pnpm@9",
        # Rust toolchain (matches rust-toolchain.toml — stable channel).
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable",
        # Catetus clone + build (CLI release).
        f"git clone --depth 1 --branch {CATETUS_REF} {CATETUS_REPO} /opt/catetus",
        "/root/.cargo/bin/cargo build --release "
        "--manifest-path /opt/catetus/Cargo.toml -p catetus-cli",
        "ln -s /opt/catetus/target/release/catetus /usr/local/bin/catetus",
        # Install ALL workspace JS deps so we can build the viewer dist
        # (the harness page imports /viewer/index.js, which the fidelity
        # script's in-process server resolves to packages/viewer/dist).
        "cd /opt/catetus && pnpm install --frozen-lockfile=false",
        "cd /opt/catetus && pnpm -F @catetus/viewer run build",
        "cd /opt/catetus && pnpm -F @catetus/report-ui run build",
        # Pull Playwright's Chromium so we don't depend on the apt one.
        "cd /opt/catetus/tests/visual && "
        "pnpm exec playwright install chromium --with-deps",
        # Pre-generate the synthetic scenes so the runtime path is fast.
        "mkdir -p /sbench/scenes && "
        "python3 /opt/catetus/benches/synth_scenes.py /sbench/scenes",
        # Pull the real Mip-NeRF360 anchors (~1.13 GB).
        "curl -L -o /sbench/scenes/bonsai.ply "
        "  https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/point_cloud/iteration_7000/point_cloud.ply",
        "curl -L -o /sbench/scenes/bicycle.ply "
        "  https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bicycle/point_cloud/iteration_7000/point_cloud.ply",
    )
)

app = modal.App("catetus-fidelity-gpu", image=image)

# Persist results outside the container so we can pull them back from outside.
results_volume = modal.Volume.from_name(
    "catetus-fidelity-results", create_if_missing=True
)


@app.function(
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={"/results": results_volume},
)
def run_corpus(skip_bicycle: bool = False, gpu_mode: str = "auto") -> dict:
    """
    gpu_mode:
      "vulkan" — Chromium uses ANGLE→Vulkan, talks to the NVIDIA ICD if present
      "swiftshader" — force CPU rasterizer (useful as a Linux-vs-Mac control)
      "auto" — probe vulkaninfo at startup and pick the best option
    """
    """Run the full SplatBench corpus through the existing fidelity script
    on a real T4 GPU. Writes the result JSON to /results/fidelity-v0-hwaccel.json
    (mounted volume) so the caller can pull it down with `modal volume get`.
    """
    repo = Path("/opt/catetus")
    scenes_dir = Path("/sbench/scenes")

    # The fidelity script reads from /tmp/sbench/scenes by default; symlink
    # the pre-baked scenes there so we don't move ~1.5 GB around.
    tmp_scenes = Path("/tmp/sbench/scenes")
    tmp_scenes.parent.mkdir(parents=True, exist_ok=True)
    if tmp_scenes.exists():
        if tmp_scenes.is_symlink():
            tmp_scenes.unlink()
        else:
            shutil.rmtree(tmp_scenes)
    tmp_scenes.symlink_to(scenes_dir)

    # Probe what GPU we actually got (logs into the result for transparency).
    gpu_probe = _gpu_probe()

    # Pick the right Chromium GL backend based on what's actually available.
    has_nvidia_icd = "NVIDIA" in gpu_probe.get("vulkaninfo", "")
    effective_mode = gpu_mode
    if gpu_mode == "auto":
        effective_mode = "vulkan" if has_nvidia_icd else "swiftshader"

    env = os.environ.copy()
    env["SBENCH_RENDERER"] = "webgl2"
    env["SBENCH_RENDER_TIMEOUT_MS"] = "1800000"
    if effective_mode == "vulkan":
        env["SBENCH_CHROME_FLAGS"] = (
            "--use-gl=angle --use-angle=vulkan "
            "--enable-features=Vulkan,UseSkiaRenderer "
            "--disable-software-rasterizer "
            "--ignore-gpu-blocklist "
            "--no-sandbox"
        )
    else:
        # SwiftShader fallback — Chromium's built-in software path so the
        # script still runs end-to-end on a Linux container without an
        # NVIDIA Vulkan ICD. Useful as a Linux-vs-Mac SwiftShader control.
        env["SBENCH_CHROME_FLAGS"] = "--enable-unsafe-swiftshader --no-sandbox"

    if skip_bicycle:
        env["SBENCH_SCENES"] = ",".join(
            [
                "bonsai_mipnerf360_iter7k",
                "splatbench_product_proxy",
                "splatbench_indoor_proxy",
                "splatbench_floater_proxy",
                "splatbench_outdoor_proxy",
                "splatbench_dense_proxy",
            ]
        )

    # Run the existing fidelity script. It writes to
    # `benches/reports/fidelity-v0.json` relative to repo root — we redirect
    # the output afterward.
    cmd = [
        "node",
        "--max-old-space-size=12288",
        str(repo / "tests/visual/scripts/splatbench-fidelity.mjs"),
    ]
    log_path = Path("/results/run.log")
    with log_path.open("wb") as log:
        proc = subprocess.run(cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT, env=env, check=False)

    src_json = repo / "benches/reports/fidelity-v0.json"
    dest_json = Path("/results/fidelity-v0-hwaccel.json")
    if src_json.exists():
        data = json.loads(src_json.read_text())
        # Stamp the GPU + renderer choice so the JSON is self-describing.
        data["renderer"] = f"webgl2-{effective_mode}"
        data["hwaccel_probe"] = gpu_probe
        data["modal_gpu"] = "T4"
        data["gpu_mode"] = effective_mode
        dest_json.write_text(json.dumps(data, indent=2) + "\n")
        results_volume.commit()
        return {
            "ok": True,
            "exit_code": proc.returncode,
            "scenes": len(data.get("scenes", [])),
            "errors": data.get("errors", []),
            "gpu_probe": gpu_probe,
            "gpu_mode": effective_mode,
            "result_path": str(dest_json),
        }
    return {
        "ok": False,
        "exit_code": proc.returncode,
        "gpu_probe": gpu_probe,
        "log_tail": _tail(log_path, 4096),
    }


def _gpu_probe() -> dict:
    """Capture what GPU Chromium would see — useful for the result JSON so
    a reader can verify they're looking at hardware-accel numbers rather
    than another SwiftShader run."""
    probe: dict = {}
    for name, cmd in (
        ("nvidia_smi", ["nvidia-smi", "-L"]),
        ("vulkaninfo", ["vulkaninfo", "--summary"]),
        ("glxinfo", ["glxinfo", "-B"]),
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            probe[name] = (out.stdout + out.stderr).strip()[:2000]
        except Exception as exc:  # noqa: BLE001
            probe[name] = f"<probe failed: {exc}>"
    return probe


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_bytes()[-n:].decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"<could not read log: {exc}>"


@app.local_entrypoint()
def main(skip_bicycle: bool = False, gpu_mode: str = "auto") -> None:
    """Local entrypoint — runs the corpus on Modal, prints the result summary.
    The committed JSON is pulled separately via `modal volume get`.
    """
    result = run_corpus.remote(skip_bicycle=skip_bicycle, gpu_mode=gpu_mode)
    print(json.dumps(result, indent=2)[:8000])
