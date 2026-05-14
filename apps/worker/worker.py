"""
SplatForge Modal worker.

Single Modal app exposing two endpoints:

- ``POST /enqueue``   spawn an optimize job. Returns immediately after
  scheduling the underlying ``run_optimize`` Modal Function. Synchronous
  body, asynchronous side effect.
- ``GET  /healthz``   liveness check.

Each optimize invocation:

1. Pulls the splat blob from a URL the API hands us (Vercel Blob today).
2. Runs the pinned ``splatforge`` CLI inside this container's filesystem.
3. POSTs the resulting SPZ + glTF + report bytes back to the API's status
   webhook so the caller can transition their job to ``Succeeded``.

Cost model: the optimize container is single-CPU and scales to zero. A typical
1M-splat optimize takes ~30 s; at the published Modal rates that's well under
$0.01 per invocation. The image pins the splatforge CLI by git tag so we
don't drift mid-flight.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------- image

# Pin both the base image and the splatforge CLI revision so the worker is
# byte-reproducible from one deploy to the next. Bump SPLATFORGE_REF when a new
# CLI release lands.
SPLATFORGE_REF = os.environ.get("SPLATFORGE_REF", "v0.1.1")
SPLATFORGE_REPO = os.environ.get(
    "SPLATFORGE_REPO", "https://github.com/montabano1/SplatForge.git"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "git", "pkg-config", "ca-certificates")
    .run_commands(
        # Install Rust toolchain (release-channel stable) so we can build the
        # CLI from source against the pinned ref. ~90 s cold; cached after
        # the first deploy.
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain 1.74.0",
        # Build splatforge at the pinned ref.
        f"git clone --depth 1 --branch {SPLATFORGE_REF} {SPLATFORGE_REPO} /opt/splatforge",
        "/root/.cargo/bin/cargo build --release "
        "--manifest-path /opt/splatforge/Cargo.toml "
        "-p splatforge-cli",
        # Symlink the binary onto PATH.
        "ln -s /opt/splatforge/target/release/splatforge /usr/local/bin/splatforge",
    )
    .pip_install("requests==2.32.3")
)


app = modal.App("splatforge-worker", image=image)

# A persistent volume the optimize function writes results into so multiple
# jobs can share staging space without ballooning ephemeral disk.
volume = modal.Volume.from_name("splatforge-worker-tmp", create_if_missing=True)


# ----------------------------------------------------- optimize core function


@app.function(
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/data": volume},
)
def run_optimize(
    job_id: str,
    preset: str,
    blob_url: str,
    filename: str,
    api_status_url: Optional[str] = None,
) -> dict:
    """Pull the splat at ``blob_url``, run ``splatforge optimize``, return
    a result dict the caller can store directly on the Job. Also POSTs the
    result back to ``api_status_url`` (when set) so the API can mark the job
    Succeeded / Failed without polling.
    """
    work = Path(tempfile.mkdtemp(prefix=f"splat-{job_id}-", dir="/data"))
    src_path = work / filename
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download the source splat. Streaming so we don't blow memory on
    # bicycle-sized scenes (~860 MB).
    _download(blob_url, src_path)

    bytes_in = src_path.stat().st_size

    # 2. Run splatforge optimize → gltf, then convert that gltf → spz so the
    # caller has the streamable artifact + the canonical glTF side-by-side.
    gltf_path = out_dir / "scene.gltf"
    optimize_log = work / "optimize.log"
    rc = _run_cli(
        [
            "splatforge",
            "optimize",
            str(src_path),
            "--preset",
            preset,
            "--out",
            str(gltf_path),
        ],
        optimize_log,
    )
    if rc != 0:
        return _ship(
            api_status_url,
            job_id,
            {
                "status": "failed",
                "error": _tail(optimize_log, 4096),
            },
        )

    spz_path = out_dir / "scene.spz"
    convert_log = work / "convert.log"
    rc = _run_cli(
        [
            "splatforge",
            "convert",
            str(gltf_path),
            "--to",
            "spz",
            "--out",
            str(spz_path),
        ],
        convert_log,
    )
    if rc != 0:
        return _ship(
            api_status_url,
            job_id,
            {
                "status": "failed",
                "error": _tail(convert_log, 4096),
            },
        )

    bytes_out = spz_path.stat().st_size
    ratio = float(bytes_in) / float(max(1, bytes_out))

    # 3. Read the analyze report so the API has rich metadata to return.
    report_path = gltf_path.with_suffix(".json")
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception as exc:  # noqa: BLE001
            report = {"_report_parse_error": str(exc)}

    result = {
        "status": "succeeded",
        "ratio": ratio,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "report": report,
        # The API decides how to surface the actual artifact URLs (probably
        # re-upload to Vercel Blob with a public ACL). For v0 we hand back
        # the worker-local paths so the API can stream them down or proxy.
        "spz_path": str(spz_path),
        "gltf_path": str(gltf_path),
        "report_path": str(report_path) if report_path.exists() else None,
    }
    volume.commit()
    return _ship(api_status_url, job_id, result)


def _download(url: str, dst: Path) -> None:
    """Stream a remote blob to disk in 4 MB chunks."""
    if url.startswith("blob://stub/"):
        # API is running in stub mode — for dev we expect a sibling local
        # fixture sharing the same basename.
        raise RuntimeError(
            "blob URL is a stub; configure BLOB_READ_WRITE_TOKEN on the API "
            "before enqueueing real jobs"
        )
    req = urllib.request.Request(url, headers={"User-Agent": "splatforge-worker/0.1.1"})
    with urllib.request.urlopen(req, timeout=60) as response, dst.open("wb") as f:
        shutil.copyfileobj(response, f, length=4 * 1024 * 1024)


def _run_cli(argv: list[str], log_path: Path) -> int:
    """Spawn the pinned splatforge CLI; capture stdout+stderr to ``log_path``."""
    with log_path.open("wb") as log:
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def _tail(path: Path, n: int) -> str:
    """Return the last ``n`` bytes of a log file as text (best-effort)."""
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()[-n:]
        return data.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"<could not read log: {exc}>"


def _ship(api_status_url: Optional[str], job_id: str, result: dict) -> dict:
    """POST the result back to the API's status webhook if configured.

    Returning the result alongside is intentional — Modal's ``.spawn()`` caller
    can poll ``Future.get()`` for the same payload when running synchronously
    in tests or dev.
    """
    if not api_status_url:
        return result
    try:
        import requests  # noqa: PLC0415

        requests.post(
            f"{api_status_url.rstrip('/')}/v1/jobs/{job_id}/status",
            json=result,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        result["_webhook_error"] = str(exc)
    return result


# ------------------------------------------------------------- web endpoints


@app.function(image=image, cpu=0.25)
@modal.fastapi_endpoint(method="POST", label="enqueue")
def enqueue(payload: dict) -> dict:
    """Accept a job descriptor from the API and spawn ``run_optimize``."""
    required = ("job_id", "preset", "blob_url", "filename")
    missing = [k for k in required if k not in payload]
    if missing:
        return {"queued": False, "error": f"missing fields: {missing}"}

    api_status_url = os.environ.get("SPLATFORGE_API_URL") or payload.get("api_status_url")

    # ``spawn`` returns immediately; the underlying Modal Function gets its
    # own container and runs to completion in the background.
    run_optimize.spawn(
        job_id=str(payload["job_id"]),
        preset=str(payload["preset"]),
        blob_url=str(payload["blob_url"]),
        filename=str(payload["filename"]),
        api_status_url=api_status_url,
    )
    return {"queued": True, "error": None}


@app.function(image=image, cpu=0.25)
@modal.fastapi_endpoint(method="GET", label="healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "service": "splatforge-worker",
        "splatforge_ref": SPLATFORGE_REF,
    }
