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
3. Uploads the resulting glTF (and any sidecar buffer files it references)
   back to Vercel Blob, rewriting buffer URIs to absolute URLs so the
   manifest is self-contained.
4. POSTs ``{status: "done", output_url}`` to the API's ``callback_url`` so
   the caller can transition their job to ``Done`` without polling.

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
        # Install Rust toolchain (current stable) so we can build the CLI from
        # source against the pinned ref. We need >=1.83 because the modern
        # rustc + std crates use the 2024 edition. ~90 s cold; cached after
        # the first deploy.
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable",
        # Build splatforge at the pinned ref. `rust-toolchain.toml` in-repo
        # also requests stable, so this is consistent with local builds.
        f"git clone --depth 1 --branch {SPLATFORGE_REF} {SPLATFORGE_REPO} /opt/splatforge",
        "/root/.cargo/bin/cargo build --release "
        "--manifest-path /opt/splatforge/Cargo.toml "
        "-p splatforge-cli",
        # Symlink the binary onto PATH.
        "ln -s /opt/splatforge/target/release/splatforge /usr/local/bin/splatforge",
    )
    .pip_install("requests==2.32.3", "fastapi[standard]==0.115.6")
)


app = modal.App("splatforge-worker", image=image)

# A persistent volume the optimize function writes results into so multiple
# jobs can share staging space without ballooning ephemeral disk.
volume = modal.Volume.from_name("splatforge-worker-tmp", create_if_missing=True)

# Vercel Blob HTTPS protocol — matches what apps/api uses on the upload path.
BLOB_HOST = "https://blob.vercel-storage.com"
BLOB_API_VERSION = "7"


# ----------------------------------------------------- optimize core function


@app.function(
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("vercel-blob", required_keys=["BLOB_READ_WRITE_TOKEN"])],
)
def run_optimize(
    job_id: str,
    preset: str,
    blob_url: str,
    filename: str,
    callback_url: Optional[str] = None,
) -> dict:
    """Pull the splat at ``blob_url``, run ``splatforge optimize``, upload the
    result back to Vercel Blob, and POST a callback to the API. Returns the
    same payload that was POSTed so test harnesses can introspect it.
    """
    work = Path(tempfile.mkdtemp(prefix=f"splat-{job_id}-", dir="/data"))
    src_path = work / (filename or "input.bin")
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download the source splat. Streaming so we don't blow memory on
        # bicycle-sized scenes (~860 MB).
        _download(blob_url, src_path)

        # 2. Run splatforge optimize → gltf.
        gltf_path = out_dir / "optimized.gltf"
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
        if rc != 0 or not gltf_path.exists():
            payload = {
                "status": "error",
                "error": _tail(optimize_log, 4096) or f"splatforge exited {rc}",
            }
            return _callback(callback_url, payload)

        # 3. Upload the optimized artifact. The splatforge CLI emits a glTF
        # manifest plus one or more sidecar files under `buffers/`. Upload
        # every sidecar first, capture their final blob URLs, then rewrite
        # the manifest's `buffers[].uri` fields to absolute URLs so the
        # downloaded .gltf is self-contained (no missing-file surprise).
        try:
            output_url = _upload_gltf_bundle(job_id, gltf_path)
        except Exception as exc:  # noqa: BLE001
            return _callback(
                callback_url,
                {"status": "error", "error": f"output upload failed: {exc}"},
            )

        volume.commit()
        return _callback(
            callback_url,
            {"status": "done", "output_url": output_url},
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _download(url: str, dst: Path) -> None:
    """Stream a remote blob to disk in 4 MB chunks."""
    if url.startswith("blob://stub/"):
        raise RuntimeError(
            "blob URL is a stub; configure BLOB_READ_WRITE_TOKEN on the API "
            "before enqueueing real jobs"
        )
    req = urllib.request.Request(url, headers={"User-Agent": "splatforge-worker/0.1.1"})
    with urllib.request.urlopen(req, timeout=300) as response, dst.open("wb") as f:
        shutil.copyfileobj(response, f, length=4 * 1024 * 1024)


def _upload_gltf_bundle(job_id: str, gltf_path: Path) -> str:
    """Upload a glTF manifest + every sidecar buffer it references.

    Returns the final URL of the patched manifest, whose buffer URIs have
    been rewritten to absolute blob URLs. Self-contained: downloading the
    .gltf is enough to render the scene anywhere.
    """
    manifest = json.loads(gltf_path.read_text(encoding="utf-8"))
    base_dir = gltf_path.parent
    buffers = manifest.get("buffers") or []
    for buf in buffers:
        uri = buf.get("uri")
        if not uri or uri.startswith("data:") or uri.startswith("http"):
            continue  # already inline / absolute, nothing to do
        sidecar = (base_dir / uri).resolve()
        if not sidecar.exists():
            raise RuntimeError(f"manifest references missing buffer: {uri}")
        key = f"jobs/{job_id}/{uri}"
        buf["uri"] = _upload_blob(key, sidecar, "application/octet-stream")
    # Write the patched manifest to a sibling file (don't mutate the
    # original on disk — keeps idempotence in case the function retries).
    patched = base_dir / "optimized.patched.gltf"
    patched.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return _upload_blob(f"jobs/{job_id}/optimized.gltf", patched, "model/gltf+json")


def _upload_blob(key: str, path: Path, content_type: str) -> str:
    """PUT ``path`` to Vercel Blob and return the public URL."""
    import requests  # noqa: PLC0415

    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN not set on worker")
    url = f"{BLOB_HOST}/{key.lstrip('/')}?addRandomSuffix=0"
    with path.open("rb") as f:
        resp = requests.put(
            url,
            headers={
                "authorization": f"Bearer {token}",
                "x-content-type": content_type,
                "x-api-version": BLOB_API_VERSION,
            },
            data=f,
            timeout=300,
        )
    if resp.status_code >= 300:
        raise RuntimeError(f"vercel blob PUT {resp.status_code}: {resp.text[:512]}")
    body = resp.json()
    return body["url"]


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


def _callback(callback_url: Optional[str], payload: dict) -> dict:
    """POST the result back to the API. Best-effort: a failed callback is
    logged into the returned payload but doesn't raise — the caller can
    still introspect the spawn future for the result in tests.
    """
    if not callback_url:
        return payload
    try:
        import requests  # noqa: PLC0415

        requests.post(callback_url, json=payload, timeout=15)
    except Exception as exc:  # noqa: BLE001
        payload = {**payload, "_callback_error": str(exc)}
    return payload


# ------------------------------------------------------------- web endpoints


@app.function(image=image, cpu=0.25)
@modal.fastapi_endpoint(method="POST", label="enqueue")
def enqueue(payload: dict) -> dict:
    """Accept a job descriptor from the API and spawn ``run_optimize``.

    Required keys: ``job_id``, ``preset``, ``blob_url``, ``callback_url``.
    ``filename`` is optional (defaults to ``input.bin`` if omitted).
    """
    required = ("job_id", "preset", "blob_url", "callback_url")
    missing = [k for k in required if k not in payload]
    if missing:
        return {"queued": False, "error": f"missing fields: {missing}"}

    run_optimize.spawn(
        job_id=str(payload["job_id"]),
        preset=str(payload["preset"]),
        blob_url=str(payload["blob_url"]),
        filename=str(payload.get("filename") or "input.bin"),
        callback_url=str(payload["callback_url"]),
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
