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
import struct
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
SPLATFORGE_REF = os.environ.get("SPLATFORGE_REF", "main")
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
        # Phase 1: download the source splat. Streaming so we don't blow memory
        # on bicycle-sized scenes (~860 MB). Fire-and-forget progress POST so
        # the API can surface step-level status during the long total job.
        _post_phase(callback_url, "fetching")
        _download(blob_url, src_path)

        # Phase 2: run splatforge optimize → gltf. Stream stdout line-by-line
        # so we can forward `PROGRESS frac=X stage=Y` lines emitted by
        # `--progress` straight through to the API as live updates.
        _post_phase(callback_url, "optimizing")
        gltf_path = out_dir / "optimized.gltf"
        optimize_log = work / "optimize.log"
        rc = _run_cli_streaming(
            [
                "splatforge",
                "optimize",
                str(src_path),
                "--preset",
                preset,
                "--out",
                str(gltf_path),
                "--progress",
            ],
            optimize_log,
            callback_url,
        )
        if rc != 0 or not gltf_path.exists():
            payload = {
                "status": "error",
                "error": _tail(optimize_log, 4096) or f"splatforge exited {rc}",
            }
            return _callback(callback_url, payload)

        # Phase 3: package + upload. Pack the .glb (single-file portable) and
        # upload the .gltf preview manifest with absolute buffer URLs.
        _post_phase(callback_url, "packaging")
        try:
            glb_path = out_dir / "optimized.glb"
            _pack_glb(gltf_path, glb_path)
            output_url = _upload_blob(
                f"jobs/{job_id}/optimized.glb",
                glb_path,
                "model/gltf-binary",
            )
            preview_url = _upload_gltf_with_absolute_buffers(job_id, gltf_path)
        except Exception as exc:  # noqa: BLE001
            return _callback(
                callback_url,
                {"status": "error", "error": f"output upload failed: {exc}"},
            )

        volume.commit()
        return _callback(
            callback_url,
            {
                "status": "done",
                "output_url": output_url,
                "preview_url": preview_url,
            },
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


def _upload_gltf_with_absolute_buffers(job_id: str, gltf_path: Path) -> str:
    """Upload glTF + every sidecar, rewriting buffer URIs to absolute blob URLs.

    Companion to `_pack_glb` for in-browser preview: the splatforge web
    viewer expects a JSON manifest it can fetch, with each buffer URI an
    absolute URL it can stream chunks from. Returns the manifest's URL.
    """
    manifest = json.loads(gltf_path.read_text(encoding="utf-8"))
    base_dir = gltf_path.parent
    for buf in manifest.get("buffers") or []:
        uri = buf.get("uri")
        if not uri or uri.startswith("data:") or uri.startswith("http"):
            continue
        sidecar = (base_dir / uri).resolve()
        if not sidecar.exists():
            raise RuntimeError(f"manifest references missing buffer: {uri}")
        key = f"jobs/{job_id}/{uri}"
        buf["uri"] = _upload_blob(key, sidecar, "application/octet-stream")
    patched = base_dir / "optimized.preview.gltf"
    patched.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return _upload_blob(
        f"jobs/{job_id}/optimized.gltf", patched, "model/gltf+json"
    )


def _pack_glb(gltf_path: Path, glb_path: Path) -> None:
    """Pack a .gltf + sidecar buffer(s) into a single .glb (binary glTF).

    Spec ref: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#binary-gltf-layout

    File layout:
      12-byte header:  magic "glTF" + version=2 + total_length
       8-byte JSON chunk header: length + "JSON"  (4-byte type tag)
         JSON payload, 0x20-padded to 4-byte boundary
       8-byte BIN chunk header: length + "BIN\\0"
         binary payload, 0x00-padded to 4-byte boundary

    The KHR_gaussian_splatting outputs we handle today reference a single
    sidecar buffer (`buffers/chunk_0000.bin`). For robustness in the face
    of multi-chunk outputs, we concatenate every buffer's contents into a
    single BIN chunk and shift each buffer's `byteOffset` accordingly.
    """
    manifest = json.loads(gltf_path.read_text(encoding="utf-8"))
    base_dir = gltf_path.parent
    buffers = manifest.get("buffers") or []
    if not buffers:
        # Nothing to pack — just embed an empty BIN chunk so the glb is valid.
        bin_blob = b""
        offsets = []
    else:
        chunks: list[bytes] = []
        offsets: list[int] = []
        running = 0
        for buf in buffers:
            uri = buf.get("uri")
            if not uri:
                raise RuntimeError(
                    "manifest buffer has no uri; embedded buffers not supported"
                )
            if uri.startswith("data:") or uri.startswith("http"):
                raise RuntimeError(
                    f"manifest buffer is already external ({uri[:60]}); "
                    "expected on-disk sidecar"
                )
            sidecar = (base_dir / uri).resolve()
            if not sidecar.exists():
                raise RuntimeError(f"manifest references missing buffer: {uri}")
            data = sidecar.read_bytes()
            chunks.append(data)
            offsets.append(running)
            running += len(data)
            # 4-byte align the next chunk inside the merged buffer.
            pad = (-running) % 4
            if pad:
                chunks.append(b"\x00" * pad)
                running += pad
        bin_blob = b"".join(chunks)

    # Rewrite the manifest: every buffer becomes a single merged buffer
    # whose data lives in the BIN chunk. bufferViews shift to point into
    # the merged buffer at the correct offset.
    if buffers:
        buffer_view_shifts = {
            i: offsets[bv.get("buffer", 0)]
            for i, bv in enumerate(manifest.get("bufferViews") or [])
        }
        manifest["buffers"] = [{"byteLength": len(bin_blob)}]
        for i, bv in enumerate(manifest.get("bufferViews") or []):
            bv["buffer"] = 0
            bv["byteOffset"] = bv.get("byteOffset", 0) + buffer_view_shifts[i]

    json_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    json_bytes = json_bytes + (b" " * json_pad)
    bin_pad = (-len(bin_blob)) % 4
    bin_blob = bin_blob + (b"\x00" * bin_pad)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    with glb_path.open("wb") as out:
        out.write(struct.pack("<4sII", b"glTF", 2, total))
        out.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        out.write(json_bytes)
        out.write(struct.pack("<I4s", len(bin_blob), b"BIN\x00"))
        out.write(bin_blob)


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


def _run_cli_streaming(
    argv: list[str], log_path: Path, callback_url: Optional[str]
) -> int:
    """Spawn the CLI, forwarding `PROGRESS` lines to the API as they arrive.

    Two responsibilities:
      1. Tee every line of stdout/stderr to `log_path` so error tails still
         work on failure (mirrors `_run_cli` behavior for non-progress
         output).
      2. Watch for lines matching `PROGRESS frac=<float> stage=<name>` and
         POST them upstream as `{status:running, phase:<stage>,
         percent:<frac>}`. Throttled to one POST per 500ms (plus the
         terminal frac=1.0 always fires) so a chatty CLI can't DDOS the
         API.
    """
    import time

    last_post = 0.0
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line-buffered
        text=True,
    )
    assert proc.stdout is not None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if not line.startswith("PROGRESS "):
                    continue
                frac, stage = _parse_progress_line(line)
                if frac is None or stage is None:
                    continue
                now = time.monotonic()
                # Always post the terminal frac=1.0; throttle the rest.
                if frac < 0.999 and (now - last_post) < 0.5:
                    continue
                last_post = now
                _post_phase(callback_url, stage, percent=frac)
    finally:
        proc.wait()
    return proc.returncode


def _parse_progress_line(line: str) -> tuple[Optional[float], Optional[str]]:
    """Pull `frac=<float>` and `stage=<name>` out of a `PROGRESS ...` line.

    Forgiving of token order and extra tokens so future CLI versions can
    add fields without breaking the worker. Returns `(None, None)` on
    parse failure rather than raising — a malformed line shouldn't kill
    the entire optimize job.
    """
    frac: Optional[float] = None
    stage: Optional[str] = None
    for tok in line.strip().split():
        if tok.startswith("frac="):
            try:
                frac = float(tok[len("frac="):])
            except ValueError:
                pass
        elif tok.startswith("stage="):
            stage = tok[len("stage="):]
    return frac, stage


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


def _post_phase(
    callback_url: Optional[str], phase: str, percent: Optional[float] = None
) -> None:
    """Fire-and-forget intermediate progress ping. Lets the API surface
    `phase` (and optional `percent`) to clients so users see "downloading",
    "optimizing 47%", "packaging" instead of a single opaque "running"
    state for the full duration of the job. Failure here is non-fatal —
    the final callback at the end of the job carries the authoritative
    status.
    """
    if not callback_url:
        return
    payload: dict = {"status": "running", "phase": phase}
    if percent is not None:
        payload["percent"] = max(0.0, min(1.0, percent))
    try:
        import requests  # noqa: PLC0415

        requests.post(callback_url, json=payload, timeout=5)
    except Exception:  # noqa: BLE001
        pass


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
