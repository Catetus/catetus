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

# Hosted-only presets that this worker does NOT run inside its own
# container — the encoders (CodecGS mixed-CRF, FCGS) live in
# private-repo Modal apps because they ship CUDA wheels and (for
# CodecGS) reference research code we aren't open-sourcing yet. The
# public worker proxies the enqueue payload to the relevant Modal
# endpoint when set; otherwise the job errors with a clear message so
# the API surfaces a known state instead of a generic timeout.
#
# Configuration: each URL points at a Modal `fastapi_endpoint` whose
# payload contract MUST match this worker's `enqueue` (job_id, preset,
# blob_url, filename, callback_url). The remote app is responsible for
# downloading the input, encoding, uploading the bitstream, and POSTing
# the terminal `{status, output_url}` back to `callback_url` — same shape
# as `run_optimize` here so the API doesn't need preset-specific result
# handling.
PRESET_DISPATCH_URLS = {
    "codec-gs-mixed": os.environ.get("SPLATFORGE_CODEC_GS_MIXED_URL"),
    "codec-gs-mixed-k5": os.environ.get("SPLATFORGE_CODEC_GS_MIXED_URL"),
    "fcgs-instant": os.environ.get("SPLATFORGE_FCGS_URL"),
    # capture-and-compress lives in the private `splatforge-capture`
    # Modal app: photos.zip → COLMAP (CPU) → 3DGS train (A100) → encode
    # (default inner preset codec-gs-mixed) → upload. Same /enqueue
    # contract as the other forwarded presets; the private app POSTs the
    # terminal `{status, output_url}` back to the API's callback_url
    # directly. URL plumbed at deploy time once the private app exists.
    "capture-and-compress": os.environ.get("SPLATFORGE_CAPTURE_URL"),
    # HAC++ Phase A + lzma passthrough. Scaffold-GS anchor-feature
    # entropy coder validated 2026-05-15: 129.95 MB Scaffold-GS native
    # → 24.21 MB .hacpp container on bonsai (-0.178 dB render-PSNR
    # delta), 11.5× lossless on Inria 3DGS PLY passthrough. Encoder
    # runs on A100 with a 5000-iter hyperprior train pass; budgeted at
    # ~$0.30-$0.60 / scene. Same /enqueue contract; private Modal app
    # POSTs terminal result directly to the API's callback_url.
    "hacpp-lzma": os.environ.get("SPLATFORGE_HACPP_LZMA_URL"),
    # On-demand per-scene neural codec. Bicycle outdoor 7.54× / +8.39 dB
    # ΔPSNR validated N=3 on aaadf09 (research/neural-codec-v0.1-m3).
    # ~120 s encode + ~$0.13 A100-time per scene. The per-scene fit
    # needs supervision images, so the encoder app accepts either a
    # bundle (point_cloud.ply + cameras.json + images/) or a registered
    # Mip-NeRF 360 scene name in `filename`. Same /enqueue contract.
    "hosted-neural": os.environ.get("SPLATFORGE_HOSTED_NEURAL_URL"),
    # splatforge-qat-scaffold — post-training QAT codec for Scaffold-GS
    # PLYs. Public bench numbers (benches/encoders/qat-scaffold-gs,
    # c3387b3, 2026-05-16): aggregate 37.25% PLY-size save across 6
    # Mip-NeRF 360 scenes (bonsai, bicycle, garden, stump, treehill,
    # flowers); +PSNR on every scene (min +0.032 dB, max +0.581 dB,
    # mean +0.172 dB) plus SSIM / LPIPS improvements measured on the
    # same training-time eval cameras. The codec is lossless in the
    # codec sense (deterministic dequant) and the output is a smaller
    # Scaffold-GS PLY. Wall-clock dominated by the Modal GPU pass.
    # Encoder lives in a private Modal app (`splatforge-qat-scaffold`);
    # same /enqueue contract as the other forwarded presets — the
    # private app downloads the input, runs the QAT pass, uploads the
    # compressed PLY, and POSTs the terminal `{status, output_url}`
    # back to `callback_url`.
    "splatforge-qat-scaffold": os.environ.get("SPLATFORGE_QAT_SCAFFOLD_URL"),
    # splatforge-qat-bundle — premium-tier full QAT recipe. Takes a
    # bundle (point_cloud.ply + 3 MLPs + cameras.json + cfg_args +
    # images/) instead of a single PLY because the int8 retrain leg
    # needs GT cameras + images to absorb feature-quant noise. ~$0.50
    # per scene, ~10 min on A100. Same /enqueue contract as the other
    # forwarded presets; the private Modal app POSTs the terminal
    # `{status, output_url, ply_save_pct, delta_psnr_db}` back to
    # `callback_url`. Numbers are honest per-scene — bonsai best-case
    # +0.58 dB / 40.5% save; flowers worst-case +0.03 dB / 33.8% save
    # (see benches/encoders/qat-scaffold-gs).
    "splatforge-qat-bundle": os.environ.get("SPLATFORGE_QAT_BUNDLE_URL"),
}

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


# Modal workspace caps web functions at 8 across all deployed apps. With
# 7 splatforge presets (codec-gs-mixed + fcgs + hacpp-lzma + hosted-neural + qat-scaffold
# + qat-bundle [premium] + this worker) and the personal linecall-machine app, separate
# `enqueue` + `healthz` endpoints push us over. We collapse the worker's
# two HTTP routes into a single ASGI app so both routes share one web
# function slot. The URLs the API + operators consume stay the same shape
# (`<base>/enqueue`, `<base>/healthz`) — only the hostname collapses to
# one `--worker.modal.run` host.
WORKER_ASGI_LABEL = "worker"


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

        # ----------------------------------------------------------------
        # MesonGS++ codec branch: the `mgs-*` presets produce a single
        # `.meson` binary container instead of a glTF + sidecars. The
        # download UX is the same (one URL → one self-contained file),
        # so we skip the glTF / .glb / preview path entirely and just
        # upload the .meson.
        # ----------------------------------------------------------------
        if preset.startswith("mgs-"):
            _post_phase(callback_url, "encoding-meson")
            meson_path = out_dir / "optimized.meson"
            mesonpp_log = work / "mesonpp.log"
            rc = _run_cli_streaming(
                [
                    "splatforge",
                    "mesonpp-encode",
                    str(src_path),
                    "-o",
                    str(meson_path),
                    "--preset",
                    preset,
                ],
                mesonpp_log,
                callback_url,
            )
            if rc != 0 or not meson_path.exists():
                payload = {
                    "status": "error",
                    "error": _tail(mesonpp_log, 4096) or f"mesonpp exited {rc}",
                }
                return _callback(callback_url, payload)
            _post_phase(callback_url, "packaging")
            try:
                output_url = _upload_blob(
                    f"jobs/{job_id}/optimized.meson",
                    meson_path,
                    "application/octet-stream",
                )
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
                    # No preview URL — .meson isn't browser-renderable
                    # yet. The viewer ships its own decoder build out of
                    # `crates/splatforge-meson` via wasm; until that's
                    # wired, the customer downloads the .meson directly.
                    "preview_url": None,
                },
            )

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


def enqueue(payload: dict) -> dict:
    """Accept a job descriptor from the API and spawn ``run_optimize``.

    Required keys: ``job_id``, ``preset``, ``blob_url``, ``callback_url``.
    ``filename`` is optional (defaults to ``input.bin`` if omitted).

    For presets registered in :data:`PRESET_DISPATCH_URLS` whose URL is
    configured at deploy time, the payload is forwarded synchronously to
    the dedicated Modal app (CodecGS mixed-CRF or FCGS) and we return its
    ack. Forwarding preserves the original ``callback_url`` so the
    private app POSTs the terminal result straight to the API; the public
    worker never sees the bytes.
    """
    required = ("job_id", "preset", "blob_url", "callback_url")
    missing = [k for k in required if k not in payload]
    if missing:
        return {"queued": False, "error": f"missing fields: {missing}"}

    preset = str(payload["preset"])
    if preset in PRESET_DISPATCH_URLS:
        target = PRESET_DISPATCH_URLS[preset]
        if not target:
            # Surface configuration gap synchronously so the API marks
            # the job Error with a clear message instead of waiting on
            # a callback that will never arrive. Name the specific env
            # var that the operator needs to set for this preset so the
            # message is actionable in the API log.
            env_var = _expected_env_var_for_preset(preset)
            return {
                "queued": False,
                "error": (
                    f"preset '{preset}' requires a dedicated Modal endpoint "
                    f"({env_var}) but none is configured on this worker"
                ),
            }
        return _forward_to_preset_app(target, payload)

    run_optimize.spawn(
        job_id=str(payload["job_id"]),
        preset=preset,
        blob_url=str(payload["blob_url"]),
        filename=str(payload.get("filename") or "input.bin"),
        callback_url=str(payload["callback_url"]),
    )
    return {"queued": True, "error": None}


def _expected_env_var_for_preset(preset: str) -> str:
    """Map a dispatched preset to the env var an operator sets to wire
    its private Modal endpoint. Used only by the "URL missing" error
    path so the message can name the specific knob to flip. Keep this
    table aligned with :data:`PRESET_DISPATCH_URLS` above.
    """
    mapping = {
        "codec-gs-mixed": "SPLATFORGE_CODEC_GS_MIXED_URL",
        "codec-gs-mixed-k5": "SPLATFORGE_CODEC_GS_MIXED_URL",
        "fcgs-instant": "SPLATFORGE_FCGS_URL",
        "capture-and-compress": "SPLATFORGE_CAPTURE_URL",
        "hacpp-lzma": "SPLATFORGE_HACPP_LZMA_URL",
        "hosted-neural": "SPLATFORGE_HOSTED_NEURAL_URL",
        "splatforge-qat-scaffold": "SPLATFORGE_QAT_SCAFFOLD_URL",
        "splatforge-qat-bundle": "SPLATFORGE_QAT_BUNDLE_URL",
    }
    return mapping.get(preset, "SPLATFORGE_<PRESET>_URL")


def _forward_to_preset_app(target_url: str, payload: dict) -> dict:
    """POST the enqueue payload to a preset-specific Modal app and
    return its ack verbatim.

    The downstream Modal app implements the same ``/enqueue`` contract
    this worker does — it must accept ``{job_id, preset, blob_url,
    filename, callback_url}`` and respond with ``{queued, error}``. By
    forwarding the original ``callback_url`` we let the private encoder
    app talk straight to the API for status updates, keeping the public
    worker out of the data path entirely.

    Failure modes (HTTP 5xx, JSON parse error, network timeout) are
    folded into a ``{queued: False, error}`` so the API can mark the
    job Error and bubble a useful message to the user.
    """
    import requests  # noqa: PLC0415

    body = {
        "job_id": str(payload["job_id"]),
        "preset": str(payload["preset"]),
        "blob_url": str(payload["blob_url"]),
        "filename": str(payload.get("filename") or "input.bin"),
        "callback_url": str(payload["callback_url"]),
    }
    try:
        resp = requests.post(target_url, json=body, timeout=30)
    except Exception as exc:  # noqa: BLE001
        return {
            "queued": False,
            "error": f"forward to preset app failed: {exc}",
        }
    if resp.status_code >= 300:
        return {
            "queued": False,
            "error": f"preset app rejected enqueue ({resp.status_code}): {resp.text[:512]}",
        }
    try:
        ack = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {
            "queued": False,
            "error": f"preset app returned non-JSON ack: {exc}",
        }
    # Defensive defaults: a misbehaving downstream that omits `queued`
    # is treated as a successful queue (HTTP 2xx already gated us here)
    # but errors are forwarded if present.
    return {
        "queued": bool(ack.get("queued", True)),
        "error": ack.get("error"),
    }


def healthz() -> dict:
    return {
        "ok": True,
        "service": "splatforge-worker",
        "splatforge_ref": SPLATFORGE_REF,
        # Surface which preset-specific Modal endpoints this deploy has
        # been wired to. We never echo the URL (it's a secret webhook);
        # just a bool per preset so operators can verify the deploy
        # without `modal secret list`.
        "preset_dispatch_configured": {
            name: bool(url) for name, url in PRESET_DISPATCH_URLS.items()
        },
    }


# Single ASGI app fronting both routes — collapses two `fastapi_endpoint`
# decorators into one Modal web function so the workspace stays under the
# 8 web-function cap. See WORKER_ASGI_LABEL above for the rationale.
@app.function(
    image=image,
    cpu=0.25,
    secrets=[
        # Same secret injection both routes consumed before; PRESET_DISPATCH_URLS
        # is built at module load from these env vars, so the secret has to be
        # attached to the function that serves both routes. Keep required_keys
        # aligned with PRESET_DISPATCH_URLS above (and the analogous list in
        # the dispatch-table tests).
        modal.Secret.from_name(
            "splatforge-preset-urls",
            required_keys=[
                "SPLATFORGE_CODEC_GS_MIXED_URL",
                "SPLATFORGE_FCGS_URL",
                "SPLATFORGE_CAPTURE_URL",
                "SPLATFORGE_HACPP_LZMA_URL",
                "SPLATFORGE_HOSTED_NEURAL_URL",
                "SPLATFORGE_QAT_SCAFFOLD_URL",
                "SPLATFORGE_QAT_BUNDLE_URL",
            ],
        )
    ],
)
@modal.asgi_app(label=WORKER_ASGI_LABEL)
def web_app():
    from fastapi import FastAPI  # noqa: PLC0415

    api = FastAPI(title="splatforge-worker")

    @api.post("/enqueue")
    def _enqueue(payload: dict) -> dict:
        return enqueue(payload)

    @api.get("/healthz")
    def _healthz() -> dict:
        return healthz()

    return api
