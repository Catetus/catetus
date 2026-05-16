"""Modal A100 wrapper around tasks/scripts/codec-gs-sanity/psnr_v2.py.

Provides two functions for the 2026-05-15 picker P2 audit re-validation
(tasks/scripts/p2-audit-v2/run_audit.py):

  audit_pair(label, scene, source_url, decoded_url)
    Pure render-PSNR harness. Takes two PLY URLs (Inria 3DGS format),
    runs psnr_v2 on A100, returns the JSON it would have written.

  audit_in_process(preset, scene, source_url)
    Full pipeline for in-process Rust presets (web-mobile, size-min):
    downloads source PLY, runs `splatforge optimize --preset <p>` then
    `splatforge convert --to ply`, then runs psnr_v2 on the result.

The container image carries:
  * gsplat 1.5.3 + torch 2.1.2 + CUDA 12.1 (matches psnr_v2 import deps)
  * splatforge CLI built from the pinned SplatForge ref (so optimize +
    convert produce the same bytes the production worker would).
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import modal

# Vendor in psnr_v2 verbatim so the Modal container doesn't need the
# splatforge repo at runtime (the splatforge clone in the image is only
# for building the CLI; not used for psnr_v2).
# Only resolve + assert when running locally (image-build time). Inside
# the container __file__ is rooted at /root/ and codec-gs-sanity doesn't
# exist — the file's already been baked in at /opt/psnr_v2.py.
PSNR_V2_SRC = Path(__file__).resolve().parents[1] / "codec-gs-sanity" / "psnr_v2.py"
if not modal.is_local():
    pass  # container path: /opt/psnr_v2.py (added via add_local_file)
elif not PSNR_V2_SRC.exists():
    raise FileNotFoundError(f"psnr_v2.py not found at {PSNR_V2_SRC}")

SPLATFORGE_REF = "main"
SPLATFORGE_REPO = "https://github.com/montabano1/SplatForge.git"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "build-essential", "ninja-build", "ca-certificates",
        "wget", "curl", "pkg-config",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
    })
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",
        "plyfile==1.0.3",
        "gsplat==1.5.3",
        "requests==2.32.3",
        "packaging==24.2",
    )
    # Build splatforge CLI from source. Same approach as apps/worker/worker.py
    # so the optimize bytes here are identical to production.
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable",
        f"git clone --depth 1 --branch {SPLATFORGE_REF} {SPLATFORGE_REPO} /opt/splatforge",
        "/root/.cargo/bin/cargo build --release "
        "--manifest-path /opt/splatforge/Cargo.toml "
        "-p splatforge-cli",
        "ln -s /opt/splatforge/target/release/splatforge /usr/local/bin/splatforge",
    )
    .add_local_file(str(PSNR_V2_SRC), "/opt/psnr_v2.py", copy=True)
)

app = modal.App("p2-audit-psnr-v2", image=image)

# Vercel Blob upload helper — lets the local driver stage decoded PLYs
# at a public URL without needing BLOB_READ_WRITE_TOKEN in the local env.
BLOB_HOST = "https://blob.vercel-storage.com"
BLOB_API_VERSION = "7"


@app.function(
    cpu=0.5, memory=1024, timeout=600,
    secrets=[modal.Secret.from_name(
        "vercel-blob", required_keys=["BLOB_READ_WRITE_TOKEN"])],
)
def upload_decoded_to_blob(key: str, content_type: str, body_bytes: bytes) -> str:
    """Upload `body_bytes` to Vercel Blob under `key`. Returns the public URL."""
    import requests
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN not set on audit upload function")
    url = f"{BLOB_HOST}/{key.lstrip('/')}?addRandomSuffix=0"
    resp = requests.put(
        url,
        headers={
            "authorization": f"Bearer {token}",
            "x-content-type": content_type,
            "x-api-version": BLOB_API_VERSION,
        },
        data=body_bytes,
        timeout=900,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"blob PUT {resp.status_code}: {resp.text[:512]}")
    return resp.json()["url"]


def _dl(url: str, dst: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "p2-audit/1.0"})
    with urllib.request.urlopen(req, timeout=900) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
    return dst.stat().st_size


def _run_psnr_v2(
    src_ply: Path, dec_ply: Path, label: str, scene: str,
    n_cams: int, image_size: int,
) -> dict:
    out_json = src_ply.parent / "psnr.json"
    t0 = time.time()
    rc = subprocess.run(
        [
            "python", "/opt/psnr_v2.py",
            "--source", str(src_ply),
            "--decoded", str(dec_ply),
            "--scene", scene,
            "--label", label,
            "--out", str(out_json),
            "--n-cams", str(n_cams),
            "--image-size", str(image_size),
        ],
        capture_output=True, text=True,
    )
    wall = time.time() - t0
    sys.stdout.write(rc.stdout[-3000:])
    sys.stdout.flush()
    if rc.returncode != 0:
        sys.stderr.write(rc.stderr[-3000:])
        return {
            "label": label, "scene": scene,
            "error": f"psnr_v2 rc={rc.returncode}",
            "stderr_tail": rc.stderr[-2000:],
            "stdout_tail": rc.stdout[-2000:],
            "psnr_v2_wall_sec": wall,
        }
    result = json.loads(out_json.read_text())
    result["psnr_v2_wall_sec"] = wall
    result["stdout_tail"] = rc.stdout[-1500:]
    return result


@app.function(gpu="A100", timeout=900, memory=32_768)
def audit_pair(
    label: str,
    scene: str,
    source_url: str,
    decoded_url: str,
    n_cams: int = 8,
    image_size: int = 512,
    decoded_is_tar: bool = False,
    decoded_member: str = "decoded.ply",
) -> dict:
    """Run psnr_v2 on two pre-staged PLY URLs (Inria 3DGS format).

    When `decoded_is_tar=True`, `decoded_url` is treated as a .tar
    artifact (e.g. the output of splatforge-fcgs / splatforge-hosted-
    neural). The tar is downloaded and `decoded_member` is extracted as
    the decoded PLY before running psnr_v2.
    """
    import tarfile
    work = Path(tempfile.mkdtemp(prefix=f"p2-{label}-"))
    src_ply = work / "source.ply"
    dec_ply = work / "decoded.ply"
    print(f"[audit_pair] downloading source {source_url}", flush=True)
    src_bytes = _dl(source_url, src_ply)
    print(f"[audit_pair]   {src_bytes:,} bytes", flush=True)
    if decoded_is_tar:
        tar_path = work / "decoded.tar"
        print(f"[audit_pair] downloading decoded TAR {decoded_url}", flush=True)
        tar_bytes = _dl(decoded_url, tar_path)
        print(f"[audit_pair]   tar {tar_bytes:,} bytes; extracting {decoded_member}",
              flush=True)
        with tarfile.open(tar_path) as tf:
            try:
                m = tf.getmember(decoded_member)
            except KeyError:
                members = [m.name for m in tf.getmembers()]
                # Walk all members for the basename
                cand = [m for m in tf.getmembers() if m.name.endswith(decoded_member)]
                if not cand:
                    raise FileNotFoundError(
                        f"member {decoded_member} not in tar (saw {members[:20]})"
                    )
                m = cand[0]
            with tf.extractfile(m) as f, dec_ply.open("wb") as out:
                shutil.copyfileobj(f, out, length=4 * 1024 * 1024)
        dec_bytes = dec_ply.stat().st_size
        print(f"[audit_pair]   extracted {dec_bytes:,} bytes", flush=True)
    else:
        print(f"[audit_pair] downloading decoded {decoded_url}", flush=True)
        dec_bytes = _dl(decoded_url, dec_ply)
        print(f"[audit_pair]   {dec_bytes:,} bytes", flush=True)
    res = _run_psnr_v2(src_ply, dec_ply, label, scene, n_cams, image_size)
    res["source_bytes"] = src_bytes
    res["decoded_bytes"] = dec_bytes
    return res


@app.function(gpu="A100", timeout=1200, memory=32_768)
def audit_in_process(
    preset: str,
    scene: str,
    source_url: str,
    n_cams: int = 8,
    image_size: int = 512,
) -> dict:
    """End-to-end audit for in-process Rust presets (web-mobile, size-min).

    Pipeline: download source PLY -> splatforge optimize -> splatforge
    convert to PLY -> psnr_v2 (src, decoded).
    """
    work = Path(tempfile.mkdtemp(prefix=f"p2-{preset}-"))
    src_ply = work / "source.ply"
    glb_path = work / "optimized.glb"
    dec_ply = work / "decoded.ply"

    print(f"[audit_in_process:{preset}] downloading {source_url}", flush=True)
    src_bytes = _dl(source_url, src_ply)
    print(f"[audit_in_process:{preset}] source {src_bytes:,} bytes", flush=True)

    t_opt = time.time()
    rc = subprocess.run(
        [
            "splatforge", "optimize", str(src_ply),
            "--preset", preset,
            "--target", "glb",
            "-o", str(glb_path),
        ],
        capture_output=True, text=True,
    )
    opt_wall = time.time() - t_opt
    print(f"[audit_in_process:{preset}] optimize wall={opt_wall:.1f}s "
          f"rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        return {
            "label": preset, "scene": scene,
            "error": f"splatforge optimize rc={rc.returncode}",
            "stderr_tail": rc.stderr[-2000:],
            "stdout_tail": rc.stdout[-2000:],
            "source_bytes": src_bytes,
        }
    glb_bytes = glb_path.stat().st_size
    print(f"[audit_in_process:{preset}] optimized GLB {glb_bytes:,} bytes", flush=True)

    t_cv = time.time()
    rc2 = subprocess.run(
        ["splatforge", "convert", str(glb_path), "--to", "ply", "-o", str(dec_ply)],
        capture_output=True, text=True,
    )
    cv_wall = time.time() - t_cv
    print(f"[audit_in_process:{preset}] convert wall={cv_wall:.1f}s "
          f"rc={rc2.returncode}", flush=True)
    if rc2.returncode != 0:
        return {
            "label": preset, "scene": scene,
            "error": f"splatforge convert rc={rc2.returncode}",
            "stderr_tail": rc2.stderr[-2000:],
            "stdout_tail": rc2.stdout[-2000:],
            "source_bytes": src_bytes,
            "glb_bytes": glb_bytes,
        }
    dec_bytes = dec_ply.stat().st_size
    print(f"[audit_in_process:{preset}] decoded PLY {dec_bytes:,} bytes", flush=True)

    res = _run_psnr_v2(src_ply, dec_ply, preset, scene, n_cams, image_size)
    res["source_bytes"] = src_bytes
    res["glb_bytes"] = glb_bytes
    res["decoded_bytes"] = dec_bytes
    res["optimize_wall_sec"] = opt_wall
    res["convert_wall_sec"] = cv_wall
    return res


@app.local_entrypoint()
def main(
    preset: str,
    scene: str,
    source_url: str,
    out_path: str,
    decoded_url: str = "",
):
    """Local CLI. Two modes:
      In-process: `--preset web-mobile --scene bonsai --source-url ... --out-path ...`
      Paired:     `--preset fcgs-instant --scene bonsai --source-url ... \\
                   --decoded-url ... --out-path ...`
    """
    if decoded_url:
        result = audit_pair.remote(
            label=preset, scene=scene,
            source_url=source_url, decoded_url=decoded_url,
        )
    else:
        result = audit_in_process.remote(
            preset=preset, scene=scene, source_url=source_url,
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"content_mean={result.get('content_mean_psnr_db')} "
          f"content_min={result.get('content_min_psnr_db')}")
