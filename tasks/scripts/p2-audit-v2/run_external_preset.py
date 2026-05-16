"""Drive a single private-app preset encode (fcgs-instant / hosted-neural /
hacpp-lzma) and run psnr_v2 on the decoded PLY.

For each preset the deployed Modal app's encode function is invoked
directly via `modal.Function.from_name(...)`, then the output tar is
fetched and unpacked to extract `decoded.ply`. That decoded.ply is then
passed to the p2-audit-psnr-v2 app's `audit_pair` function alongside the
source bonsai PLY URL.

Usage:
  python3 run_external_preset.py --preset fcgs-instant --scene bonsai
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import modal

HF_BONSAI = (
    "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/"
    "bonsai/point_cloud/iteration_7000/point_cloud.ply"
)
HF_BICYCLE = (
    "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/"
    "bicycle/point_cloud/iteration_7000/point_cloud.ply"
)

SCENES = {
    "bonsai": {"url": HF_BONSAI, "registered_name": "bonsai"},
    "bicycle": {"url": HF_BICYCLE, "registered_name": "bicycle"},
}

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _dl(url: str, dst: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "p2-audit/1.0"})
    with urllib.request.urlopen(req, timeout=900) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
    return dst.stat().st_size


def _upload_to_blob(local_path: Path, key: str, content_type: str) -> str:
    """Upload via Modal's vercel-blob secret. Returns public URL.

    Spins a quick CPU function on the audit app's image to do the PUT
    (avoiding the need for BLOB_READ_WRITE_TOKEN in our local env).
    """
    upload = modal.Function.from_name("p2-audit-psnr-v2", "upload_decoded_to_blob")
    return upload.remote(
        key=key,
        content_type=content_type,
        body_bytes=local_path.read_bytes(),
    )


def encode_fcgs(scene: str) -> dict:
    """Invoke deployed FCGS encoder; returns dict with tar_url + decoded_member."""
    src_url = SCENES[scene]["url"]
    job_id = f"p2audit-fcgs-{uuid.uuid4().hex[:8]}"
    print(f"[fcgs] job_id={job_id} src={src_url}", flush=True)
    run_fcgs = modal.Function.from_name("splatforge-fcgs", "run_fcgs")
    t0 = time.time()
    result = run_fcgs.remote(
        job_id=job_id,
        preset="fcgs-instant",
        blob_url=src_url,
        filename=f"{scene}.ply",
        callback_url=None,
    )
    wall = time.time() - t0
    print(f"[fcgs] encode wall={wall:.1f}s result keys={list(result.keys())}", flush=True)
    if result.get("status") != "done":
        raise RuntimeError(f"fcgs failed: {result}")
    return {
        "tar_url": result["output_url"],
        "decoded_member": "decoded.ply",
        "encode_meta": result,
    }


def encode_hosted_neural(scene: str) -> dict:
    src_url = SCENES[scene]["url"]
    registered = SCENES[scene]["registered_name"]
    job_id = f"p2audit-hn-{uuid.uuid4().hex[:8]}"
    print(f"[hosted-neural] job_id={job_id} registered_scene={registered}", flush=True)
    run_hn = modal.Function.from_name("splatforge-hosted-neural", "run_hosted_neural")
    t0 = time.time()
    result = run_hn.remote(
        job_id=job_id,
        preset="hosted-neural",
        blob_url=src_url,
        filename=registered,
        callback_url=None,
    )
    wall = time.time() - t0
    print(f"[hosted-neural] encode wall={wall:.1f}s result keys={list(result.keys())}", flush=True)
    if result.get("status") != "done":
        raise RuntimeError(f"hosted-neural failed: {result}")
    return {
        "tar_url": result["output_url"],
        "decoded_member": "repacked.ply",
        "encode_meta": result,
    }


def encode_hacpp_lzma(scene: str) -> Path:
    """Invoke hacpp-lzma. Requires a Scaffold-GS bundle, not an Inria PLY.

    For the audit this is a special case (see report) — we surface the
    encode-side ratio + render-PSNR delta only from the prior native
    Scaffold-GS rasterizer measurement.
    """
    raise NotImplementedError(
        "hacpp-lzma requires a Scaffold-GS bundle (point_cloud.ply + 3 MLPs); "
        "psnr_v2 cannot evaluate anchor-PLY without the Scaffold-GS rasterizer. "
        "See p2-audit-2026-05-15.md for the analysis + prior render-PSNR cite."
    )


def _fetch_and_extract_decoded(
    output_url: str, label: str, scene: str, ply_name: str = "decoded.ply",
) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f"p2-{label}-"))
    tar_path = tmp / "out.tar"
    print(f"[{label}] downloading {output_url}", flush=True)
    sz = _dl(output_url, tar_path)
    print(f"[{label}] tar {sz:,} bytes", flush=True)
    extracted = tmp / "extracted"
    extracted.mkdir()
    with tarfile.open(tar_path) as tf:
        tf.extractall(extracted)
    dec = extracted / ply_name
    if not dec.exists():
        # Try recursive find
        for p in extracted.rglob(ply_name):
            dec = p
            break
    if not dec.exists():
        raise FileNotFoundError(
            f"{ply_name} not found in {output_url} (entries: "
            f"{[p.name for p in extracted.rglob('*') if p.is_file()]})"
        )
    # Stage to /tmp/p2-audit so it's persistent across this script run
    persistent = OUT_DIR / f"{label}_{scene}_decoded.ply"
    shutil.copy(dec, persistent)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{label}] decoded saved -> {persistent}", flush=True)
    return persistent


def run_psnr_audit(label: str, scene: str, source_url: str, encode_out: dict) -> dict:
    """Pass the tar URL straight to audit_pair so it can extract decoded.ply
    inside the GPU container (avoids round-tripping a 280 MB PLY through
    the local Mac for Vercel Blob re-upload).
    """
    audit_pair = modal.Function.from_name("p2-audit-psnr-v2", "audit_pair")
    t0 = time.time()
    result = audit_pair.remote(
        label=label, scene=scene,
        source_url=source_url,
        decoded_url=encode_out["tar_url"],
        decoded_is_tar=True,
        decoded_member=encode_out["decoded_member"],
    )
    wall = time.time() - t0
    print(f"[audit] wall={wall:.1f}s", flush=True)
    result["_encode_meta"] = encode_out.get("encode_meta", {})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True,
                    choices=["fcgs-instant", "hosted-neural", "hacpp-lzma"])
    ap.add_argument("--scene", default="bonsai", choices=list(SCENES.keys()))
    args = ap.parse_args()

    if args.preset == "fcgs-instant":
        encode_out = encode_fcgs(args.scene)
    elif args.preset == "hosted-neural":
        encode_out = encode_hosted_neural(args.scene)
    elif args.preset == "hacpp-lzma":
        return encode_hacpp_lzma(args.scene)  # raises
    else:
        raise SystemExit(f"unknown preset {args.preset}")

    source_url = SCENES[args.scene]["url"]
    result = run_psnr_audit(args.preset, f"{args.scene}_iter7000", source_url, encode_out)

    out_json = OUT_DIR / f"{args.preset}_{args.scene}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_json}")
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 1
    print(f"content_mean_psnr_db={result.get('content_mean_psnr_db')} "
          f"content_min_psnr_db={result.get('content_min_psnr_db')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
