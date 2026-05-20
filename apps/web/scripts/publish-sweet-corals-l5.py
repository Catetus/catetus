#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
publish-sweet-corals-l5.py — upload Sweet Corals LODGE L5 chunks + manifest
to Vercel Blob so the public /scale page can stream them in-browser.

Why this lives outside Modal:
    The premise originally said "mount the existing Sweet Corals lodge
    Modal volume". That volume does not exist — the LODGE A.1 chunker
    output (`.bench-scenes/sweet-corals-full.lodge/`) was produced
    on the 4090 (Windows 11 / WSL2) and never round-tripped through a
    Modal volume. Round-tripping 758 MB of L5 PLYs through Modal just
    to re-upload them to Vercel Blob would burn ~2x the bytes and
    several minutes of Modal GPU instance time (the only "Modal volume"
    semantically equivalent to the chunks is the disk on `montespc`).

    So this script targets two run modes:

      1. **On the 4090 (preferred for L5 today).** Read BLOB_READ_WRITE_TOKEN
         from env (set by caller), iterate the local `.lodge/level_5/`
         directory, PUT each chunk to https://blob.vercel-storage.com,
         then PUT a rewritten manifest.json with absolute blob URLs.

      2. **On Modal (future levels).** Wrap the same logic in a
         `modal.Function` with `secret=modal.Secret.from_name("vercel-blob")`
         once the chunks are also resident on a Modal volume. This
         module exposes `upload_chunk()` + `build_manifest()` as plain
         functions so the Modal wrapper is a thin remote driver.

Usage (run on the 4090, in WSL):
    BLOB_READ_WRITE_TOKEN=vercel_blob_rw_... \
      python3 publish-sweet-corals-l5.py \
        --lodge-dir /mnt/c/Users/monta/Catetus/.bench-scenes/sweet-corals-full.lodge \
        --level 5 \
        --blob-prefix samples/sweet-corals.lodge \
        --output-urls /tmp/sweet-corals-l5-urls.json

Output:
    A JSON file listing the public chunk URLs + manifest URL, e.g.:
      {
        "manifest_url": "https://<store>.public.blob.vercel-storage.com/.../manifest.json",
        "chunk_urls": ["...chunk_0000.ply", ...],
        "level": 5,
        "splat_count": 3201888
      }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]


VERCEL_BLOB_PUT_BASE = "https://blob.vercel-storage.com"


def upload_blob(
    token: str, key: str, body: bytes, content_type: str = "application/octet-stream"
) -> str:
    """PUT bytes to Vercel Blob; return the public URL."""
    url = f"{VERCEL_BLOB_PUT_BASE}/{key}"
    params = {"addRandomSuffix": "0"}
    headers = {
        "Authorization": f"Bearer {token}",
        "x-content-type": content_type,
        "x-api-version": "7",
        "x-cache-control-max-age": "31536000",
    }
    # Chunked upload retries.
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            r = requests.put(url, params=params, headers=headers, data=body, timeout=300)
            if r.status_code in (200, 201):
                return r.json()["url"]
            print(
                f"  attempt {attempt + 1}: HTTP {r.status_code} {r.text[:200]}",
                file=sys.stderr,
            )
        except requests.RequestException as e:
            last_err = e
            print(f"  attempt {attempt + 1}: {e}", file=sys.stderr)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"upload failed for {key}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lodge-dir", required=True, help="Path to <name>.lodge directory")
    ap.add_argument(
        "--level",
        type=int,
        required=True,
        action="append",
        help="LODGE level to publish (repeat for multiple levels, e.g. --level 5 --level 4)",
    )
    ap.add_argument(
        "--blob-prefix",
        default="samples/sweet-corals.lodge",
        help="Vercel Blob key prefix",
    )
    ap.add_argument("--output-urls", default=None, help="Where to write the URL inventory JSON")
    ap.add_argument(
        "--dry-run", action="store_true", help="Print plan but don't upload"
    )
    args = ap.parse_args()

    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: BLOB_READ_WRITE_TOKEN env var is required", file=sys.stderr)
        return 2

    lodge_dir = Path(args.lodge_dir).resolve()
    manifest_path = lodge_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
        return 2

    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    levels = manifest.get("levels", [])
    target_levels = [next((l for l in levels if l["level"] == lvl), None) for lvl in args.level]
    if any(t is None for t in target_levels):
        missing = [lv for lv, t in zip(args.level, target_levels) if t is None]
        print(f"ERROR: levels {missing} not in manifest", file=sys.stderr)
        return 2

    plan_total = 0
    for tl in target_levels:
        chunks = tl["chunks"]
        b = sum((lodge_dir / c["path"]).stat().st_size for c in chunks)
        plan_total += b
        print(
            f"Plan: level {tl['level']}, {len(chunks)} chunks, "
            f"{tl['splat_count']:,} splats, {b / 1e6:.1f} MB",
            file=sys.stderr,
        )
    print(f"Plan total: {plan_total / 1e6:.1f} MB", file=sys.stderr)

    if args.dry_run:
        return 0

    rewritten_levels: list[dict[str, Any]] = []
    all_chunk_urls: list[str] = []
    t0 = time.time()
    for tl in target_levels:
        chunks = tl["chunks"]
        chunk_urls: list[str] = []
        for i, chunk in enumerate(chunks):
            ply_path = lodge_dir / chunk["path"]
            body = ply_path.read_bytes()
            key = f"{args.blob_prefix}/level_{tl['level']}/chunk_{chunk['index']:04d}.ply"
            url = upload_blob(token, key, body)
            chunk_urls.append(url)
            elapsed = time.time() - t0
            print(
                f"[L{tl['level']} {i + 1:>3}/{len(chunks)}] {ply_path.name} -> {url}  "
                f"({len(body) / 1e6:.1f} MB, {elapsed:.1f}s elapsed)",
                file=sys.stderr,
            )

        rewritten_chunks = [
            {
                "index": c["index"],
                "path": url,
                "splat_count": c["splat_count"],
                "bbox": c["bbox"],
                "centroid": c["centroid"],
                "radius": c["radius"],
                "blake3": c.get("blake3", ""),
            }
            for c, url in zip(chunks, chunk_urls)
        ]
        rewritten_levels.append(
            {
                "level": tl["level"],
                "splat_count": tl["splat_count"],
                "reduction": tl.get("reduction", 1.0),
                "depth_threshold": tl.get("depth_threshold", 0.0),
                "chunks": rewritten_chunks,
            }
        )
        all_chunk_urls.extend(chunk_urls)

    # Sort levels finest -> coarsest (level index ascending) for the
    # parseLodgeManifest contract.
    rewritten_levels.sort(key=lambda l: l["level"])

    rewritten = {
        "version": manifest["version"],
        "source": manifest.get("source", ""),
        "original_splat_count": manifest["original_splat_count"],
        "bbox": manifest["bbox"],
        "levels": rewritten_levels,
    }
    lvl_tag = "-".join(f"l{l['level']}" for l in rewritten_levels)
    manifest_key = f"{args.blob_prefix}/manifest-{lvl_tag}.json"
    manifest_url = upload_blob(
        token, manifest_key, json.dumps(rewritten).encode("utf-8"), "application/json"
    )
    print(f"manifest -> {manifest_url}", file=sys.stderr)

    out = {
        "manifest_url": manifest_url,
        "chunk_urls": all_chunk_urls,
        "levels": [l["level"] for l in rewritten_levels],
        "num_chunks": sum(len(l["chunks"]) for l in rewritten_levels),
        "total_bytes": plan_total,
        "elapsed_s": time.time() - t0,
    }
    if args.output_urls:
        Path(args.output_urls).write_text(json.dumps(out, indent=2))
        print(f"wrote URL inventory to {args.output_urls}", file=sys.stderr)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
