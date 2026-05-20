#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""publish_l2_l3.py — upload Sweet Corals LODGE L2+L3 chunks to Vercel Blob,
merge with the existing L4+L5 manifest, and write a combined L2-L5 manifest.

Run on the 4090 (WSL) where the source .lodge directory lives:

    BLOB_READ_WRITE_TOKEN=vercel_blob_rw_... \
      python3 publish_l2_l3.py \
        --lodge-dir /mnt/c/Users/monta/Catetus/.bench-scenes/sweet-corals-full.lodge \
        --reuse-manifest https://xmcqr5nqjygbqjqw.public.blob.vercel-storage.com/samples/sweet-corals.lodge/manifest-l4-l5-QevCTqvCPEvNGoZKyeEUn9iYFWgEoX.json \
        --blob-prefix samples/sweet-corals.lodge \
        --output-urls /home/montabano1/sweet-corals-l2-l3-urls.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

VERCEL_BLOB_PUT_BASE = "https://blob.vercel-storage.com"


def upload_blob(token: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> str:
    url = f"{VERCEL_BLOB_PUT_BASE}/{key}"
    headers = {
        "Authorization": f"Bearer {token}",
        "x-content-type": content_type,
        "x-api-version": "7",
        "x-cache-control-max-age": "31536000",
    }
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.put(url, headers=headers, data=body, timeout=600)
            if r.status_code in (200, 201):
                return r.json()["url"]
            print(f"  attempt {attempt + 1}: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        except requests.RequestException as e:
            last_err = e
            print(f"  attempt {attempt + 1}: {e}", file=sys.stderr)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"upload failed for {key}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lodge-dir", required=True)
    ap.add_argument("--reuse-manifest", required=True,
                    help="Existing public manifest URL to copy L4/L5 chunk URLs from")
    ap.add_argument("--level", type=int, action="append", default=None,
                    help="Levels to upload (default: 2 3)")
    ap.add_argument("--blob-prefix", default="samples/sweet-corals.lodge")
    ap.add_argument("--output-urls", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    levels_to_upload = args.level or [2, 3]

    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: BLOB_READ_WRITE_TOKEN env var is required", file=sys.stderr)
        return 2

    lodge_dir = Path(args.lodge_dir).resolve()
    manifest_path = lodge_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
        return 2

    local_manifest: dict[str, Any] = json.loads(manifest_path.read_text())

    print(f"Fetching reuse manifest: {args.reuse_manifest}", file=sys.stderr)
    r = requests.get(args.reuse_manifest, timeout=60)
    r.raise_for_status()
    reuse_manifest: dict[str, Any] = r.json()
    reuse_by_level: dict[int, dict[str, Any]] = {l["level"]: l for l in reuse_manifest["levels"]}
    print(f"  reuse manifest has levels: {sorted(reuse_by_level.keys())}", file=sys.stderr)

    plan_total = 0
    for lvl in levels_to_upload:
        tl = next((l for l in local_manifest["levels"] if l["level"] == lvl), None)
        if tl is None:
            print(f"ERROR: level {lvl} not in local manifest", file=sys.stderr)
            return 2
        b = sum((lodge_dir / c["path"]).stat().st_size for c in tl["chunks"])
        plan_total += b
        print(f"Plan: level {lvl}, {len(tl['chunks'])} chunks, "
              f"{tl['splat_count']:,} splats, {b / 1e6:.1f} MB", file=sys.stderr)
    print(f"Plan total upload: {plan_total / 1e6:.1f} MB", file=sys.stderr)

    if args.dry_run:
        return 0

    rewritten_levels: list[dict[str, Any]] = []
    all_new_chunk_urls: list[str] = []
    t0 = time.time()
    for lvl in levels_to_upload:
        tl = next(l for l in local_manifest["levels"] if l["level"] == lvl)
        chunks = tl["chunks"]
        chunk_urls: list[str] = []
        for i, chunk in enumerate(chunks):
            ply_path = lodge_dir / chunk["path"]
            body = ply_path.read_bytes()
            key = f"{args.blob_prefix}/level_{lvl}/chunk_{chunk['index']:04d}.ply"
            url = upload_blob(token, key, body)
            chunk_urls.append(url)
            elapsed = time.time() - t0
            print(f"[L{lvl} {i + 1:>3}/{len(chunks)}] {ply_path.name} -> {url}  "
                  f"({len(body) / 1e6:.1f} MB, {elapsed:.1f}s elapsed)", file=sys.stderr)
            sys.stderr.flush()

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
        rewritten_levels.append({
            "level": tl["level"],
            "splat_count": tl["splat_count"],
            "reduction": tl.get("reduction", 1.0),
            "depth_threshold": tl.get("depth_threshold", 0.0),
            "chunks": rewritten_chunks,
        })
        all_new_chunk_urls.extend(chunk_urls)

    new_level_ids = {lvl for lvl in levels_to_upload}
    for lvl_id, lvl_desc in reuse_by_level.items():
        if lvl_id not in new_level_ids:
            rewritten_levels.append(lvl_desc)
    rewritten_levels.sort(key=lambda l: l["level"])

    rewritten = {
        "version": local_manifest["version"],
        "source": local_manifest.get("source", reuse_manifest.get("source", "")),
        "original_splat_count": local_manifest["original_splat_count"],
        "bbox": local_manifest["bbox"],
        "levels": rewritten_levels,
    }
    lvl_tag = "-".join(f"l{l['level']}" for l in rewritten_levels)
    manifest_key = f"{args.blob_prefix}/manifest-{lvl_tag}.json"
    manifest_url = upload_blob(token, manifest_key, json.dumps(rewritten).encode("utf-8"),
                                "application/json")
    print(f"manifest -> {manifest_url}", file=sys.stderr)

    out = {
        "manifest_url": manifest_url,
        "new_chunk_urls": all_new_chunk_urls,
        "uploaded_levels": levels_to_upload,
        "final_levels": [l["level"] for l in rewritten_levels],
        "num_new_chunks": len(all_new_chunk_urls),
        "total_new_bytes": plan_total,
        "elapsed_s": time.time() - t0,
    }
    if args.output_urls:
        Path(args.output_urls).write_text(json.dumps(out, indent=2))
        print(f"wrote URL inventory to {args.output_urls}", file=sys.stderr)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
