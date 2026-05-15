#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reorder a SoA chunk-bytes scene file (the .bin + .meta.json produced by
`ply-to-soa.py`) so splats are sorted by their 3D Morton code over the scene
bounding box.

Rationale (novel-5): trained Gaussians naturally cluster in 3D, and the
runtime radix sort over depth keys is data-bandwidth bound. Morton-sorting at
encode time amplifies that spatial locality, which should give the sort
better cache hit rates on the same number of passes.

The transform is metadata-equivalent — `splatCount`, `bbox`, `byteLength`,
and `layout` are preserved. Only the row order inside each attribute table
changes. Render output is unchanged (we use alpha-blend with depth-sort, so
splat permutation is invisible). The source .meta.json's `source` field is
prefixed with `morton:` so the file is self-describing.

Usage:
    python soa-morton-permute.py <input.bin> <output.bin>

Both <input>.meta.json and <output>.meta.json are read/written alongside the
.bin files.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np


def _interleave3_u64(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Interleave 16-bit values from x, y, z into a 48-bit Morton code packed
    in u64. Bit-i of x lands at position 3i, bit-i of y at 3i+1, bit-i of z
    at 3i+2. Vectorized via the classic bit-spreading trick.
    """
    def spread(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64) & 0xFFFF
        v = (v | (v << 32)) & np.uint64(0x00000000FFFF0000_00000000FFFF)
        # The mask above isn't valid Python — split into the proper 64-bit
        # constants used in the 3D-spread literature.
        return v
    # Standard 16-bit→48-bit spread, three of them OR'd together.
    masks = [
        np.uint64(0x0000_FFFF_0000_FFFF),
        np.uint64(0x00FF_00FF_00FF_00FF),
        np.uint64(0x0F0F_0F0F_0F0F_0F0F),
        np.uint64(0x3333_3333_3333_3333),
        np.uint64(0x5555_5555_5555_5555),
    ]

    def s(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64) & np.uint64(0xFFFF)
        # 16 -> 48 bit spread using 5 mask-shift steps.
        v = (v | (v << np.uint64(16))) & masks[0]
        v = (v | (v << np.uint64(8)))  & masks[1]
        v = (v | (v << np.uint64(4)))  & masks[2]
        v = (v | (v << np.uint64(2)))  & masks[3]
        v = (v | (v << np.uint64(1)))  & masks[4]
        return v

    return s(x) | (s(y) << np.uint64(1)) | (s(z) << np.uint64(2))


def _morton_codes(xyz: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """16-bit-per-axis Morton codes (48-bit total in u64) for each row."""
    extent = (bbox_max - bbox_min)
    extent = np.where(extent <= 0, 1.0, extent)
    norm = (xyz - bbox_min) / extent  # [0, 1]
    norm = np.clip(norm, 0.0, 1.0)
    q = (norm * 65535.0 + 0.5).astype(np.uint32)
    q = np.minimum(q, np.uint32(65535))
    return _interleave3_u64(q[:, 0], q[:, 1], q[:, 2])


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src_bin = Path(sys.argv[1])
    dst_bin = Path(sys.argv[2])
    src_meta = src_bin.with_suffix(".meta.json")
    dst_meta = dst_bin.with_suffix(".meta.json")
    if not src_bin.exists():
        print(f"input bin not found: {src_bin}", file=sys.stderr)
        return 1
    if not src_meta.exists():
        print(f"input meta not found: {src_meta}", file=sys.stderr)
        return 1

    meta = json.loads(src_meta.read_text())
    n = int(meta["splatCount"])
    layout = meta["layout"]
    bbox_min = np.asarray(meta["bbox"]["min"], dtype=np.float32)
    bbox_max = np.asarray(meta["bbox"]["max"], dtype=np.float32)

    raw = src_bin.read_bytes()
    if len(raw) != int(meta["byteLength"]):
        print(
            f"byteLength mismatch: meta says {meta['byteLength']}, file is {len(raw)}",
            file=sys.stderr,
        )
        return 1

    # The five tables, in the canonical order ply-to-soa writes them:
    # positions (n*12), rotations (n*16), scales (n*12), opacities (n*4),
    # colorDC (n*12). Per-row stride is 4-byte f32 so we view each table
    # as a (n, k) f32 array. Permutation is along axis 0.
    table_specs = [
        ("positions", 3),
        ("rotations", 4),
        ("scales", 3),
        ("opacities", 1),
        ("colorDC", 3),
    ]

    arrays: dict[str, np.ndarray] = {}
    for name, k in table_specs:
        spec = layout[name]
        offset = int(spec["byteOffset"])
        nbytes = int(spec["byteLength"])
        expected = n * k * 4
        if nbytes != expected:
            print(
                f"layout/{name} byteLength={nbytes} != expected {expected} (n*k*4)",
                file=sys.stderr,
            )
            return 1
        block = np.frombuffer(raw, dtype=np.float32, count=n * k, offset=offset).reshape(n, k)
        arrays[name] = block

    t0 = time.perf_counter()
    codes = _morton_codes(arrays["positions"], bbox_min, bbox_max)
    perm = np.argsort(codes, kind="stable")
    sort_ms = (time.perf_counter() - t0) * 1000

    out_parts: list[bytes] = []
    for name, _k in table_specs:
        permuted = arrays[name][perm]
        # Force contiguous + correct dtype for tobytes.
        permuted = np.ascontiguousarray(permuted, dtype=np.float32)
        out_parts.append(permuted.tobytes("C"))

    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    with dst_bin.open("wb") as out:
        for part in out_parts:
            out.write(part)
    total_out = sum(len(p) for p in out_parts)
    if total_out != int(meta["byteLength"]):
        print(
            f"output byteLength {total_out} != input {meta['byteLength']}",
            file=sys.stderr,
        )
        return 1

    meta_out = dict(meta)
    src_name = meta_out.get("source", "")
    if not src_name.startswith("morton:"):
        meta_out["source"] = f"morton:{src_name}"
    meta_out["mortonPermuted"] = True
    dst_meta.write_text(json.dumps(meta_out, indent=2))

    print(
        f"[morton] {src_bin.name} -> {dst_bin.name} n={n} sort_ms={sort_ms:.1f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
