#!/usr/bin/env python3
"""
build-hero.py — pre-process a vanilla Inria 3DGS PLY for hero use.

Two-step cleanup that the in-binary FloaterPrune cannot do alone:

  1. Per-axis percentile bbox crop. Measured against high-opacity splats only
     so floater halos do not pad the bbox. Drops splats that lie outside
     [p_lo, p_hi] on each axis.

  2. Importance-rank decimation. Keeps the top-N splats by
     (sigmoid(opacity) * mean(exp(scale))^2) — a rough proxy for the
     screen-space contribution of each splat. Stable, deterministic, no
     reliance on view-dependent metrics.

The input PLY must use the vanilla Inria 3DGS layout (x/y/z/scale_0..2/
opacity/rot_0..3/f_dc_*/f_rest_*). Scaffold-GS layouts are not supported.

Output is a binary little-endian PLY with the same property layout but a
smaller vertex count, ready for `catetus optimize`.

Exit codes:
  0  success
  1  bad CLI args
  2  PLY parse failed (likely Scaffold-GS or other non-vanilla layout)
  3  preprocessing produced too few splats (< 1000)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PLY_TYPE_MAP = {
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "uchar": np.uint8,
    "int": np.int32,
}


def load_ply(src: Path):
    """Parse a binary little-endian PLY. Returns (header_lines, structured ndarray)."""
    with src.open("rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                print(f"error: unexpected EOF reading PLY header from {src}", file=sys.stderr)
                sys.exit(2)
            header += line
        body = f.read()

    lines = header.decode("latin-1").splitlines()
    try:
        n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    except StopIteration:
        print("error: PLY header missing 'element vertex'", file=sys.stderr)
        sys.exit(2)

    props = [l.split() for l in lines if l.startswith("property")]
    try:
        dtype = np.dtype([(p[2], PLY_TYPE_MAP[p[1]]) for p in props])
    except KeyError as e:
        print(f"error: unsupported PLY property type {e}", file=sys.stderr)
        sys.exit(2)

    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    have = {p[2] for p in props}
    missing = required - have
    if missing:
        print(
            f"error: PLY is missing required vanilla 3DGS properties: {sorted(missing)}. "
            f"Is this a Scaffold-GS or otherwise non-vanilla layout?",
            file=sys.stderr,
        )
        sys.exit(2)

    arr = np.frombuffer(body, dtype=dtype, count=n)
    return lines, arr


def write_ply(dst: Path, header_lines, arr) -> None:
    new_header = []
    for l in header_lines:
        if l.startswith("element vertex"):
            new_header.append(f"element vertex {len(arr)}")
        else:
            new_header.append(l)
    with dst.open("wb") as f:
        f.write(("\n".join(new_header) + "\n").encode("latin-1"))
        f.write(arr.tobytes())


def per_axis_crop(arr, p_lo: float, p_hi: float):
    """Crop to the per-axis [p_lo, p_hi] percentile of high-opacity splats only."""
    on = arr["opacity"] > 0.0
    if on.sum() < 100:
        print(
            f"error: only {on.sum()} high-opacity splats; cannot derive a stable bbox. "
            "Is the scene converged?",
            file=sys.stderr,
        )
        sys.exit(3)
    lo = {ax: float(np.percentile(arr[ax][on], p_lo * 100)) for ax in ("x", "y", "z")}
    hi = {ax: float(np.percentile(arr[ax][on], p_hi * 100)) for ax in ("x", "y", "z")}
    print(f"  per-axis bbox at [{p_lo}, {p_hi}] of high-opacity splats:")
    for ax in ("x", "y", "z"):
        print(f"    {ax} [{lo[ax]:.3f}, {hi[ax]:.3f}]")
    mask = (
        (arr["x"] >= lo["x"]) & (arr["x"] <= hi["x"]) &
        (arr["y"] >= lo["y"]) & (arr["y"] <= hi["y"]) &
        (arr["z"] >= lo["z"]) & (arr["z"] <= hi["z"])
    )
    return arr[mask], (lo, hi)


def importance_decimate(arr, target_n: int):
    """Keep top-N splats by sigmoid(opacity) * mean(exp(scale))^2."""
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    alpha = sigmoid(arr["opacity"].astype(np.float64))
    s0 = np.exp(arr["scale_0"].astype(np.float64))
    s1 = np.exp(arr["scale_1"].astype(np.float64))
    s2 = np.exp(arr["scale_2"].astype(np.float64))
    mean_s = (s0 + s1 + s2) / 3.0
    importance = alpha * mean_s * mean_s
    order = np.argsort(-importance)
    keep_n = min(target_n, len(arr))
    return arr[order[:keep_n]]


def parse_axis_pct(s: str):
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--axis-pct must be 'lo,hi' (e.g. 0.03,0.97)")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 <= lo < hi <= 1.0):
        raise argparse.ArgumentTypeError("--axis-pct requires 0 <= lo < hi <= 1")
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="input vanilla 3DGS PLY")
    ap.add_argument("dst", type=Path, help="output preprocessed PLY")
    ap.add_argument("--axis-pct", type=parse_axis_pct, default=(0.03, 0.97),
                    help="per-axis percentile crop, 'lo,hi' (default 0.03,0.97)")
    ap.add_argument("--decimate", type=int, default=150_000,
                    help="target splat count after importance decimation (default 150000)")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"error: input PLY {args.src} does not exist", file=sys.stderr)
        return 1

    print(f"loading {args.src}")
    header_lines, arr = load_ply(args.src)
    n0 = len(arr)
    print(f"  loaded {n0} splats")
    print(f"  bbox before:")
    for ax in ("x", "y", "z"):
        print(f"    {ax} [{arr[ax].min():.3f}, {arr[ax].max():.3f}]")

    p_lo, p_hi = args.axis_pct
    cropped, _ = per_axis_crop(arr, p_lo, p_hi)
    print(f"  after crop: {len(cropped)} / {n0} ({100*len(cropped)/n0:.1f}%)")

    decimated = importance_decimate(cropped, args.decimate)
    print(f"  decimated to top {len(decimated)} by importance")
    print(f"  bbox after:")
    for ax in ("x", "y", "z"):
        print(f"    {ax} [{decimated[ax].min():.3f}, {decimated[ax].max():.3f}]")

    if len(decimated) < 1000:
        print(f"error: only {len(decimated)} splats survived; refusing to write", file=sys.stderr)
        return 3

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.dst, header_lines, decimated)
    print(f"wrote {args.dst} ({args.dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
