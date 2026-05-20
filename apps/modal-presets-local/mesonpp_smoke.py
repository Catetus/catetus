#!/usr/bin/env python3
"""mesonpp_smoke.py — End-to-end smoke test for the MesonGS++ codec.

Drives the `mesonpp` Rust CLI through an encode → decode round-trip on a
real 3DGS PLY and reports:

  * input PLY size and splat count
  * encoded `.meson` size + compression ratio
  * decoded PLY splat count + bbox delta (axis-aligned)
  * encode / decode wall-clock seconds (measured by the binary itself
    and re-measured here for sanity)

This isn't a render-PSNR test — that requires a fidelity GPU and lives
in `apps/fidelity-gpu/`. What we prove here is the artifact pipeline:
the decoded PLY parses, has the right splat count, and stays inside the
input bbox to within float epsilon (any axis-aligned drift > 1 % of
bbox span is a bug in the quantization).

Default scene is `benches/scenes/real/bonsai_iter7000.ply` (Mip-NeRF360
bonsai re-trained to iter 7000 — 273.7 MB, 1.16 M splats). Override
with `--scene <path>` to point at any other Inria-style PLY locally.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "benches" / "scenes" / "real" / "bonsai_iter7000.ply"
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "mesonpp"
DEFAULT_OUTDIR = REPO_ROOT / "tasks" / "scripts" / "meson-out"


def parse_ply_header(path: Path) -> tuple[int, int]:
    """Return (vertex_count, header_byte_size) for a binary-LE Inria PLY.

    Doesn't validate beyond the bits the smoke test needs.
    """
    with path.open("rb") as f:
        header = b""
        while True:
            chunk = f.read(4096)
            if not chunk:
                raise ValueError(f"{path}: PLY header didn't terminate")
            header += chunk
            idx = header.find(b"end_header\n")
            if idx != -1:
                header_size = idx + len(b"end_header\n")
                break
    text = header[:header_size].decode("latin-1", errors="replace")
    vertex_count = 0
    for line in text.splitlines():
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
            break
    return vertex_count, header_size


def bbox_from_ply(path: Path) -> tuple[list[float], list[float]]:
    """Compute the axis-aligned bbox of the `x,y,z` columns in a PLY.

    Streams the file in chunks to keep memory bounded; PLYs in this
    repo run up to 900 MB and we don't want to OOM the smoke harness.
    Assumes the first three properties of element `vertex` are float x,
    float y, float z — which is the Inria convention.
    """
    vertex_count, header_size = parse_ply_header(path)
    # Inria PLY: 17 float properties + 45 SH-rest f32 + 3 f32 scale + 4 f32
    # rotation = 62 floats? Use the file size + header to compute stride.
    file_size = path.stat().st_size
    body_size = file_size - header_size
    if vertex_count == 0:
        raise ValueError(f"{path}: zero vertices")
    stride = body_size // vertex_count
    if stride < 12 or stride * vertex_count != body_size:
        raise ValueError(
            f"{path}: stride math mismatch (body={body_size}, n={vertex_count})"
        )
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with path.open("rb") as f:
        f.seek(header_size)
        # Read in chunks of ~64k splats.
        chunk_splats = 65536
        chunk_size = stride * chunk_splats
        remaining = vertex_count
        while remaining > 0:
            n_now = min(chunk_splats, remaining)
            raw = f.read(stride * n_now)
            if len(raw) != stride * n_now:
                raise ValueError(f"{path}: truncated body")
            # Just the first 12 bytes of each splat are x,y,z.
            for i in range(n_now):
                xyz = struct.unpack_from("<fff", raw, i * stride)
                for a in range(3):
                    if xyz[a] < lo[a]:
                        lo[a] = xyz[a]
                    if xyz[a] > hi[a]:
                        hi[a] = xyz[a]
            remaining -= n_now
    return lo, hi


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}\n"
        )
    return res.stdout


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--k-low",
        type=int,
        default=256,
        help="K-means K for scale/rot/opacity (default 256).",
    )
    p.add_argument(
        "--k-color",
        type=int,
        default=256,
        help="K-means K for f_dc / f_rest (default 256).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=10,
        help="K-means Lloyd's iterations (default 10).",
    )
    p.add_argument(
        "--preserve-order",
        action="store_true",
        help="Store the Morton permutation so the decoded PLY keeps the input ordering.",
    )
    args = p.parse_args()

    scene = args.scene.resolve()
    if not scene.exists():
        print(f"FAIL: scene not found: {scene}", file=sys.stderr)
        return 2
    binary = args.bin.resolve()
    if not binary.exists():
        print(
            f"FAIL: mesonpp binary not found at {binary}. "
            f"Run `cargo build --release --bin mesonpp` first.",
            file=sys.stderr,
        )
        return 2
    args.outdir.mkdir(parents=True, exist_ok=True)
    meson_path = args.outdir / (scene.stem + ".meson")
    rt_ply = args.outdir / (scene.stem + "_rt.ply")

    print(f"[smoke] scene: {scene}")
    in_n, _ = parse_ply_header(scene)
    in_bytes = scene.stat().st_size
    print(
        f"[smoke] input: {in_bytes / 1e6:.2f} MB, {in_n} splats"
    )

    print("[smoke] computing input bbox …")
    t0 = time.time()
    in_lo, in_hi = bbox_from_ply(scene)
    print(
        f"[smoke] input bbox: lo={in_lo} hi={in_hi} ({time.time() - t0:.1f}s)"
    )

    enc_cmd = [
        str(binary),
        "encode",
        str(scene),
        str(meson_path),
        "--k-low",
        str(args.k_low),
        "--k-color",
        str(args.k_color),
        "--iters",
        str(args.iters),
    ]
    if args.preserve_order:
        enc_cmd.append("--preserve-order")
    print(f"[smoke] $ {' '.join(enc_cmd)}")
    t1 = time.time()
    enc_out = run(enc_cmd)
    enc_secs = time.time() - t1
    print(enc_out.rstrip())
    out_bytes = meson_path.stat().st_size
    ratio = in_bytes / out_bytes
    print(
        f"[smoke] encoded: {out_bytes / 1e6:.2f} MB in {enc_secs:.2f}s "
        f"({ratio:.2f}× ratio)"
    )

    dec_cmd = [str(binary), "decode", str(meson_path), str(rt_ply)]
    print(f"[smoke] $ {' '.join(dec_cmd)}")
    t2 = time.time()
    dec_out = run(dec_cmd)
    dec_secs = time.time() - t2
    print(dec_out.rstrip())
    print(f"[smoke] decoded in {dec_secs:.2f}s")

    out_n, _ = parse_ply_header(rt_ply)
    out_lo, out_hi = bbox_from_ply(rt_ply)
    print(f"[smoke] output bbox: lo={out_lo} hi={out_hi}")

    # Validation: splat count exact, bbox within 1 % of input span.
    ok = True
    if out_n != in_n:
        print(f"FAIL: splat count drift: in={in_n} out={out_n}")
        ok = False
    for a in range(3):
        span = max(in_hi[a] - in_lo[a], 1e-6)
        d_lo = abs(out_lo[a] - in_lo[a]) / span
        d_hi = abs(out_hi[a] - in_hi[a]) / span
        if d_lo > 0.01 or d_hi > 0.01:
            print(
                f"FAIL: bbox axis {a} drifted >1%: lo Δ={d_lo:.4f} hi Δ={d_hi:.4f}"
            )
            ok = False

    summary = {
        "scene": str(scene),
        "n_splats_in": in_n,
        "n_splats_out": out_n,
        "in_bytes": in_bytes,
        "meson_bytes": out_bytes,
        "ratio": ratio,
        "encode_secs": enc_secs,
        "decode_secs": dec_secs,
        "preserve_order": args.preserve_order,
        "k_low": args.k_low,
        "k_color": args.k_color,
        "iters": args.iters,
        "bbox_in": [in_lo, in_hi],
        "bbox_out": [out_lo, out_hi],
        "passed": ok,
    }
    out_json = args.outdir / (scene.stem + "_smoke.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[smoke] wrote summary: {out_json}")
    if ok:
        print(f"PASS — ratio {ratio:.2f}×, encode {enc_secs:.1f}s, decode {dec_secs:.1f}s")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
