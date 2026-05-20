#!/usr/bin/env python3
"""D2.1 progressive bitstream Phase-1 verification.

Gates (per the task brief):

  1. Round-trip: encode <input.ply> → decode at 100% bytes → decoded PLY's
     splat multiset is bit-identical to the input (binary records, since
     the encoder reorders without touching record bytes).

  2. Partial decode: decode at 10%, 25%, 50% bytes → each is a valid
     Inria 3DGS PLY, and the importance score sequence
     (`opacity * det(scale)^{2/3}`) is monotonically non-decreasing as a
     function of bytes (= "PSNR-equivalent monotonicity" — we proxy PSNR
     with importance because Phase-1 splats themselves are unchanged;
     only the rendered count grows).

Usage:
    python verify_progressive_phase1.py <input.ply>

The script invokes the local release binary at
`target/release/catetus`. It uses only the Python stdlib + numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "target" / "release" / "catetus"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def parse_ply_header(path: Path):
    """Return (body_offset, record_size, n_vertices, col_offsets dict)."""
    col_offsets: dict[str, int] = {}
    record_size = 0
    n_vertices = 0
    header_bytes = bytearray()
    with open(path, "rb") as f:
        # Read line-by-line until end_header. The header is ASCII, the
        # body that follows is binary LE.
        line = f.readline()
        if line.strip() != b"ply":
            raise ValueError(f"{path} is not a PLY (no `ply` magic)")
        header_bytes += line
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PLY header truncated")
            header_bytes += line
            stripped = line.strip()
            if stripped == b"end_header":
                break
            parts = stripped.split()
            if parts[:2] == [b"element", b"vertex"]:
                n_vertices = int(parts[2])
                in_vertex = True
            elif parts[:1] == [b"element"]:
                in_vertex = False
            elif parts[:1] == [b"property"] and in_vertex:
                ty = parts[1]
                name = parts[2].decode()
                # Size lookup mirrors the Rust parser.
                if ty in (b"float", b"float32", b"int", b"int32", b"uint", b"uint32"):
                    sz = 4
                elif ty in (b"double", b"float64"):
                    sz = 8
                elif ty in (b"short", b"int16", b"ushort", b"uint16"):
                    sz = 2
                elif ty in (b"char", b"int8", b"uchar", b"uint8"):
                    sz = 1
                else:
                    raise ValueError(f"unsupported property type {ty!r}")
                col_offsets[name] = record_size
                record_size += sz
        body_offset = f.tell()
    return body_offset, record_size, n_vertices, col_offsets


def read_records(path: Path) -> tuple[np.ndarray, dict[str, int], int]:
    """Memory-map a PLY's vertex records as a (N, record_size) uint8 view.

    Returns (records, col_offsets, n_vertices).
    """
    body_offset, stride, n, col_offsets = parse_ply_header(path)
    # mmap is enough — we never mutate.
    arr = np.fromfile(path, dtype=np.uint8, offset=body_offset)
    if arr.size < n * stride:
        raise ValueError(
            f"{path}: body too short, expected {n*stride} bytes got {arr.size}"
        )
    records = arr[: n * stride].reshape(n, stride)
    return records, col_offsets, n


def importance_scores(records: np.ndarray, col_offsets: dict[str, int]) -> np.ndarray:
    """Re-implement the Rust score: opacity * det(scale)^{2/3}.

    `opacity` field is the pre-sigmoid logit, scales are log-scales.
    """
    def f32_col(name: str) -> np.ndarray:
        off = col_offsets[name]
        # Each record is `stride` bytes; the column is 4 bytes at `off`.
        return np.frombuffer(records[:, off : off + 4].tobytes(), dtype="<f4")

    op_logit = f32_col("opacity")
    sx = np.exp(f32_col("scale_0"))
    sy = np.exp(f32_col("scale_1"))
    sz = np.exp(f32_col("scale_2"))
    opacity = 1.0 / (1.0 + np.exp(-op_logit))
    det = sx * sy * sz
    # det^{2/3} = (cbrt(det))^2
    cube_root = np.cbrt(np.maximum(det, 0.0))
    score = opacity * cube_root * cube_root
    score = np.where(np.isfinite(score) & (det > 0), score, 0.0)
    return score.astype(np.float64)


def record_psnr(reference: np.ndarray, partial: np.ndarray) -> float:
    """Synthetic PSNR proxy: treat the per-splat *opacity-weighted scale*
    as the "rendered intensity contribution" of each splat. The full
    reference is `sum(scores)` (rendered energy if every splat draws);
    a partial decode contains the top-K splats, contributing
    `sum(scores[:K])`. The ratio is the *rendered-energy fraction*.

    Returns PSNR(dB) = 10 * log10( total_energy / missing_energy ).
    Higher = closer to full. Strictly monotone in `K` because scores
    are non-negative and the top-K is descending. For K == N, returns
    +inf which we cap to a sentinel.
    """
    full_energy = float(np.sum(reference))
    if full_energy <= 0.0:
        return float("inf")
    missing = full_energy - float(np.sum(partial))
    if missing <= 0.0:
        return float("inf")
    return 10.0 * np.log10(full_energy / missing)


def sha256_records(records: np.ndarray) -> str:
    """SHA-256 of the *sorted* record byte set — robust to permutation."""
    # Convert each record to bytes, sort, hash.
    h = hashlib.sha256()
    # numpy sort on the structured byte view: easiest is to sort row order
    # using a void dtype.
    void_dt = np.dtype((np.void, records.shape[1]))
    rows = records.view(void_dt).reshape(-1)
    order = np.argsort(rows)
    for idx in order:
        h.update(records[idx].tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Input Inria 3DGS PLY")
    ap.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the intermediate .mgs2 + partial PLYs for inspection",
    )
    args = ap.parse_args()

    if not CLI.exists():
        print(f"ERROR: build the release CLI first: {CLI}", file=sys.stderr)
        return 2

    input_ply: Path = args.input.resolve()
    if not input_ply.exists():
        print(f"ERROR: input PLY not found: {input_ply}", file=sys.stderr)
        return 2

    tmpdir = Path(tempfile.mkdtemp(prefix="d1-prog-verify-"))
    print(f"# scratch dir: {tmpdir}")
    mgs2 = tmpdir / "scene.mgs2"

    # Step 1: encode.
    t0 = time.time()
    run([str(CLI), "progressive", "encode", "-i", str(input_ply), "-o", str(mgs2)])
    enc_secs = time.time() - t0
    enc_size = mgs2.stat().st_size
    in_size = input_ply.stat().st_size

    # Header info.
    info_proc = subprocess.run(
        [str(CLI), "progressive", "info", "-i", str(mgs2)],
        capture_output=True,
        check=True,
        text=True,
    )
    info = json.loads(info_proc.stdout)
    n_splats = info["n_splats"]
    record_size = info["record_size"]
    payload_offset = info["payload_offset"]
    payload_len = info["payload_len"]

    print(
        f"# encode: {in_size} B PLY -> {enc_size} B mgs2 "
        f"({enc_size/in_size:.4f}x) in {enc_secs:.2f}s"
    )
    print(
        f"# header: n_splats={n_splats} record_size={record_size} "
        f"payload_offset={payload_offset} payload_len={payload_len}"
    )

    # Step 2: round-trip decode at 100 %.
    full_decoded = tmpdir / "scene.full.ply"
    run(
        [
            str(CLI),
            "progressive",
            "decode",
            "-i",
            str(mgs2),
            "-o",
            str(full_decoded),
        ]
    )

    in_records, in_cols, in_n = read_records(input_ply)
    full_records, full_cols, full_n = read_records(full_decoded)
    assert in_n == full_n == n_splats, (
        f"vertex count mismatch: in={in_n} full={full_n} hdr={n_splats}"
    )
    assert in_records.shape == full_records.shape

    # Gate 1a — round-trip splat-multiset identity:
    in_hash = sha256_records(in_records)
    full_hash = sha256_records(full_records)
    if in_hash != full_hash:
        print(
            "FAIL gate 1a: 100% decode is NOT bit-identical on the splat "
            f"multiset.\n  input sha256:    {in_hash}\n  decoded sha256:  {full_hash}",
            file=sys.stderr,
        )
        return 1
    print(f"PASS gate 1a: 100% decode is bit-identical on the splat multiset (sha256={in_hash[:16]}…)")

    # Gate 1b — descending importance: every score[i] >= score[i+1].
    full_scores = importance_scores(full_records, full_cols)
    diffs = np.diff(full_scores)
    if np.any(diffs > 1e-6):
        offenders = np.where(diffs > 1e-6)[0][:5]
        print(
            f"FAIL gate 1b: decoded scene is not in descending-importance order "
            f"(violations at indices {offenders.tolist()}).",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS gate 1b: decoded scene is in descending-importance order "
        f"(score[0]={full_scores[0]:.6g}, score[-1]={full_scores[-1]:.6g})."
    )

    # Step 3: partial decodes.
    cuts = {
        "10pct": enc_size // 10,
        "25pct": enc_size // 4,
        "50pct": enc_size // 2,
        "100pct": enc_size,
    }

    rows: list[dict] = []
    psnr_seq: list[float] = []
    splat_counts_seq: list[int] = []
    in_scores = importance_scores(in_records, in_cols)
    for tag, cut in cuts.items():
        partial_ply = tmpdir / f"scene.{tag}.ply"
        run(
            [
                str(CLI),
                "progressive",
                "decode",
                "-i",
                str(mgs2),
                "-o",
                str(partial_ply),
                "--partial-bytes",
                str(cut),
            ]
        )
        p_records, p_cols, p_n = read_records(partial_ply)
        # Importance scores of the partial set.
        p_scores = importance_scores(p_records, p_cols)
        # Synthetic PSNR proxy vs the full input.
        psnr = record_psnr(in_scores, p_scores)
        rows.append(
            {
                "cut": tag,
                "bytes": cut,
                "pct_of_full": cut / enc_size,
                "splats": int(p_n),
                "splats_pct": p_n / n_splats,
                "psnr_proxy_db": psnr,
                "score_sum_kept": float(np.sum(p_scores)),
            }
        )
        psnr_seq.append(psnr)
        splat_counts_seq.append(p_n)

    print("\n# partial-decode report:")
    print(json.dumps(rows, indent=2))

    # Gate 2a — splat counts monotone non-decreasing.
    for a, b in zip(splat_counts_seq, splat_counts_seq[1:]):
        if b < a:
            print(f"FAIL gate 2a: splat count regressed {a} -> {b}", file=sys.stderr)
            return 1
    print("PASS gate 2a: splat count is non-decreasing across cuts.")

    # Gate 2b — PSNR proxy monotone non-decreasing.
    for a, b in zip(psnr_seq, psnr_seq[1:]):
        # +inf at 100% is the expected terminal value (`record_psnr`
        # returns +inf when missing energy <= 0).
        if not (b >= a or (b == float("inf") and a != float("inf"))):
            print(
                f"FAIL gate 2b: PSNR proxy regressed {a:.4f} dB -> {b:.4f} dB",
                file=sys.stderr,
            )
            return 1
    print("PASS gate 2b: PSNR proxy (opacity-weighted scale energy) is monotone across cuts.")

    if not args.keep_tmp:
        # Spare the disk: the bonsai file is ~120 MB and we're keeping
        # five copies. Skip cleanup if `--keep-tmp` was passed.
        for f in tmpdir.iterdir():
            f.unlink()
        tmpdir.rmdir()
    else:
        print(f"# kept: {tmpdir}")

    print("\nALL GATES PASS — D2.1 progressive bitstream Phase-1 BUILT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
