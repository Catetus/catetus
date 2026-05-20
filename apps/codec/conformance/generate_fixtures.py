#!/usr/bin/env python3
"""Generate the QAT-PLY v1 conformance fixtures.

Each fixture is a (a) PLY file written in binary_little_endian 1.0 with
the Catetus QAT-PLY v1 quantized_field comment(s), and (b) a JSON
sidecar listing the input quantized values, the scales, and the
expected dequantized fp32 outputs (bit-exact).

This script is the authoritative encoder used to populate
`fixtures/`. A renderer claiming v1-conformance must dequantize each
fixture and match the sidecar's `expected_fp32_b64` byte-for-byte.

Run: `python3 generate_fixtures.py`
"""

from __future__ import annotations

import base64
import json
import os
import struct
from dataclasses import dataclass
from typing import Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")


@dataclass
class IntQ:
    name: str
    dtype: str          # "int8" or "int4"
    channels: int
    # For int8: per-channel fp32 scale (length C).
    # For int4: per-anchor fp32 scale (length N).
    scales: list[float]
    # For int8: shape (N, C), signed int8 in [-128, 127].
    # For int4: shape (N, C), signed in [-8, 7] (unsigned-shifted on disk).
    q: list[list[int]]


def fp32_le_bytes(xs: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<f", x) for x in xs)


def b64(xs: Iterable[float]) -> str:
    return base64.b64encode(fp32_le_bytes(xs)).decode("ascii")


def write_ply(path: str, n_anchors: int, fields: list[IntQ],
              extra_fp32_columns: dict[str, list[float]] | None = None,
              extra_comments: list[str] | None = None) -> None:
    """Write a minimal binary_little_endian PLY containing:

    - One per-anchor x, y, z fp32 (all zero — geometry isn't part of
      the conformance assertions, only the quant columns).
    - For each int8 field: C `property uchar <name>_q_<i>` columns.
    - For each int4 field: ceil(C/2) `property uchar <name>_q_<i>` +
      one `property float <name>_scale` per-anchor scale column.
    - Optional extra fp32 columns (e.g. for ablation tests).

    Layout order in the binary block must match the header order.
    """
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment Catetus QAT-PLY v1 conformance fixture",
    ]
    for f in fields:
        if f.dtype == "int8":
            header_lines.append(
                f"comment quantized_field {f.name} int8 "
                f"channels={f.channels} scale_b64={b64(f.scales)}"
            )
        else:
            header_lines.append(
                f"comment quantized_field {f.name} int4 "
                f"channels={f.channels} packed_per_byte=2 "
                f"scale_kind=per_anchor"
            )
    for c in (extra_comments or []):
        header_lines.append(c)
    header_lines.append(f"element vertex {n_anchors}")
    header_lines += [
        "property float x",
        "property float y",
        "property float z",
    ]
    # Field-specific columns.
    for f in fields:
        if f.dtype == "int8":
            for i in range(f.channels):
                header_lines.append(f"property char {f.name}_q_{i}")
        else:
            n_bytes = (f.channels + 1) // 2
            for i in range(n_bytes):
                header_lines.append(f"property char {f.name}_q_{i}")
            header_lines.append(f"property float {f.name}_scale")
    # Extra fp32 columns (single-value-per-anchor).
    for name in (extra_fp32_columns or {}).keys():
        header_lines.append(f"property float {name}")
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    # Build body.
    body = bytearray()
    for n in range(n_anchors):
        # x, y, z = 0
        body += struct.pack("<fff", 0.0, 0.0, 0.0)
        for f in fields:
            if f.dtype == "int8":
                for c in range(f.channels):
                    val = f.q[n][c]
                    body += struct.pack("<b", val)
            else:
                # Pack two nibbles per byte.
                n_bytes = (f.channels + 1) // 2
                # Unsigned-shifted [0, 15].
                u = [(v + 8) & 0x0F for v in f.q[n]]
                # Pad odd channel with zero.
                if len(u) < n_bytes * 2:
                    u = u + [0] * (n_bytes * 2 - len(u))
                for bi in range(n_bytes):
                    low = u[2 * bi]
                    high = u[2 * bi + 1]
                    b = (high << 4) | low
                    # PLY emits a signed byte ("char") but the bit pattern is identical;
                    # we reinterpret as int8 to match plyfile's i1 dtype.
                    body += struct.pack("<b", b if b < 128 else b - 256)
                # Per-anchor scale.
                body += struct.pack("<f", f.scales[n])
        for name, vals in (extra_fp32_columns or {}).items():
            body += struct.pack("<f", vals[n])

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(body)


def _fp32(x: float) -> float:
    """Round a Python double to its fp32 representation."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def expected_int8(f: IntQ) -> list[list[float]]:
    """Bit-exact float reconstruction for int8 (per-channel scale).

    Matches the on-disk semantics: scales are first round-tripped through
    fp32 (because they live in the header as base64-encoded fp32 bytes),
    then multiplied by the int8 quant value. The product is then rounded
    to fp32 to mirror what a compliant fp32 decoder produces.
    """
    s32 = [_fp32(s) for s in f.scales]
    return [[_fp32(float(q) * s32[c]) for c, q in enumerate(row)] for row in f.q]


def expected_int4(f: IntQ) -> list[list[float]]:
    """Bit-exact float reconstruction for int4 (per-anchor scale)."""
    s32 = [_fp32(s) for s in f.scales]
    return [[_fp32(float(q) * s32[n]) for q in row] for n, row in enumerate(f.q)]


def emit_assertion(name: str, fields: list[IntQ]) -> dict:
    """Build the JSON sidecar payload for a fixture."""
    decoded = {}
    for f in fields:
        if f.dtype == "int8":
            mat = expected_int8(f)
        else:
            mat = expected_int4(f)
        flat = [v for row in mat for v in row]
        decoded[f.name] = {
            "shape": [len(mat), len(mat[0]) if mat else 0],
            "expected_fp32_b64": b64(flat),
        }
    return {"name": name, "n_anchors": len(fields[0].q), "fields": decoded}


def main() -> None:
    os.makedirs(FIX, exist_ok=True)

    manifest = []

    # Case 1: int8, 1 anchor, 1 channel — minimal.
    f = IntQ("f_anchor_feat", "int8", 1, [0.5], [[42]])
    write_ply(os.path.join(FIX, "case01_int8_1x1.ply"), 1, [f])
    manifest.append(emit_assertion("case01_int8_1x1.ply", [f]))

    # Case 2: int8, 4 anchors, 3 channels — typical small.
    f = IntQ(
        "f_anchor_feat", "int8", 3, [0.1, 0.2, 0.3],
        [[-128, 0, 127], [10, -20, 30], [0, 0, 0], [1, 1, 1]],
    )
    write_ply(os.path.join(FIX, "case02_int8_4x3.ply"), 4, [f])
    manifest.append(emit_assertion("case02_int8_4x3.ply", [f]))

    # Case 3: int8, 32 channels (covers a realistic anchor-feat slice).
    rows = [[((n * 7 + c * 3) % 256) - 128 for c in range(32)] for n in range(5)]
    scales = [0.01 + 0.001 * c for c in range(32)]
    f = IntQ("f_anchor_feat", "int8", 32, scales, rows)
    write_ply(os.path.join(FIX, "case03_int8_5x32.ply"), 5, [f])
    manifest.append(emit_assertion("case03_int8_5x32.ply", [f]))

    # Case 4: int4, 1 anchor, 4 channels — minimal.
    f = IntQ("f_offset", "int4", 4, [0.25], [[-8, -1, 0, 7]])
    write_ply(os.path.join(FIX, "case04_int4_1x4.ply"), 1, [f])
    manifest.append(emit_assertion("case04_int4_1x4.ply", [f]))

    # Case 5: int4, 3 anchors, 30 channels (Scaffold-GS shape).
    rows = [
        [((c % 16) - 8) for c in range(30)],
        [(((c + 5) % 16) - 8) for c in range(30)],
        [(((c * 3) % 16) - 8) for c in range(30)],
    ]
    f = IntQ("f_offset", "int4", 30, [0.5, 0.25, 0.125], rows)
    write_ply(os.path.join(FIX, "case05_int4_3x30.ply"), 3, [f])
    manifest.append(emit_assertion("case05_int4_3x30.ply", [f]))

    # Case 6: int4 with odd channel count (5) — exercise the pad path.
    f = IntQ("f_x", "int4", 5, [0.5, 1.0],
             [[-8, 7, 0, -1, 1], [3, -3, 5, -5, 2]])
    write_ply(os.path.join(FIX, "case06_int4_2x5_odd.ply"), 2, [f])
    manifest.append(emit_assertion("case06_int4_2x5_odd.ply", [f]))

    # Case 7: mixed int8 + int4 in one PLY.
    f1 = IntQ("f_anchor_feat", "int8", 2, [0.1, 0.2],
              [[-50, 50], [10, -10]])
    f2 = IntQ("f_offset", "int4", 4, [0.25, 0.5],
              [[-8, 7, 0, 1], [4, -4, 2, -2]])
    write_ply(os.path.join(FIX, "case07_mixed.ply"), 2, [f1, f2])
    manifest.append(emit_assertion("case07_mixed.ply", [f1, f2]))

    # Case 8: int8 with all-zero quantized values (edge: pure zero block).
    f = IntQ("f_anchor_feat", "int8", 4, [0.1, 0.2, 0.3, 0.4],
             [[0, 0, 0, 0], [0, 0, 0, 0]])
    write_ply(os.path.join(FIX, "case08_int8_zero.ply"), 2, [f])
    manifest.append(emit_assertion("case08_int8_zero.ply", [f]))

    # Case 9: int8 with extreme scales (very small, very large).
    f = IntQ("f_anchor_feat", "int8", 3, [1e-10, 1.0, 1e10],
             [[1, 1, 1], [-1, -1, -1], [127, 0, -128]])
    write_ply(os.path.join(FIX, "case09_int8_extreme_scales.ply"), 3, [f])
    manifest.append(emit_assertion("case09_int8_extreme_scales.ply", [f]))

    # Case 10: int8 + extra constant_field comment (forward compat).
    f = IntQ("f_anchor_feat", "int8", 2, [0.5, 0.5],
             [[10, -10], [20, -20]])
    write_ply(
        os.path.join(FIX, "case10_int8_with_constant.ply"),
        2, [f],
        extra_comments=["constant_field opacity 0x1.0000000000000p+0"],
    )
    manifest.append(emit_assertion("case10_int8_with_constant.ply", [f]))

    with open(os.path.join(HERE, "conformance.json"), "w") as fh:
        json.dump({"version": 1, "cases": manifest}, fh, indent=2)

    print(f"Wrote {len(manifest)} fixtures + conformance.json")


if __name__ == "__main__":
    main()
