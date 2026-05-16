#!/usr/bin/env python3
"""Verify that the QAT-PLY v1 reference decoder matches the conformance
sidecar JSON byte-for-byte. Walks each fixture in `conformance.json`,
parses the PLY header to locate quantized_field comments, reads the
quant body, dequantizes, and compares fp32 bytes against the expected
base64 from the sidecar.

This is a pure-Python implementation following the spec — it acts as
a second independent decoder so a passing run means at least TWO
implementations of the spec agree byte-exactly.

Usage:
    python3 verify.py [--rebuild]

`--rebuild` regenerates the fixtures by calling generate_fixtures.py
before running the verification pass.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_ply(path: str) -> tuple[list[str], dict, bytes]:
    """Return (header_lines, vertex_props_in_order, body_bytes).

    `vertex_props_in_order` is a list of (name, ply_type) preserving
    declaration order so we can index into the binary body.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    # Find end_header\n
    needle = b"\nend_header\n"
    idx = data.find(needle)
    if idx < 0:
        raise ValueError(f"{path}: no end_header found")
    header = data[: idx + 1].decode("ascii", errors="replace")
    body = data[idx + len(needle):]
    lines = header.split("\n")
    return lines, body, body  # placeholder; we re-parse below


def vertex_layout(header_lines: list[str]) -> tuple[int, list[tuple[str, str]]]:
    """Return (n_vertices, [(name, ply_type), ...]) from header lines."""
    n = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element" and len(parts) >= 3 and parts[1] == "vertex":
            n = int(parts[2])
            in_vertex = True
            continue
        if parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
            continue
        if parts[0] == "property" and in_vertex:
            # property <type> <name>  (no list types in v1 quant fixtures)
            props.append((parts[2], parts[1]))
    return n, props


PLY_TYPE_FMT = {
    "char": ("<b", 1),
    "uchar": ("<B", 1),
    "short": ("<h", 2),
    "ushort": ("<H", 2),
    "int": ("<i", 4),
    "uint": ("<I", 4),
    "float": ("<f", 4),
    "double": ("<d", 8),
    "float32": ("<f", 4),
}


def read_columns(body: bytes, n: int,
                 props: list[tuple[str, str]]) -> dict[str, list]:
    """Read column-major data from a PLY binary body."""
    cols: dict[str, list] = {name: [] for name, _ in props}
    row_size = sum(PLY_TYPE_FMT[t][1] for _, t in props)
    expected = row_size * n
    if len(body) < expected:
        raise ValueError(
            f"body too short: have {len(body)} need {expected}"
        )
    off = 0
    for _ in range(n):
        for name, t in props:
            fmt, sz = PLY_TYPE_FMT[t]
            (v,) = struct.unpack(fmt, body[off : off + sz])
            cols[name].append(v)
            off += sz
    return cols


def parse_quantized_fields(header_lines: list[str]) -> list[dict]:
    out = []
    for line in header_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[0] != "comment" or parts[1] != "quantized_field":
            continue
        rec = {"name": parts[2], "dtype": parts[3],
               "scale_b64": None, "scale_kind": None,
               "packed_per_byte": None, "channels": None}
        for tok in parts[4:]:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            if k == "channels":
                rec["channels"] = int(v)
            elif k == "scale_b64":
                rec["scale_b64"] = v
            elif k == "scale_kind":
                rec["scale_kind"] = v
            elif k == "packed_per_byte":
                rec["packed_per_byte"] = int(v)
        out.append(rec)
    return out


def dequant_int8(cols: dict, name: str, channels: int,
                 scales: list[float]) -> list[float]:
    """Return a flat row-major fp32 list of length n*C."""
    n = len(cols[f"{name}_q_0"])
    out = []
    for r in range(n):
        for c in range(channels):
            q = cols[f"{name}_q_{c}"][r]  # signed int8
            out.append(float(q) * scales[c])
    return out


def dequant_int4(cols: dict, name: str, channels: int) -> list[float]:
    n_bytes = (channels + 1) // 2
    scales = cols[f"{name}_scale"]
    n = len(scales)
    out = []
    for r in range(n):
        s = scales[r]
        for c in range(channels):
            byte_signed = cols[f"{name}_q_{c // 2}"][r]
            byte_u = byte_signed & 0xFF
            if c % 2 == 0:
                nib = byte_u & 0x0F
            else:
                nib = (byte_u >> 4) & 0x0F
            signed_q = nib - 8
            out.append(float(signed_q) * s)
    return out


def fp32_b64(xs: list[float]) -> str:
    return base64.b64encode(b"".join(struct.pack("<f", x) for x in xs)).decode("ascii")


C_RUNNER = os.path.join(HERE, "c_decode_runner")


def c_decode(fixture_path: str, field_name: str) -> bytes:
    """Invoke the C reference decoder and return raw fp32 bytes."""
    r = subprocess.run(
        [C_RUNNER, fixture_path, field_name],
        capture_output=True, check=True,
    )
    return r.stdout


def verify_case(case: dict) -> tuple[bool, list[str]]:
    name = case["name"]
    fpath = os.path.join(HERE, "fixtures", name)
    if not os.path.exists(fpath):
        return False, [f"missing fixture: {fpath}"]
    header_lines, body, _ = parse_ply(fpath)
    n_verts, props = vertex_layout(header_lines)
    cols = read_columns(body, n_verts, props)
    qfields = parse_quantized_fields(header_lines)
    errs = []
    for rec in qfields:
        fname = rec["name"]
        if fname not in case["fields"]:
            continue
        expected = case["fields"][fname]
        # ---------- Python decoder ----------
        if rec["dtype"] == "int8":
            scales_bytes = base64.b64decode(rec["scale_b64"])
            scales = list(struct.unpack(f"<{len(scales_bytes)//4}f", scales_bytes))
            if len(scales) != rec["channels"]:
                errs.append(
                    f"{name}/{fname}: scale count {len(scales)} != channels {rec['channels']}"
                )
                continue
            out = dequant_int8(cols, fname, rec["channels"], scales)
        elif rec["dtype"] == "int4":
            out = dequant_int4(cols, fname, rec["channels"])
        else:
            errs.append(f"{name}/{fname}: unknown dtype {rec['dtype']}")
            continue
        py_b64 = fp32_b64(out)
        if py_b64 != expected["expected_fp32_b64"]:
            errs.append(
                f"{name}/{fname}: PY byte mismatch (got {py_b64[:24]}... "
                f"want {expected['expected_fp32_b64'][:24]}...)"
            )
        # ---------- C decoder cross-check ----------
        if os.path.exists(C_RUNNER):
            try:
                c_bytes = c_decode(fpath, fname)
                c_b64 = base64.b64encode(c_bytes).decode("ascii")
                if c_b64 != expected["expected_fp32_b64"]:
                    errs.append(
                        f"{name}/{fname}: C byte mismatch (got {c_b64[:24]}... "
                        f"want {expected['expected_fp32_b64'][:24]}...)"
                    )
            except subprocess.CalledProcessError as e:
                errs.append(f"{name}/{fname}: C decoder errored: {e.stderr!r}")
    return (len(errs) == 0), errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate fixtures before verifying")
    args = ap.parse_args()

    if args.rebuild:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "generate_fixtures.py")],
            check=True,
        )

    with open(os.path.join(HERE, "conformance.json")) as fh:
        manifest = json.load(fh)
    passed = 0
    failed = 0
    for case in manifest["cases"]:
        ok, errs = verify_case(case)
        if ok:
            passed += 1
            print(f"PASS {case['name']}")
        else:
            failed += 1
            print(f"FAIL {case['name']}")
            for e in errs:
                print(f"     {e}")
    print(f"---\n{passed}/{passed + failed} cases passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
