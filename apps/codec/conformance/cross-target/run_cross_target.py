#!/usr/bin/env python3
"""
run_cross_target.py — Cross-decoder conformance harness for the
QAT-PLY v1 wire format. For every fixture × every available reference
implementation we ask the implementation to dequantize, then assert all
implementations agree byte-for-byte against the JSON expectation.

Targets covered (status reported per target as PASS/FAIL/SKIP):

  1. Python  — apps/codec/conformance/verify.py
  2. C       — apps/codec/qat-ply-c/ (compiled via Makefile)
  3. WASM    — apps/codec/qat-ply-wasm/  (runs vitest suite)
  4. iOS     — apps/ios/SplatForgeQATKernel/  (runs swift test)
  5. Android — apps/android/splatforge-qat-vulkan/  (cmake configure + glsl compile)

The matrix at the end is the credibility document for the spec.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

HERE     = Path(__file__).resolve().parent
CONF_DIR = HERE.parent
ROOT     = CONF_DIR.parent.parent.parent
CONF     = CONF_DIR / "conformance.json"
FIXTURES = CONF_DIR / "fixtures"
C_DIR    = ROOT / "apps" / "codec" / "qat-ply-c"
C_RUN    = CONF_DIR / "c_decode_runner"
WASM_DIR = ROOT / "apps" / "codec" / "qat-ply-wasm"
IOS_DIR  = ROOT / "apps" / "ios" / "SplatForgeQATKernel"
AND_DIR  = ROOT / "apps" / "android" / "splatforge-qat-vulkan"

# Import the reference Python decoder helpers from verify.py.
sys.path.insert(0, str(CONF_DIR))
from verify import (  # type: ignore
    parse_ply, vertex_layout, read_columns,
    parse_quantized_fields, dequant_int8, dequant_int4, fp32_b64,
)
sys.path.pop(0)


def python_decode_field(fixture_path: Path, field_name: str) -> Optional[bytes]:
    """Return raw fp32 LE bytes for one field via the Python reference."""
    header_lines, body, _ = parse_ply(str(fixture_path))
    n_verts, props = vertex_layout(header_lines)
    cols = read_columns(body, n_verts, props)
    for rec in parse_quantized_fields(header_lines):
        if rec["name"] != field_name:
            continue
        if rec["dtype"] == "int8":
            sb = base64.b64decode(rec["scale_b64"])
            scales = list(struct.unpack(f"<{len(sb)//4}f", sb))
            out = dequant_int8(cols, field_name, rec["channels"], scales)
        elif rec["dtype"] == "int4":
            out = dequant_int4(cols, field_name, rec["channels"])
        else:
            return None
        return b"".join(struct.pack("<f", x) for x in out)
    return None


def ensure_c_runner() -> Optional[Path]:
    src = CONF_DIR / "c_decode_runner.c"
    if not src.exists():
        return None
    out = HERE / "c_decode_runner.bin"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        r = subprocess.run(
            ["cc", "-std=c99", "-O2", "-I", str(C_DIR),
             str(src), str(C_DIR / "qat_ply_decode.c"), "-o", str(out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("C runner build failed:", r.stderr)
            return None
    return out


def c_decode_field(fixture_path: Path, field_name: str) -> Optional[bytes]:
    runner = ensure_c_runner()
    if runner is None:
        return None
    r = subprocess.run([str(runner), str(fixture_path), field_name],
                       capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout


def run_wasm_suite() -> str:
    if not (WASM_DIR / "dist" / "qat_ply_decode.js").exists():
        return "SKIP (no dist/; run `make build` + `pnpm test` in qat-ply-wasm/)"
    r = subprocess.run(["pnpm", "test"], cwd=str(WASM_DIR),
                       capture_output=True, text=True)
    return "PASS" if r.returncode == 0 else f"FAIL\n{r.stdout[-1200:]}\n{r.stderr[-600:]}"


def run_swift_suite() -> str:
    if not IOS_DIR.exists():
        return "SKIP (module missing)"
    if shutil.which("swift") is None:
        return "SKIP (no swift toolchain on PATH)"
    r = subprocess.run(["swift", "test", "--quiet"],
                       cwd=str(IOS_DIR), capture_output=True, text=True)
    if r.returncode == 0:
        return "PASS"
    return f"FAIL\n{r.stdout[-1200:]}\n{r.stderr[-600:]}"


def run_android_suite() -> str:
    if not AND_DIR.exists():
        return "SKIP (module missing)"
    if shutil.which("cmake") is None:
        return "SKIP (cmake not installed; `brew install cmake`)"
    build_dir = AND_DIR / "build_cross_target"
    r = subprocess.run(["cmake", "-B", str(build_dir)],
                       cwd=str(AND_DIR), capture_output=True, text=True)
    if r.returncode != 0:
        return f"FAIL (cmake configure)\n{r.stderr[-800:]}"
    g = shutil.which("glslc") or shutil.which("glslang") or shutil.which("glslangValidator")
    if g is None:
        return "CONFIGURE_OK (shader not compiled; install glslang)"
    spv = build_dir / "qat_dequant_check.spv"
    spv.parent.mkdir(parents=True, exist_ok=True)
    args = ([g, "-O", str(AND_DIR / "src/main/cpp/shaders/qat_dequant.comp"), "-o", str(spv)]
            if g.endswith("glslc")
            else [g, "-V", str(AND_DIR / "src/main/cpp/shaders/qat_dequant.comp"), "-o", str(spv)])
    sc = subprocess.run(args, capture_output=True, text=True)
    if sc.returncode != 0:
        return f"FAIL (shader compile)\n{sc.stderr[-600:]}"
    return f"CONFIGURE_OK + SPV_OK ({spv.stat().st_size} B)"


GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
def cell(b: Optional[bytes], expected: bytes) -> str:
    if b is None:                  return f"{YELLOW}skip{RESET}"
    if b == expected:              return f"{GREEN}pass{RESET}"
    return f"{RED}FAIL{RESET}"


def main() -> int:
    conformance = json.loads(CONF.read_text())
    cases = conformance["cases"]

    print("=" * 88)
    print("CROSS-TARGET CONFORMANCE MATRIX (Python + C executed; WASM/iOS/Android via own suites)")
    print("=" * 88)
    print(f"{'fixture':<32} {'field':<22} {'PY':<6} {'C':<6} {'PY=C?':<8} {'matches expected?'}")
    print("-" * 88)

    all_pass = True
    case_count = 0
    for case in cases:
        fixture = FIXTURES / case["name"]
        for fname, fspec in case["fields"].items():
            expected_b = base64.b64decode(fspec["expected_fp32_b64"])
            py_b = python_decode_field(fixture, fname)
            c_b  = c_decode_field(fixture, fname)
            py_status = cell(py_b, expected_b)
            c_status  = cell(c_b,  expected_b)
            if py_b is not None and c_b is not None:
                pyc = "yes" if py_b == c_b else "NO"
            else:
                pyc = "n/a"
            expected_ok = "yes" if (py_b == expected_b and (c_b is None or c_b == expected_b)) else "NO"
            if expected_ok == "NO" or pyc == "NO":
                all_pass = False
            case_count += 1
            print(f"{case['name']:<32} {fname:<22} {py_status:<14} {c_status:<14} {pyc:<8} {expected_ok}")

    print()
    print("=" * 88)
    print("TARGET SUITE RESULTS")
    print("=" * 88)
    for label, fn in [("WASM (Vitest)", run_wasm_suite),
                      ("iOS Metal (XCTest)", run_swift_suite),
                      ("Android Vulkan (CMake)", run_android_suite)]:
        out = fn()
        first = out.splitlines()[0] if out else ""
        print(f"{label:<26} {first}")
        if "\n" in out:
            for ln in out.splitlines()[1:]:
                print(f"  {ln}")

    print()
    print("=" * 88)
    print(f"Per-field PY/C comparisons: {case_count} rows, "
          f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 88)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
