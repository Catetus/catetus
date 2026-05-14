#!/usr/bin/env python3
"""
SplatForge fixture builder.

Generates tiny, deterministic Gaussian Splat fixtures for tests:
  - tiny/basic_binary.ply        3 splats, binary LE, Inria 3DGS layout
  - tiny/basic_ascii.ply         same scene, ASCII format
  - tiny/basic.spz               placeholder SPZ tombstone
  - tiny/basic_khr.gltf          minimal KHR_gaussian_splatting glTF
  - tiny/basic_splat.usda        USDA stub
  - invalid/missing_rotation.ply
  - invalid/missing_scale.ply
  - invalid/nan_position.ply
  - invalid/extreme_outlier.ply
  - invalid/floater_cluster.ply
  - invalid/truncated_binary.ply
  - invalid/unsupported_khr_version.gltf
  - corpus/{product_scan,indoor_room,person_scan}.ply  (placeholder copies)

Run:  python3 fixtures/build.py

Determinism: no time, no random; values are hard-coded constants and small
synthetic patterns derived from indices. Safe to regenerate in CI.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TINY = ROOT / "tiny"
INVALID = ROOT / "invalid"
CORPUS = ROOT / "corpus"
GOLDEN = ROOT / "golden"

for d in (TINY, INVALID, CORPUS,
          GOLDEN / "expected_reports",
          GOLDEN / "expected_gltf",
          GOLDEN / "expected_frames"):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Splat math helpers
# ---------------------------------------------------------------------------
def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


LN_SCALE = math.log(0.05)           # scale_* (log-space) so sigma ~ 0.05
DC = 0.5                            # f_dc_* RGB grey
OPACITY_LOGIT = logit(0.9)          # opacity in logit space
QUAT_IDENTITY = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)

# Inria 3DGS layout used throughout:
#   x y z nx ny nz
#   f_dc_0 f_dc_1 f_dc_2
#   f_rest_0 .. f_rest_44      (45 SH rest coefs, deg-3)
#   opacity
#   scale_0 scale_1 scale_2
#   rot_0 rot_1 rot_2 rot_3    (w, x, y, z)
N_REST = 45


def field_names(include_rot: bool = True, include_scale: bool = True) -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(N_REST)]
    names.append("opacity")
    if include_scale:
        names += [f"scale_{i}" for i in range(3)]
    if include_rot:
        names += [f"rot_{i}" for i in range(4)]
    return names


def make_splat(x: float, y: float, z: float,
               dc: tuple[float, float, float] = (DC, DC, DC),
               opacity_logit: float = OPACITY_LOGIT,
               scale: tuple[float, float, float] = (LN_SCALE,) * 3,
               quat: tuple[float, float, float, float] = QUAT_IDENTITY,
               sh_rest: list[float] | None = None,
               include_rot: bool = True,
               include_scale: bool = True) -> list[float]:
    rest = sh_rest if sh_rest is not None else [0.0] * N_REST
    row = [x, y, z, 0.0, 0.0, 0.0, dc[0], dc[1], dc[2], *rest, opacity_logit]
    if include_scale:
        row += list(scale)
    if include_rot:
        row += list(quat)
    return row


# ---------------------------------------------------------------------------
# PLY writers
# ---------------------------------------------------------------------------
def ply_header(format_line: str, vertex_count: int, fields: list[str]) -> bytes:
    lines = ["ply", format_line, f"element vertex {vertex_count}"]
    lines += [f"property float {n}" for n in fields]
    lines += ["end_header", ""]
    return ("\n".join(lines)).encode("ascii")


def write_binary_ply(path: Path, rows: list[list[float]],
                     include_rot: bool = True, include_scale: bool = True) -> None:
    fields = field_names(include_rot=include_rot, include_scale=include_scale)
    assert all(len(r) == len(fields) for r in rows), \
        f"row width mismatch: row={len(rows[0])} fields={len(fields)}"
    header = ply_header("format binary_little_endian 1.0", len(rows), fields)
    payload = io.BytesIO()
    for row in rows:
        for v in row:
            payload.write(struct.pack("<f", v))
    path.write_bytes(header + payload.getvalue())


def write_ascii_ply(path: Path, rows: list[list[float]]) -> None:
    fields = field_names()
    header = ply_header("format ascii 1.0", len(rows), fields)
    body_lines = []
    for row in rows:
        # ryu-ish: keep python's default repr; tests only need parseability
        body_lines.append(" ".join(repr(v) for v in row))
    body = ("\n".join(body_lines) + "\n").encode("ascii")
    path.write_bytes(header + body)


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------
def basic_three_splats(include_rot: bool = True, include_scale: bool = True) -> list[list[float]]:
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    return [make_splat(*p, include_rot=include_rot, include_scale=include_scale)
            for p in positions]


# ---------------------------------------------------------------------------
# Tiny corpus
# ---------------------------------------------------------------------------
def build_basic_binary() -> None:
    write_binary_ply(TINY / "basic_binary.ply", basic_three_splats())


def build_basic_ascii() -> None:
    write_ascii_ply(TINY / "basic_ascii.ply", basic_three_splats())


def build_basic_spz() -> None:
    """
    Tombstone SPZ. Real .spz is produced by the Rust crate; this is a 32-byte
    header followed by an empty zlib stream so loaders can at least sniff it.
    Layout (little-endian):
      u32  magic   = 0x474e5350  ('PSNG' read as LE 'SPNG'-ish)
      u32  version = 2
      u32  splat_count = 0
      u32  reserved[5] = 0
    """
    magic = 0x474E5350
    version = 2
    splat_count = 0
    header = struct.pack("<3I", magic, version, splat_count) + b"\x00" * (32 - 12)
    payload = zlib.compress(b"")
    (TINY / "basic.spz").write_bytes(header + payload)


def build_basic_khr_gltf() -> None:
    """
    Minimal glTF 2.0 with KHR_gaussian_splatting. The buffer holds raw float32
    interleaved POSITION(3) + _ROTATION(4) + _SCALE(3) + _OPACITY(1) + _COLOR_DC(3)
    for 3 splats = 14 floats * 3 = 168 bytes.
    """
    stride_floats = 3 + 4 + 3 + 1 + 3
    buf = io.BytesIO()
    for (x, y, z) in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
        floats = [x, y, z,
                  *QUAT_IDENTITY,
                  math.exp(LN_SCALE), math.exp(LN_SCALE), math.exp(LN_SCALE),
                  0.9,
                  DC, DC, DC]
        for f in floats:
            buf.write(struct.pack("<f", f))
    raw = buf.getvalue()
    byte_length = len(raw)
    uri = "data:application/octet-stream;base64," + base64.b64encode(raw).decode("ascii")

    stride = stride_floats * 4
    bv = lambda offset, count_floats: {
        "buffer": 0,
        "byteOffset": offset * 4,
        "byteLength": count_floats * 4 + (3 - 1) * stride,  # 3 elements interleaved
        "byteStride": stride,
        "target": 34962,
    }
    # Simpler: one bufferView spanning the whole buffer with stride; accessors point in.
    buffer_view = {
        "buffer": 0,
        "byteOffset": 0,
        "byteLength": byte_length,
        "byteStride": stride,
        "target": 34962,
    }

    def accessor(offset_floats: int, count: int, kind: str, components: int) -> dict:
        return {
            "bufferView": 0,
            "byteOffset": offset_floats * 4,
            "componentType": 5126,  # FLOAT
            "count": count,
            "type": kind,
        }

    gltf = {
        "asset": {"version": "2.0", "generator": "SplatForge fixtures/build.py"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "extensionsRequired": ["KHR_gaussian_splatting"],
        "buffers": [{"uri": uri, "byteLength": byte_length}],
        "bufferViews": [buffer_view],
        "accessors": [
            accessor(0, 3, "VEC3", 3),   # POSITION
            accessor(3, 3, "VEC4", 4),   # _ROTATION
            accessor(7, 3, "VEC3", 3),   # _SCALE
            accessor(10, 3, "SCALAR", 1),  # _OPACITY
            accessor(11, 3, "VEC3", 3),  # _COLOR_DC
        ],
        "meshes": [{
            "name": "splats",
            "primitives": [{
                "mode": 0,  # POINTS
                "attributes": {
                    "POSITION": 0,
                },
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "version": "1.0",
                        "splatCount": 3,
                        "attributes": {
                            "POSITION": 0,
                            "_ROTATION": 1,
                            "_SCALE": 2,
                            "_OPACITY": 3,
                            "_COLOR_DC": 4,
                        },
                    }
                },
            }],
        }],
        "nodes": [{"mesh": 0, "name": "splatNode"}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    (TINY / "basic_khr.gltf").write_text(
        json.dumps(gltf, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_basic_splat_usda() -> None:
    text = (
        "#usda 1.0\n"
        'def "scene" {\n'
        '  def ParticleField3DGaussianSplat "splats" {\n'
        "    int splatCount = 3\n"
        "  }\n"
        "}\n"
    )
    (TINY / "basic_splat.usda").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Invalid corpus
# ---------------------------------------------------------------------------
def build_missing_rotation() -> None:
    rows = basic_three_splats(include_rot=False)
    write_binary_ply(INVALID / "missing_rotation.ply", rows, include_rot=False)


def build_missing_scale() -> None:
    rows = basic_three_splats(include_scale=False)
    write_binary_ply(INVALID / "missing_scale.ply", rows, include_scale=False)


def build_nan_position() -> None:
    rows = basic_three_splats()
    # Overwrite x of first splat with raw NaN bits (0x7fc00000) after writing.
    # Simpler: write normally, then patch the first float in the payload.
    path = INVALID / "nan_position.ply"
    write_binary_ply(path, rows)
    data = bytearray(path.read_bytes())
    # Find end_header\n
    marker = b"end_header\n"
    idx = data.find(marker)
    assert idx >= 0
    payload_start = idx + len(marker)
    # First float in payload is x of vertex 0.
    data[payload_start:payload_start + 4] = struct.pack("<I", 0x7FC00000)
    path.write_bytes(bytes(data))


def build_extreme_outlier() -> None:
    rows = basic_three_splats()
    # third splat -> huge coords
    rows[2][0] = 1e6
    rows[2][1] = 1e6
    rows[2][2] = 1e6
    write_binary_ply(INVALID / "extreme_outlier.ply", rows)


def build_floater_cluster() -> None:
    rows: list[list[float]] = []
    # 40 tight around origin, deterministic spiral within radius 0.1
    for i in range(40):
        t = i / 40.0
        r = 0.1 * t
        a = i * 0.7
        x = r * math.cos(a)
        y = r * math.sin(a)
        z = 0.05 * math.sin(a * 2.0)
        rows.append(make_splat(x, y, z))
    # 10 scattered floaters at radius ~50
    for i in range(10):
        a = i * 0.97
        b = i * 1.31
        x = 50.0 * math.cos(a)
        y = 50.0 * math.sin(b)
        z = 50.0 * math.sin(a + b)
        rows.append(make_splat(x, y, z))
    write_binary_ply(INVALID / "floater_cluster.ply", rows)


def build_truncated_binary() -> None:
    src = (TINY / "basic_binary.ply").read_bytes()
    marker = b"end_header\n"
    idx = src.find(marker)
    payload_start = idx + len(marker)
    header = src[:payload_start]
    payload = src[payload_start:]
    truncated = header + payload[: len(payload) // 2]
    (INVALID / "truncated_binary.ply").write_bytes(truncated)


def build_unsupported_khr_version() -> None:
    gltf = json.loads((TINY / "basic_khr.gltf").read_text(encoding="utf-8"))
    gltf["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]["version"] = "999.0"
    (INVALID / "unsupported_khr_version.gltf").write_text(
        json.dumps(gltf, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Corpus placeholders
# ---------------------------------------------------------------------------
def build_corpus_placeholders() -> None:
    src = TINY / "basic_binary.ply"
    for name in ("product_scan.ply", "indoor_room.ply", "person_scan.ply"):
        shutil.copyfile(src, CORPUS / name)


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------
def build_golden_report() -> None:
    """
    Hand-authored expected analyze report for tiny/basic_binary.ply.
    Keys are emitted in stable lexical order. Hash is a placeholder; the
    Rust analyze command regenerates the real BLAKE3 in CI.
    """
    # NOTE: hash MUST be regenerated by CI; see TODO below.
    report = {
        "attributes": {
            "color_dc": True,
            "opacity": True,
            "position": True,
            "rotation": True,
            "scale": True,
            "sh_rest": False,
        },
        "boundingBox": {
            "max": [1.0, 1.0, 0.0],
            "min": [0.0, 0.0, 0.0],
        },
        "coordinateSystem": {"handedness": "right", "up": "Y"},
        "estimatedMemory": {"ramMb": 1, "vramMb": 1},
        "fileSize": (TINY / "basic_binary.ply").stat().st_size,
        "format": "ply",
        # TODO(splatforge-core): regenerate this hash from the canonical
        # report bytes once the analyzer is wired up. Until then, golden.sh
        # treats the placeholder as a wildcard.
        "hash": "blake3:PLACEHOLDER_REGENERATE",
        "opacityDistribution": {"max": 0.9, "mean": 0.9, "median": 0.9, "min": 0.9},
        "recommendations": [],
        "scaleDistribution": {
            "max": [0.05, 0.05, 0.05],
            "mean": [0.05, 0.05, 0.05],
            "min": [0.05, 0.05, 0.05],
        },
        "schemaVersion": "1",
        "shDegree": 0,
        "splatCount": 3,
        "warnings": [],
    }
    out = GOLDEN / "expected_reports" / "basic_binary.analyze.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")


def build_golden_gltf_placeholder() -> None:
    placeholder = {
        "asset": {
            "version": "2.0",
            "generator": "SplatForge placeholder — CI regenerates this file",
        },
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "scenes": [{"nodes": []}],
        "scene": 0,
        "nodes": [],
        "meshes": [],
    }
    (GOLDEN / "expected_gltf" / "basic_binary.gltf").write_text(
        json.dumps(placeholder, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def build_golden_frames_gitkeep() -> None:
    (GOLDEN / "expected_frames" / ".gitkeep").write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    steps = [
        ("tiny/basic_binary.ply",        build_basic_binary),
        ("tiny/basic_ascii.ply",         build_basic_ascii),
        ("tiny/basic.spz",               build_basic_spz),
        ("tiny/basic_khr.gltf",          build_basic_khr_gltf),
        ("tiny/basic_splat.usda",        build_basic_splat_usda),
        ("invalid/missing_rotation.ply", build_missing_rotation),
        ("invalid/missing_scale.ply",    build_missing_scale),
        ("invalid/nan_position.ply",     build_nan_position),
        ("invalid/extreme_outlier.ply",  build_extreme_outlier),
        ("invalid/floater_cluster.ply",  build_floater_cluster),
        ("invalid/truncated_binary.ply", build_truncated_binary),
        ("invalid/unsupported_khr_version.gltf", build_unsupported_khr_version),
        ("corpus/*.ply",                 build_corpus_placeholders),
        ("golden/expected_reports/basic_binary.analyze.json", build_golden_report),
        ("golden/expected_gltf/basic_binary.gltf",  build_golden_gltf_placeholder),
        ("golden/expected_frames/.gitkeep",         build_golden_frames_gitkeep),
    ]
    for label, fn in steps:
        fn()
        print(f"  built {label}")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
