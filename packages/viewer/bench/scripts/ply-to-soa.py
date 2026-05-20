#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert an Inria-format 3DGS PLY into the SoA chunk-bytes layout consumed
by ComputeDecodePipeline.

Mirrors packages/viewer/bench/compute-decode.bench.ts::buildSyntheticScene:
  POSITION  vec3 f32 = 12 B
  ROTATION  vec4 f32 = 16 B  (normalized to unit quaternion)
  SCALE     vec3 f32 = 12 B  (expf so we send linear scale)
  OPACITY   f32       =  4 B  (sigmoid so we send 0..1)
  COLOR_DC  vec3 f32 = 12 B  (RAW f_dc — the GPU decode kernel bakes
                              `0.5 + 0.28209479 * f_dc` in cs_decode.)

Usage: python ply-to-soa.py <input.ply> <output.bin>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

SH_C0 = 0.28209479177387814

def _read_header(fh):
    line = fh.readline().strip()
    if line != b"ply":
        raise ValueError(f"not a PLY: starts with {line!r}")
    fmt = fh.readline().strip()
    if fmt != b"format binary_little_endian 1.0":
        raise ValueError(f"unsupported PLY format: {fmt!r}")
    n = -1
    props = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("unexpected EOF in PLY header")
        s = line.decode().strip()
        if s == "end_header":
            break
        if s.startswith("element vertex "):
            n = int(s.split()[-1])
        elif s.startswith("property float "):
            props.append(s.split()[-1])
        elif s.startswith("comment"):
            continue
        elif s.startswith("element"):
            raise ValueError(f"unexpected element: {s!r}")
    if n < 0:
        raise ValueError("no vertex count in PLY")
    return n, props

def main():
    if len(sys.argv) != 3:
        print(__doc__); return 2
    src = Path(sys.argv[1]); dst = Path(sys.argv[2])
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr); return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fh:
        n, props = _read_header(fh)
        required = ["x","y","z","opacity","scale_0","scale_1","scale_2",
                    "rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"]
        for r in required:
            if r not in props:
                raise ValueError(f"missing property {r} in PLY")
        stride = len(props) * 4
        print(f"[ply->soa] n={n} stride={stride} props={len(props)}", file=sys.stderr)
        body = np.frombuffer(fh.read(n * stride), dtype=np.float32).reshape(n, len(props))
    idx = {name: props.index(name) for name in props}
    xyz = body[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32, copy=True)
    rot = body[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]].astype(np.float32, copy=True)
    log_scl = body[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]].astype(np.float32, copy=True)
    scl = np.exp(log_scl).astype(np.float32)
    op = (1.0 / (1.0 + np.exp(-body[:, idx["opacity"]]))).astype(np.float32)
    f_dc = body[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].astype(np.float32, copy=True)
    # COLOR_DC slot stores RAW f_dc; the GPU decode kernel (decode.wgsl::cs_decode)
    # applies `0.5 + SH_C0 * f_dc` and clamps to [0,1]. Matches the TS
    # `splatSceneToSoaChunk()` adapter in packages/viewer/src/loader/to-soa.ts.
    color_dc = f_dc
    rot_n = np.linalg.norm(rot, axis=1, keepdims=True)
    rot_n[rot_n == 0] = 1.0
    rot = (rot / rot_n).astype(np.float32)
    bbox_min = xyz.min(axis=0).tolist()
    bbox_max = xyz.max(axis=0).tolist()
    parts = [xyz.tobytes("C"), rot.tobytes("C"), scl.tobytes("C"), op.tobytes("C"), color_dc.tobytes("C")]
    with dst.open("wb") as out:
        for p in parts: out.write(p)
    total = sum(len(p) for p in parts)
    expected = n * (12 + 16 + 12 + 4 + 12)
    assert total == expected, (total, expected)
    meta = {
        "splatCount": int(n),
        "bbox": {"min": bbox_min, "max": bbox_max},
        "byteLength": total,
        "source": str(src.name),
        "layout": {
            "positions": {"byteOffset": 0, "byteLength": n*12, "componentType": 5126},
            "rotations": {"byteOffset": n*12, "byteLength": n*16, "componentType": 5126},
            "scales":    {"byteOffset": n*28, "byteLength": n*12, "componentType": 5126},
            "opacities": {"byteOffset": n*40, "byteLength": n*4,  "componentType": 5126},
            "colorDC":   {"byteOffset": n*44, "byteLength": n*12, "componentType": 5126},
        },
    }
    with dst.with_suffix(".meta.json").open("w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"[ply->soa] wrote {dst} ({total} B) + meta", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
