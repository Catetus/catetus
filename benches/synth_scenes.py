#!/usr/bin/env python3
"""
Generate synthetic Gaussian-splat PLY scenes for SplatBench v0.

Each scene is deterministic (fixed seed), uses the Inria 3DGS binary PLY layout,
and is sized to exercise a different PRD corpus class:

    splatbench_product_proxy   (10K splats, tight bbox)            class: product scan
    splatbench_indoor_proxy    (100K splats, indoor-room bbox)     class: indoor real estate
    splatbench_outdoor_proxy   (500K splats, large bbox)           class: outdoor scene
    splatbench_floater_proxy   (250K total, 50K outliers)          class: noisy capture
    splatbench_dense_proxy     (2M splats, dense bbox)             class: dense large scene

Reproducibility: every output is a function of the (seed, splat_count, class) tuple.
Two runs of this script produce byte-identical PLY files.

Usage:
    python3 synth_scenes.py /path/to/output/dir
"""
import math
import os
import random
import struct
import sys

# Inria 3DGS field order (same as fixtures/tiny/basic_binary.ply)
HEADER_TEMPLATE = """ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
""" + "\n".join(f"property float f_rest_{i}" for i in range(45)) + """
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""


def logit(x: float) -> float:
    """Inverse of the sigmoid applied on import. Maps [0,1] -> R."""
    return math.log(x / (1.0 - x))


def write_splat(buf: bytearray, rng: random.Random, x: float, y: float, z: float,
                color: tuple, scale: float, opacity: float, sh_rest: int = 45) -> None:
    """Append one splat (62 floats) to buf in binary LE format."""
    # position
    buf += struct.pack("<fff", x, y, z)
    # normals (unused, write zeros to match fixture layout)
    buf += struct.pack("<fff", 0.0, 0.0, 0.0)
    # f_dc_0..2 (color DC term)
    buf += struct.pack("<fff", color[0], color[1], color[2])
    # f_rest_0..44 (SH-3 rest), small random values
    for _ in range(sh_rest):
        buf += struct.pack("<f", rng.uniform(-0.05, 0.05))
    # opacity stored as logit, so it sigmoids back to opacity on read
    buf += struct.pack("<f", logit(opacity))
    # scale stored as ln, exps back to scale on read
    ln_s = math.log(scale)
    buf += struct.pack("<fff", ln_s, ln_s, ln_s)
    # rotation (w, x, y, z) — random small jitter around identity
    qw = 1.0
    qx = rng.uniform(-0.02, 0.02)
    qy = rng.uniform(-0.02, 0.02)
    qz = rng.uniform(-0.02, 0.02)
    nrm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    buf += struct.pack("<ffff", qw / nrm, qx / nrm, qy / nrm, qz / nrm)


def gen_uniform_cube(n: int, half: float, seed: int) -> bytes:
    rng = random.Random(seed)
    buf = bytearray(HEADER_TEMPLATE.format(n=n).encode("utf-8"))
    for _ in range(n):
        x = rng.uniform(-half, half)
        y = rng.uniform(-half, half)
        z = rng.uniform(-half, half)
        # color: gradient by position so analyze sees variance
        cr = 0.5 + 0.3 * (x / half)
        cg = 0.5 + 0.3 * (y / half)
        cb = 0.5 + 0.3 * (z / half)
        scale = 0.005 + rng.random() * 0.015
        op = 0.20 + rng.random() * 0.60
        write_splat(buf, rng, x, y, z, (cr, cg, cb), scale, op)
    return bytes(buf)


def gen_indoor_room(n: int, seed: int) -> bytes:
    """6m x 4m x 3m room — walls denser than centre."""
    rng = random.Random(seed)
    buf = bytearray(HEADER_TEMPLATE.format(n=n).encode("utf-8"))
    for _ in range(n):
        # Bias to walls: probability of being near a wall scales with 1 - dist
        if rng.random() < 0.6:
            # near a wall
            wall = rng.randint(0, 5)
            if wall == 0:   x, y, z = -3.0 + rng.uniform(-0.05, 0.05), rng.uniform(-2, 2), rng.uniform(-1.5, 1.5)
            elif wall == 1: x, y, z =  3.0 + rng.uniform(-0.05, 0.05), rng.uniform(-2, 2), rng.uniform(-1.5, 1.5)
            elif wall == 2: x, y, z = rng.uniform(-3, 3), -2.0 + rng.uniform(-0.05, 0.05), rng.uniform(-1.5, 1.5)
            elif wall == 3: x, y, z = rng.uniform(-3, 3),  2.0 + rng.uniform(-0.05, 0.05), rng.uniform(-1.5, 1.5)
            elif wall == 4: x, y, z = rng.uniform(-3, 3), rng.uniform(-2, 2), -1.5 + rng.uniform(-0.05, 0.05)
            else:           x, y, z = rng.uniform(-3, 3), rng.uniform(-2, 2),  1.5 + rng.uniform(-0.05, 0.05)
        else:
            # interior volume
            x = rng.uniform(-2.8, 2.8)
            y = rng.uniform(-1.8, 1.8)
            z = rng.uniform(-1.3, 1.3)
        cr = 0.40 + rng.random() * 0.40
        cg = 0.35 + rng.random() * 0.40
        cb = 0.30 + rng.random() * 0.45
        scale = 0.008 + rng.random() * 0.020
        op = 0.50 + rng.random() * 0.45
        write_splat(buf, rng, x, y, z, (cr, cg, cb), scale, op)
    return bytes(buf)


def gen_outdoor(n: int, seed: int) -> bytes:
    """50m x 50m x 20m outdoor patch, ground-biased, sky-thin."""
    rng = random.Random(seed)
    buf = bytearray(HEADER_TEMPLATE.format(n=n).encode("utf-8"))
    for _ in range(n):
        x = rng.uniform(-25, 25)
        y = rng.uniform(-25, 25)
        # exponential bias toward ground
        z = (rng.random() ** 2) * 18.0 + rng.uniform(-1, 1)
        sky = z > 8.0
        cr = 0.5 + rng.uniform(-0.2, 0.2) if not sky else 0.5 + 0.2 * rng.random()
        cg = 0.5 + rng.uniform(-0.2, 0.2) if not sky else 0.6 + 0.2 * rng.random()
        cb = 0.4 + rng.uniform(-0.2, 0.2) if not sky else 0.8 + 0.2 * rng.random()
        scale = 0.03 + rng.random() * 0.10
        op = 0.30 + rng.random() * 0.50
        write_splat(buf, rng, x, y, z, (cr, cg, cb), scale, op)
    return bytes(buf)


def gen_floater_heavy(n_clean: int, n_floaters: int, seed: int) -> bytes:
    """200K-ish clean splats in 4x4x4 box + 50K low-opacity outliers scattered to radius 80."""
    rng = random.Random(seed)
    total = n_clean + n_floaters
    buf = bytearray(HEADER_TEMPLATE.format(n=total).encode("utf-8"))
    # clean cluster
    for _ in range(n_clean):
        x = rng.uniform(-2, 2)
        y = rng.uniform(-2, 2)
        z = rng.uniform(-2, 2)
        cr = 0.5 + rng.uniform(-0.2, 0.2)
        cg = 0.5 + rng.uniform(-0.2, 0.2)
        cb = 0.5 + rng.uniform(-0.2, 0.2)
        scale = 0.010 + rng.random() * 0.030
        op = 0.60 + rng.random() * 0.35
        write_splat(buf, rng, x, y, z, (cr, cg, cb), scale, op)
    # floaters: low opacity, scattered far, prime targets for OpacityPrune / FloaterPrune
    for _ in range(n_floaters):
        # uniform on sphere of radius up to 80
        u = rng.random()
        v = rng.random()
        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1)
        r = 20 + rng.random() * 60
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        scale = 0.05 + rng.random() * 0.30
        op = 0.005 + rng.random() * 0.040   # very low opacity
        write_splat(buf, rng, x, y, z, (0.5, 0.5, 0.5), scale, op)
    return bytes(buf)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    scenes = [
        ("splatbench_product_proxy.ply",   gen_uniform_cube,    {"n": 10_000,   "half": 0.5,  "seed": 1}),
        ("splatbench_indoor_proxy.ply",    gen_indoor_room,     {"n": 100_000,                "seed": 2}),
        ("splatbench_outdoor_proxy.ply",   gen_outdoor,         {"n": 500_000,                "seed": 3}),
        ("splatbench_floater_proxy.ply",   gen_floater_heavy,   {"n_clean": 200_000, "n_floaters": 50_000, "seed": 4}),
        ("splatbench_dense_proxy.ply",     gen_uniform_cube,    {"n": 2_000_000, "half": 5.0, "seed": 5}),
    ]
    for name, fn, kwargs in scenes:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            print(f"  skip (exists): {name}")
            continue
        print(f"  generating {name} ...", flush=True)
        data = fn(**kwargs)
        with open(path, "wb") as f:
            f.write(data)
        print(f"     wrote {len(data):,} bytes")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    main(out)
