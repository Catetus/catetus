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
    splatbench_specular_proxy  (12K splats, view-dep SH cluster)   class: specular highlights
    splatbench_foliage_proxy   (30K thin anisotropic splats)       class: dense translucency
    splatbench_lowlight_proxy  (8K dark high-SH splats)            class: narrow dynamic range
    splatbench_portrait_proxy  (12K, foreground/background split)  class: salient region

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


def write_splat_ex(
    buf: bytearray,
    rng: random.Random,
    x: float, y: float, z: float,
    color: tuple,
    scale: tuple,            # (sx, sy, sz)
    opacity: float,
    rotation: tuple = None,  # (qw, qx, qy, qz) — normalized inside
    sh_rest_fn=None,         # callable(rng, idx) -> float, idx in 0..44
) -> None:
    """Append one splat with explicit anisotropic scale, rotation, SH-rest.

    `scale` components are stored as `ln(s)` (matches Inria convention).
    `rotation` is normalized before writing.
    `sh_rest_fn` is invoked 45 times with (rng, i). If None, defaults to the
    existing uniform(-0.05, 0.05) jitter so callers that don't care can omit it.
    """
    buf += struct.pack("<fff", x, y, z)
    buf += struct.pack("<fff", 0.0, 0.0, 0.0)
    buf += struct.pack("<fff", color[0], color[1], color[2])
    if sh_rest_fn is None:
        for _ in range(45):
            buf += struct.pack("<f", rng.uniform(-0.05, 0.05))
    else:
        for i in range(45):
            buf += struct.pack("<f", sh_rest_fn(rng, i))
    buf += struct.pack("<f", logit(opacity))
    sx, sy, sz = scale
    buf += struct.pack("<fff", math.log(sx), math.log(sy), math.log(sz))
    if rotation is None:
        qw = 1.0
        qx = rng.uniform(-0.02, 0.02)
        qy = rng.uniform(-0.02, 0.02)
        qz = rng.uniform(-0.02, 0.02)
    else:
        qw, qx, qy, qz = rotation
    nrm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if nrm == 0.0:
        qw, qx, qy, qz, nrm = 1.0, 0.0, 0.0, 0.0, 1.0
    buf += struct.pack("<ffff", qw / nrm, qx / nrm, qy / nrm, qz / nrm)


def random_unit_quat(rng: random.Random) -> tuple:
    """Uniform random unit quaternion (Shoemake)."""
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    s1 = math.sqrt(1 - u1)
    s2 = math.sqrt(u1)
    return (
        s1 * math.sin(2 * math.pi * u2),
        s1 * math.cos(2 * math.pi * u2),
        s2 * math.sin(2 * math.pi * u3),
        s2 * math.cos(2 * math.pi * u3),
    )


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


def gen_specular(n_base: int, n_highlight: int, seed: int) -> bytes:
    """Heavy specular highlights — base sphere + a cluster of view-dependent splats.

    The base is ~`n_base` splats arranged on a unit sphere with mild SH jitter.
    A cluster of `n_highlight` splats has high-opacity, very small scale, and
    strongly directional SH2/SH3 magnitudes concentrated along a single SH band
    component. This stresses any preset that aggressively reduces SH degree
    (e.g. web-mobile / size-min): the highlight direction collapses when SH2/3
    are dropped, which surfaces as a large ΔE94 spike in the fidelity runner.
    """
    rng = random.Random(seed)
    total = n_base + n_highlight
    buf = bytearray(HEADER_TEMPLATE.format(n=total).encode("utf-8"))
    # Pick a deterministic "view direction" the highlights are tuned for; we
    # pour magnitude into SH bands (indices 6..8 of f_rest = first SH2 triplet,
    # indices 15..17 = first SH3 triplet — both per-channel, repeated 3x).
    for _ in range(n_base):
        # uniform on sphere of radius 1.0
        u = rng.random()
        v = rng.random()
        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1)
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        cr = 0.45 + 0.10 * (x)
        cg = 0.45 + 0.10 * (y)
        cb = 0.45 + 0.10 * (z)
        s = 0.012 + rng.random() * 0.008
        op = 0.55 + rng.random() * 0.35
        write_splat_ex(
            buf, rng, x, y, z, (cr, cg, cb), (s, s, s), op,
            rotation=None, sh_rest_fn=None,
        )
    # Highlight cluster: tight patch on +X side, near surface.
    for _ in range(n_highlight):
        # cluster within a 0.2-radius spherical cap at (1, 0, 0)
        dx = rng.uniform(-0.10, 0.10)
        dy = rng.uniform(-0.20, 0.20)
        dz = rng.uniform(-0.20, 0.20)
        x = 1.0 + dx
        y = dy
        z = dz
        cr = 0.85 + rng.random() * 0.10  # near-white DC
        cg = 0.85 + rng.random() * 0.10
        cb = 0.80 + rng.random() * 0.15
        s = 0.0035 + rng.random() * 0.0025  # very small
        op = 0.90 + rng.random() * 0.08

        def sh(rng_, i, _r=rng):
            # SH-rest layout for 3 channels × 15 coeffs = 45 floats, interleaved
            # as [c0_sh1_x, c0_sh1_y, c0_sh1_z, c0_sh2_..., c1_..., c2_...]
            # depending on writer; the Inria/Inria-derived layout used here is
            # 15 coeffs per channel laid out as channel-major.
            # We bias the first SH2 coeff (idx 3 within channel, i.e. global
            # indices 3, 18, 33 for R,G,B) and first SH3 coeff (idx 8 within
            # channel → global 8, 23, 38). Other rest coeffs stay small.
            channel = i // 15  # 0=R, 1=G, 2=B
            within = i % 15
            base = _r.uniform(-0.03, 0.03)
            if within == 3:        # first SH-degree-2 coeff
                return base + 1.2 if channel != 2 else base + 1.0
            if within == 8:        # first SH-degree-3 coeff
                return base + 0.9 if channel != 2 else base + 0.7
            return base

        write_splat_ex(
            buf, rng, x, y, z, (cr, cg, cb), (s, s, s), op,
            rotation=None, sh_rest_fn=sh,
        )
    return bytes(buf)


def gen_foliage(n: int, seed: int) -> bytes:
    """Dense overlapping translucency — thin, elongated, mid-opacity splats.

    ~`n` heavily anisotropic splats (effectively oriented disks) with random
    orientations packed into a 2×2×2 cube. Mid-opacity (0.3..0.6) forces the
    renderer to actually sort and blend rather than alpha-test through; aggressive
    opacity-prune or rejected back-to-front sort surfaces as bleeding/ghosting.
    """
    rng = random.Random(seed)
    buf = bytearray(HEADER_TEMPLATE.format(n=n).encode("utf-8"))
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        # leaf-green palette with some variance
        cr = 0.10 + rng.random() * 0.20
        cg = 0.30 + rng.random() * 0.35
        cb = 0.10 + rng.random() * 0.20
        # anisotropic scale: one axis 50× the others → disk/needle
        scale = (0.001 + rng.random() * 0.0005,
                 0.001 + rng.random() * 0.0005,
                 0.045 + rng.random() * 0.010)
        op = 0.30 + rng.random() * 0.30
        write_splat_ex(
            buf, rng, x, y, z, (cr, cg, cb), scale, op,
            rotation=random_unit_quat(rng), sh_rest_fn=None,
        )
    return bytes(buf)


def gen_lowlight(n: int, seed: int) -> bytes:
    """Narrow dynamic range — very dark DC, high-magnitude SH.

    `n` splats with DC color components in [0, 0.15] (so the bulk of the u8
    range is unused) but with strong SH-rest values so view-dependent shading
    pushes effective brightness up. Stresses any u8 color quantization that
    assumes [0,1] full-range mapping — the dark base gets crushed into ~38
    discrete levels instead of 256.
    """
    rng = random.Random(seed)
    buf = bytearray(HEADER_TEMPLATE.format(n=n).encode("utf-8"))
    for _ in range(n):
        # roughly uniform in a 1.5-radius sphere
        u = rng.random()
        v = rng.random()
        w = rng.random()
        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1)
        r = 1.5 * (w ** (1 / 3))
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        cr = rng.random() * 0.15
        cg = rng.random() * 0.15
        cb = rng.random() * 0.15
        s = 0.010 + rng.random() * 0.015
        op = 0.55 + rng.random() * 0.40

        def sh(rng_, i, _r=rng):
            # Large magnitude across all SH bands so view-dependent term
            # dominates the appearance — the metric will fire if any preset
            # quantizes f_rest to fewer levels than expected.
            return _r.uniform(-0.6, 0.6)

        write_splat_ex(
            buf, rng, x, y, z, (cr, cg, cb), (s, s, s), op,
            rotation=None, sh_rest_fn=sh,
        )
    return bytes(buf)


def gen_portrait(n_face: int, n_wall: int, seed: int) -> bytes:
    """Salient-region scene — small dense "face" + large sparse "wall" backdrop.

    The foreground is a tight ellipsoidal cluster of high-opacity, varied-color
    splats simulating skin/hair detail. The background is a 6×6 plane at z=-3
    of low-opacity gradient splats. A saliency-aware pruner should preserve
    the foreground; a naive opacity- or density-based pruner will treat them
    similarly and degrade the face.
    """
    rng = random.Random(seed)
    total = n_face + n_wall
    buf = bytearray(HEADER_TEMPLATE.format(n=total).encode("utf-8"))
    # Foreground: ellipsoid at origin, radii (0.18, 0.25, 0.18)
    for _ in range(n_face):
        # rejection-sample inside the ellipsoid
        while True:
            ex = rng.uniform(-1, 1)
            ey = rng.uniform(-1, 1)
            ez = rng.uniform(-1, 1)
            if ex * ex + ey * ey + ez * ez <= 1.0:
                break
        x = 0.18 * ex
        y = 0.25 * ey
        z = 0.18 * ez
        # skin/hair colors with strong variance
        kind = rng.random()
        if kind < 0.6:
            cr = 0.70 + rng.random() * 0.20
            cg = 0.55 + rng.random() * 0.20
            cb = 0.45 + rng.random() * 0.15
        elif kind < 0.8:
            cr = 0.20 + rng.random() * 0.15
            cg = 0.15 + rng.random() * 0.10
            cb = 0.10 + rng.random() * 0.08
        else:
            # eye/lip highlights
            cr = 0.10 + rng.random() * 0.10
            cg = 0.05 + rng.random() * 0.10
            cb = 0.05 + rng.random() * 0.10
        s = 0.0040 + rng.random() * 0.0030
        op = 0.85 + rng.random() * 0.12
        write_splat_ex(
            buf, rng, x, y, z, (cr, cg, cb), (s, s, s), op,
            rotation=None, sh_rest_fn=None,
        )
    # Background: 6×6 plane at z=-3, gradient color, lower opacity
    for _ in range(n_wall):
        x = rng.uniform(-3, 3)
        y = rng.uniform(-3, 3)
        z = -3.0 + rng.uniform(-0.02, 0.02)
        # left→right gradient teal→amber
        t = (x + 3.0) / 6.0
        cr = 0.25 + 0.45 * t
        cg = 0.35 + 0.20 * (1 - t)
        cb = 0.50 - 0.30 * t
        s = 0.040 + rng.random() * 0.020
        op = 0.20 + rng.random() * 0.20
        write_splat_ex(
            buf, rng, x, y, z, (cr, cg, cb), (s, s, s), op,
            rotation=None, sh_rest_fn=None,
        )
    return bytes(buf)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    scenes = [
        ("splatbench_product_proxy.ply",   gen_uniform_cube,    {"n": 10_000,   "half": 0.5,  "seed": 1}),
        ("splatbench_indoor_proxy.ply",    gen_indoor_room,     {"n": 100_000,                "seed": 2}),
        ("splatbench_outdoor_proxy.ply",   gen_outdoor,         {"n": 500_000,                "seed": 3}),
        ("splatbench_floater_proxy.ply",   gen_floater_heavy,   {"n_clean": 200_000, "n_floaters": 50_000, "seed": 4}),
        ("splatbench_dense_proxy.ply",     gen_uniform_cube,    {"n": 2_000_000, "half": 5.0, "seed": 5}),
        # v0.1.2 corpus extension: PRD failure-mode probes.
        ("splatbench_specular_proxy.ply",  gen_specular,        {"n_base": 10_000, "n_highlight": 2_000, "seed": 6}),
        ("splatbench_foliage_proxy.ply",   gen_foliage,         {"n": 30_000, "seed": 7}),
        ("splatbench_lowlight_proxy.ply",  gen_lowlight,        {"n": 8_000,  "seed": 8}),
        ("splatbench_portrait_proxy.ply",  gen_portrait,        {"n_face": 8_000, "n_wall": 4_000, "seed": 9}),
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
