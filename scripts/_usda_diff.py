#!/usr/bin/env python3
"""Tiny USDA attribute-array diff used by `scripts/usdc-roundtrip.sh`.

We don't compare USDA text directly (whitespace, attribute order, layer
metadata, and float formatting all differ between authoring tools). Instead
we extract the five canonical ParticleField3DGaussianSplat arrays from each
file and compare them numerically within an epsilon.

Exits 0 iff the two files agree on every attribute within 1e-4 per channel.
"""

from __future__ import annotations

import re
import sys
from typing import Iterable

EPS = 1e-4


def parse_array(text: str, attr: str) -> list[list[float]] | list[float]:
    """Locate `... <attr> = [ ... ]` and return parsed numeric data.

    For tuple-valued attributes (positions, orientations, etc.) we return a
    list of lists. For flat-scalar attributes we return a list of floats.
    """
    # Find the attribute. Use a regex that tolerates whitespace and any USDA
    # array type-prefix (`point3f[]`, `quatf[]`, `float[]`, `color3f[]`,
    # `float3[]`).
    pat = re.compile(rf"\b{re.escape(attr)}\s*=\s*\[(.*?)\]", re.DOTALL)
    m = pat.search(text)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    if "(" in body:
        tuples: list[list[float]] = []
        for tup in re.findall(r"\(([^()]*)\)", body):
            tuples.append([float(x.strip()) for x in tup.split(",") if x.strip()])
        return tuples
    return [float(x.strip()) for x in body.split(",") if x.strip()]


ATTRS = ("points", "orientations", "scales", "opacities", "colorsDC")


def approx_eq(a: Iterable[float], b: Iterable[float]) -> bool:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        return False
    return all(abs(x - y) <= EPS for x, y in zip(a_list, b_list))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _usda_diff.py A.usda B.usda", file=sys.stderr)
        return 64
    a_text = open(sys.argv[1]).read()
    b_text = open(sys.argv[2]).read()
    ok = True
    for attr in ATTRS:
        a = parse_array(a_text, attr)
        b = parse_array(b_text, attr)
        if not a and not b:
            continue
        if not a or not b:
            print(f"  attr {attr}: present in one but not the other", file=sys.stderr)
            ok = False
            continue
        if isinstance(a[0], list):  # tuple-valued
            if len(a) != len(b):
                print(
                    f"  attr {attr}: tuple count {len(a)} != {len(b)}",
                    file=sys.stderr,
                )
                ok = False
                continue
            for i, (av, bv) in enumerate(zip(a, b)):
                if not approx_eq(av, bv):
                    print(
                        f"  attr {attr}[{i}]: {av} != {bv}",
                        file=sys.stderr,
                    )
                    ok = False
        else:  # flat
            if not approx_eq(a, b):
                # Find first mismatch for diagnostics.
                for i, (x, y) in enumerate(zip(a, b)):
                    if abs(x - y) > EPS:
                        print(
                            f"  attr {attr}[{i}]: {x} != {y}",
                            file=sys.stderr,
                        )
                        break
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
