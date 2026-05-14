# SplatForge fixture corpus

Tiny, deterministic Gaussian Splat assets used by unit, integration, and
golden tests. Everything in `tiny/`, `invalid/`, and the public part of
`corpus/` is **generated** by `build.py` — do not hand-edit the binaries.

## Layout

```
fixtures/
  build.py                  generator (run with python3)
  tiny/                     3-splat scenes in every supported format
    basic_binary.ply        binary LE, Inria 3DGS layout (canonical)
    basic_ascii.ply         same scene, ASCII PLY
    basic.spz               SPZ tombstone (real .spz comes from the Rust crate)
    basic_khr.gltf          minimal KHR_gaussian_splatting glTF
    basic_splat.usda        USDA stub (USD path is deferred)
  invalid/                  malformed/edge-case inputs for failure tests
    missing_rotation.ply    header valid, rot_* fields removed
    missing_scale.ply       header valid, scale_* fields removed
    nan_position.ply        first splat x = NaN (raw 0x7fc00000)
    extreme_outlier.ply     third splat at (1e6, 1e6, 1e6)
    floater_cluster.ply     40 tight + 10 distant splats (floater detector)
    truncated_binary.ply    valid header, payload cut in half
    unsupported_khr_version.gltf   KHR_gaussian_splatting version=999.0
  corpus/                   "real-world" placeholders (design partners replace)
    product_scan.ply        currently a copy of basic_binary.ply
    indoor_room.ply         currently a copy of basic_binary.ply
    person_scan.ply         currently a copy of basic_binary.ply
  golden/                   expected outputs for byte-diff tests
    expected_reports/basic_binary.analyze.json   hand-authored, hash TODO
    expected_gltf/basic_binary.gltf              CI regenerates
    expected_frames/                             populated by viewer-parity job
  private/                  reserved for NDA assets (not committed; gitignored)
```

## Regenerate

```bash
python3 fixtures/build.py
```

The script is pure-Python stdlib (no numpy, no torch) and produces byte-identical
output on Linux and macOS. Re-run it any time you add a new fixture; CI runs it
before integration tests as a sanity check.

## Conventions

* Coordinate system: right-handed, Y-up.
* Scales are log-space (`scale_* = ln(sigma)`), opacities are logit-space.
* SH layout: 1 DC term (RGB) + 45 rest terms for degree-3 (Inria 3DGS).
* Quaternions are stored `(w, x, y, z)` and assumed normalized.
* All fixtures fit in well under 100 KB so they can live in git.

## Private corpus

Design-partner captures live in `fixtures/private/` and are NOT committed.
Place files there locally and reference them via the env var
`SPLATFORGE_PRIVATE_CORPUS` from scripts that need NDA scenes. CI skips any
test that depends on private assets when the directory is empty.

## Licenses

Generated fixtures are released under the repo's main license (see
`/LICENSE`). Once real captures land in `corpus/`, each scene must ship with
a sibling `<name>.LICENSE.txt` describing source, author, and redistribution
terms.

## TODOs

* Regenerate the `blake3:` hash inside
  `golden/expected_reports/basic_binary.analyze.json` once the analyzer is
  wired up (the placeholder is `blake3:PLACEHOLDER_REGENERATE`).
* Replace the three `corpus/*.ply` placeholders with real captures.
* Produce a real `tiny/basic.spz` via `splatforge convert` and check it in.
