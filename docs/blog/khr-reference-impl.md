# Submitting the first KHR_gaussian_splatting reference implementation

*Draft, May 2026 — author: SplatForge engineering*

Six weeks ago the Khronos Group cut a release candidate for
`KHR_gaussian_splatting`, the first standardised glTF extension for 3D
Gaussian splatting. Like every Khronos extension, the RC needs two things
before it ratifies: a reference implementation and a conformance test
suite. As of today, neither exists upstream.

This week we're submitting both.

## Why this matters

3D Gaussian splatting has spent eighteen months in the same place every
new graphics format gets stuck: every viewer ships its own bespoke binary
container, no two writers agree on layout, and the production pipeline is
a chain of "this PLY only works in *that* renderer." If your job is
moving splats from training to a web viewer to a USD scene to a Cesium
tileset, you've already paid this tax.

`KHR_gaussian_splatting` is the way out. It bolts Gaussian splatting onto
glTF 2.0, the format the entire 3D-on-the-web ecosystem already knows how
to load. A splat asset becomes a `mesh.primitive` carrying six well-known
accessors — `POSITION`, `_ROTATION`, `_SCALE`, `_OPACITY`, `_COLOR_DC`,
optional `_COLOR_SH` — plus a top-level `extensions` block. Nothing else.

But "well-known" is doing a lot of work in that sentence. Until the RC
ratifies, there's no canonical answer to questions like *"is `_COLOR_SH`
one accessor element per coefficient or one per splat?"* (More on that
below.) And until a conformance suite exists, every implementor is
free to disagree by accident.

## What SplatForge built

Three artifacts, all in the public SplatForge repo
([github.com/splatforge/splatforge](https://github.com/splatforge/splatforge)):

1. **`splatforge-gltf`** — the writer. ~1,400 lines of Rust that emit
   either an external `.gltf` plus `.bin` sidecars, or a single self-
   contained `.glb`, with full support for `KHR_mesh_quantization`
   integer accessors when the asset budget is tight.
2. **`splatforge-khr-validate`** — the validator. A pure-Rust CLI that
   takes a glTF or GLB and emits a per-clause pass/fail report. 23 spec
   clauses cover everything from "is the extension declared in
   `extensionsUsed`" through "do all per-splat accessors share a `count`"
   down to bufferView bounds checks.
3. **A 10-file fixture corpus**, generated deterministically from the
   writer. Five valid fixtures (FLOAT, quantized, with SH, with the
   `KHR_gaussian_splatting_compression_spz` sub-extension declared, plus
   an external-buffer `.gltf` variant); five negative fixtures that each
   exercise a different class of invalid asset — missing extension
   declaration, missing required attribute, wrong accessor type, missing
   `POSITION` min/max, mismatched per-splat counts.

All three live under `crates/splatforge-khr-conformance`. To regenerate
the fixtures from a clean tree:

```bash
crates/splatforge-khr-conformance/scripts/generate-fixtures.sh
```

To validate any glTF or GLB against the suite:

```bash
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    path/to/asset.glb
```

The validator exits 0 if every clause passes, 1 on any failure, 2 on a
validator-level error. CI runs the entire matrix on every PR.

## The ambiguity

We did not get through this without finding something the spec text
should clarify. The single largest ambiguity is the wire layout of
`_COLOR_SH`. Strict glTF accessor semantics (§3.6.1 — *"count is the
number of elements of `type`"*) imply that `_COLOR_SH` should be a
`SCALAR FLOAT` accessor with `count = splat_count * 45`. A more relaxed
"per-splat record" reading would set `count = splat_count` and treat the
45 floats per splat as a packed stride. Both readings produce identical
byte buffers, but `splatforge-khr-validate` rejects one and accepts the
other.

Our submission asks Khronos to formalise Reading A — strict accessor
semantics — because anything else breaks glTF 2.0 readers that don't
understand the new extension. Fixture `04_valid_with_sh.glb` is
deliberately written under the loose reading so manual review of the
validator output makes the disagreement visible.

## How this fits the bigger picture

SplatForge's v2 plan (24-month, 30-engineer, $5-15M ARR target) makes
"land Khronos contributor status before RC ratifies" a non-negotiable Q1
deliverable. The conformance suite is what makes us a contributor. The
underlying `splatforge-gltf` crate is what makes us the *reference*
contributor — every Gaussian splatting writer that ever needs to round-
trip through glTF can now point at a working, tested, public-licence
implementation.

The plan after this: open a Khronos issue linking to the conformance
directory, request a slot on the next working-group call, and start
shipping per-clause fixes upstream as we find them in real assets.

## Try it yourself

```bash
git clone https://github.com/splatforge/splatforge
cd splatforge
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb
```

Expected output: `summary: 20 pass, 0 fail, 3 skip (of 23 clauses)`.
Skips are the SPZ and SH clauses that don't apply to the baseline.

If you run it against a glTF or GLB *you* produced and the validator
complains, please open an issue — we want every disagreement on the
table before the RC ratifies in Q2.
