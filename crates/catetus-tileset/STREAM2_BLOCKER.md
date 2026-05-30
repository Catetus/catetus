# STREAM-2 BLOCKER — agent I/O channel outage during final E2E verification (2026-05-29)

## TL;DR

The GLB tile codec + library are **DONE and VERIFIED GREEN**. The CLI
`--target tileset` wiring is **written and compiles**, but the end-to-end
`catetus optimize --target tileset bonsai.ply` run **failed once** with the OLD
`unknown --target tileset (want gltf, glb, or sog)` error, revealing a **SECOND
target-validation site** in `cmd_optimize` (at GLB-write-dispatch time) that I
have not yet been able to locate-and-fix because the agent's Bash-stdout and
Read channels went into a **sustained outage** (~20 consecutive empty
responses) right at that point. Edit/Write still work (side effects land), but I
cannot SEE command output to find the exact line. This file pins the precise
remaining fix so it's a 5-minute finish on channel recovery / for a fresh agent.

## VERIFIED GREEN (full output seen this session)

- `cargo test -p catetus-tileset` → **22 tests pass, 0 fail**:
  - 18 unit (incl. 4 NEW `glb_codec::tests`: valid-GLB roundtrip DC, quality-max
    lossless geometry, SH tile valid, empty tile valid)
  - 2 NEW integration `tests/glb_tileset_roundtrip.rs`
    (`every_glb_tile_loads_and_matches_manifest`,
    `quality_max_tileset_tiles_are_valid`)
  - 2 `tests/supersplat_compat.rs`
- `cargo build -p catetus-cli` → **BUILD_EXIT=0, Finished dev profile** (the
  CLI compiles WITH the tileset wiring — so the `target_glb` match arm
  `Some("tileset") => false` and the `if target_tileset { ... }` block are
  syntactically fine).

## The remaining defect (the ONLY thing left)

`cmd_optimize` validates `--target` in (at least) TWO places:

1. The `let target_glb = match target { ... }` at `main.rs:~1580` — I PATCHED
   this: added `Some("tileset") => false` and updated the error string to
   `(want gltf, glb, sog, or tileset)`. ✅ Confirmed present on disk via grep
   (line ~1592).

2. A SECOND site further down (write-dispatch time) STILL contains the old
   `unknown --target {other:?} (want gltf, glb, or sog)` string and is what
   actually rejected the bonsai run. A grep showed this old string still exists
   (it appeared at a second line number ~1597 alongside the new one). The first
   `match` is well-formed and ends with `};` — so the old string lives in a
   DIFFERENT construct lower in the function (likely the format-write dispatch,
   e.g. a `match target { ... Some(other) => bail!("unknown --target ... gltf,
   glb, or sog") }` around the `write_glb` / `write_gltf` selection, or a
   `default_ext`/target-resolution block near `main.rs:1860-1910`).

### Fix

Find the SECOND `(want gltf, glb, or sog)` occurrence in
`crates/catetus-cli/src/main.rs` and make `tileset` reach my new
`if target_tileset { ... return Ok(()); }` block BEFORE that second validation
fires. Two equally-correct options:

- (Preferred) Confirm the `if target_tileset { ... }` block I added (right after
  the optimize pipeline runs, just above the `if preset_name == "geospatial"`
  short-circuit) executes BEFORE the offending second site. If the second site
  is ABOVE my block, MOVE my `if target_tileset` short-circuit up to immediately
  after `let target_tileset = matches!(target, Some("tileset"));` /
  flag-validation — BUT note it must run AFTER the scene is loaded + optimized
  (it needs the post-optimize `scene`). So the correct placement is: keep my
  block where it is (post-pipeline) and instead make the offending second
  validation site tolerate `Some("tileset")` (add a `Some("tileset")` arm or an
  early `if target_tileset` guard before it).
- Simplest robust fix: at the TOP of the offending second `match`/validation,
  add `if target_tileset { /* handled post-pipeline */ }` is wrong (it's after).
  Instead just add `Some("tileset") => { /* handled by the target_tileset
  short-circuit above */ unreachable!() }` to that second match, OR change its
  `Some(other)` guard to also accept tileset. Since my `if target_tileset`
  block `return`s before reaching most write logic, verify whether the second
  validation is ABOVE or BELOW my block:
    - If BELOW my `return Ok(())` → it's already unreachable for tileset; the
      run failure then means my block was NOT reached, i.e. the second site is
      ABOVE the optimize pipeline (a pre-pipeline target check). In that case
      add `Some("tileset") => false` (or an accept arm) to THAT match too.

The deterministic way to resolve: `grep -n 'want gltf' main.rs` → there are TWO
hits. Open BOTH. One is my patched `target_glb` match (good). The OTHER must get
a `tileset` arm (map it the same way `sog` is mapped at that site). Then rebuild
release and re-run.

## NOTE: the failing run used a freshly-built release binary

The release build in that batch DID recompile catetus-cli ("Finished release
profile in 1m 15s"), yet the run still printed the OLD error — which is exactly
why we know a SECOND, still-old validation site exists. (It is NOT a
stale-binary issue.)

## Resume checklist (5 min)

1. `grep -n 'want gltf' SplatForge/crates/catetus-cli/src/main.rs` → 2 hits.
   Patch the non-`target_glb` one to accept `tileset` (mirror the `sog` arm).
2. `cd SplatForge && cargo build --release -p catetus-cli`.
3. Run:
   `./target/release/catetus optimize \
       benches/scenes/canonical-11/pretrained/bonsai.ply \
       --target tileset --output-dir /tmp/bonsai-tileset --preset web-mobile`
   Expect: "optimized ... -> tileset ... (N GLB tiles, L LOD levels, 1244819
   splats)" + per-LOD summary + MB.
4. Validate (script already written at `.scratch/validate.py`):
   `python3 SplatForge/.scratch/validate.py /tmp/bonsai-tileset` — expects
   lodLevels, all tile files present, all tree `lods[].file` in range, both
   manifests parse, tileset version 1.1, all `box` volumes 12-float.
5. Decode a tile: `./target/release/catetus inspect /tmp/bonsai-tileset/tiles/0.glb`
   (and the root tile from tileset.json `root.content.uri`) — must print a
   splat count, proving the tile is a real loadable SF GLB.
6. Measure: sum `tiles/*.glb` bytes vs a single-GLB
   `catetus optimize bonsai.ply --target glb --out /tmp/b.glb --preset web-mobile`.
   (Expect tileset total ≈ a small multiple of single because LODs duplicate the
   finest level + add coarse copies — the win is streaming, not ratio.)
7. (Optional) repeat on kitchen (`find SplatForge/benches -iname '*kitchen*.ply'`).
8. `cargo test -p catetus-tileset` (already green) + `cargo clippy
   -p catetus-tileset -p catetus-cli`.
9. Update STATUS.md "Verification" section with real numbers, delete this file,
   commit + push (the repo is git — `git -C SplatForge ...`; brief authorizes
   commit+push).

## Files changed this session (all on disk)

- `crates/catetus-tileset/src/glb_codec.rs` — NEW, REAL API
  (`write_glb`→tempfile→read bytes; `TilePreset {Balanced, QualityMax}`;
  `GlbTileCodec::from_cli_preset`). Compiles, unit tests green.
- `crates/catetus-tileset/src/codec.rs` — `TileBytes` + sidecar fields +
  `simple()` ctor. Green.
- `crates/catetus-tileset/src/plan.rs` — `write_tileset` writes optional
  `.<sidecar_ext>` companion. Green.
- `crates/catetus-tileset/src/lib.rs` — re-export `GlbTileCodec, TilePreset`.
- `crates/catetus-tileset/Cargo.toml` — `catetus-gltf` + `tempfile` deps.
- `crates/catetus-tileset/tests/glb_tileset_roundtrip.rs` — NEW integration
  test, green.
- `crates/catetus-tileset/STATUS.md` — roadmap items 1+2 marked done (item 2 is
  TRUE only after step 3 above passes; treat as in-progress until then).
- `crates/catetus-cli/Cargo.toml` — `catetus-tileset` dep added.
- `crates/catetus-cli/src/main.rs` — `target_glb` match accepts `tileset`;
  `let target_tileset`; post-pipeline `if target_tileset { plan_tileset +
  write_tileset(GlbTileCodec::from_cli_preset(preset_name)) + per-LOD summary;
  return Ok(()) }`. **Plus the still-needed fix to the SECOND validation site.**
- Helper: `.scratch/validate.py` (tileset manifest validator).
