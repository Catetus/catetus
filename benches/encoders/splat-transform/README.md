# splat-transform bench runner

Runs PlayCanvas [`@playcanvas/splat-transform`](https://github.com/playcanvas/splat-transform)
against the SplatBench corpus. The output format is **SOG** (PlayCanvas's
Self-Organizing Gaussians) — the format splat-transform users actually ship
to PlayCanvas Engine runtimes. We do not force splat-transform to emit SPZ
or glTF; comparing the real production formats on each side is more honest.

## Install

```
nvm use 20  # any Node ≥ 18
# splat-transform is fetched on first run via npx
```

## Run

The bench harness invokes `run.sh INPUT_PLY OUTPUT_DIR` automatically. To
exercise it manually:

```
./run.sh /path/to/bonsai.ply /tmp/sw-bonsai
cat /tmp/sw-bonsai/meta.json
```

## Output

`meta.json` carries the canonical bench metadata. Compression ratio is
computed downstream by `benches/run-encoders.mjs` from
`input_bytes / output_bytes`. Fidelity is scored downstream by rendering
the SOG via PlayCanvas's reference web viewer through the same 8-orbit
camera path Catetus uses on its own outputs — see
`benches/run-encoders.mjs` for how the cross-format render flow works.

## Pinning

`SPLAT_TRANSFORM_VERSION` env var pins the npm version. Default is
`latest`. Recommended: pin to whatever shipped at the start of a bench
campaign so re-runs are reproducible. The actual resolved version is
recorded in `meta.json` regardless.

## Caveats

- SOG decode requires the PlayCanvas viewer or `splat-transform`'s own
  decoder. Cross-format fidelity scoring is harder than apples-to-apples
  SPZ→SPZ — track the methodology gap in the leaderboard footnote.
- splat-transform doesn't surface a fidelity report; the column will show
  compression-only until we pipe the SOG through Catetus's WebGL2
  viewer (which can read SOG via the `catetus-spz` codepath).
