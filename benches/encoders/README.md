# Third-party encoder bench runners

Drop a directory in here named after the encoder. Each directory must contain:

- `run.sh` — given `INPUT_PLY=$1 OUTPUT_DIR=$2`, runs the encoder and produces
  exactly **one** output file in `$OUTPUT_DIR` named `output.<ext>` plus a
  `meta.json` with:
  ```json
  { "encoder": "<name>", "version": "<version>", "output_bytes": <int>,
    "wall_seconds": <float>, "command": "<the command we ran>" }
  ```
- `README.md` — install instructions and any caveats.

The bench harness (`benches/run-encoders.mjs`) iterates every scene in
`benches/scenes/*.ply`, runs `run.sh` for each registered encoder, fetches
the resulting file, and writes per-encoder columns into the SplatBench
leaderboard JSON.

Encoders we've registered:

- `splat-transform/` — PlayCanvas `@playcanvas/splat-transform` (npm).
