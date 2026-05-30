# @catetus/mcp

Model Context Protocol server for **Catetus** — SOTA 3D Gaussian Splatting compression.

Exposes the SplatBench corpus, 14 tools (analyze, encode, score_fidelity, repack, predict_quality, recommend_preset, batch_jobs, etc.), 4 starter prompts, and resources backed by [api.catetus.com](https://api.catetus.com).

This package implements the **server core** per `ARCHITECTURE.md` (file ownership: stdio + HTTP entrypoints, all 14 tools, 4 prompts, auth/tier resolution, error envelope, pagination, api-client). Resources (`src/resources/`), MCPB packaging (`mcpb-manifest.json`, `dist-bundle/`), and per-client example configs (`examples/`) are owned by parallel agents.

## Install

```bash
# Public tier (no API key — free tools only)
npx @catetus/mcp@latest

# Paid tier (set env var)
CATETUS_API_KEY=cat_live_REPLACE_ME npx @catetus/mcp@latest
```

## Local dev

```bash
cd packages/mcp
npm install
npm run build
node dist/server-stdio.js          # stdio transport
PORT=3000 node dist/server-http.js  # HTTP transport (Streamable)
```

## Tier matrix

| Tool | Tier | Description |
|---|---|---|
| `analyze` | free | Splat-file analysis (count, bbox, SH, hash, recommendations) |
| `list_presets` | free | Catalog of free + paid encoder presets |
| `list_scenes` | free | SplatBench corpus listing |
| `get_scene` | free | Per-scene record with leaderboard row |
| `optimize` | free (stdio-only) | Run free preset over a local PLY |
| `compare` | free (stdio-only) | Render before/after diff |
| `list_competitor_codecs` | free | Public competitor codec catalog |
| `validate_pipeline` | free | Sanity-check Catetus output |
| `encode` | paid | Hosted SOG/V5.2/T2.1.R encode |
| `score_fidelity` | paid | ML perceptual quality score |
| `repack` | paid | Differentiable repack on A100 |
| `predict_quality` | paid | Predict fidelity without encoding |
| `recommend_preset` | paid | Pick best preset for constraints |
| `batch_jobs` | paid | Batched encode/repack |
| `list_jobs` | paid | Poll job results |

Paid tools are filtered out of `tools/list` when no API key is present. The LLM never sees them, never tries to call them.

## Smoke test with MCP Inspector

```bash
cd packages/mcp
npm run build
npx @modelcontextprotocol/inspector --cli node dist/server-stdio.js --method tools/list
```

Expected: 8 public tools. With `CATETUS_API_KEY=…` set, expect 15 (8 public + 7 paid including `list_jobs`).

## Architecture

See `splatforge-private/docs/mcp/ARCHITECTURE.md` for the full design doc.

## License

MIT.
