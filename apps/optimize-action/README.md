# SplatForge Optimize — GitHub Action

A drop-in GitHub Action that submits the Gaussian-splat files in a pull request to
[SplatForge Cloud](https://splatforge-api.fly.dev), gates the PR on a compression
target, and posts a sticky badge comment with the fidelity report.

```yaml
- uses: splatforge/optimize-action@v1
  with:
    api-key: ${{ secrets.SPLATFORGE_API_KEY }}
```

That's it. Every `.ply` / `.splat` / `.spz` / `.ksplat` file changed in the PR is
uploaded to the optimizer, the resulting `.glb` is published behind a public URL,
and a sticky comment is rendered like:

> ![SplatForge fidelity](https://img.shields.io/badge/SplatForge%20fidelity-78%2F100-brightgreen) Optimized 1 splat for commit `a1b2c3d`.
>
> | File | Input | Output | Ratio | Status | Download |
> | --- | ---: | ---: | ---: | --- | --- |
> | `scenes/bonsai.ply` | 17.4 MB | 3.8 MB | 21.8% | ✓ done | [.glb](...) |
>
> **Aggregate ratio:** 21.8% — threshold 60% → **PASS**

---

## Why a Node 20 action (and not Docker)

The action is implemented as a `runs.using: node20` JavaScript action with **zero
runtime dependencies** — every `require()` resolves into Node 20's standard
library (`https`, `fs`, `child_process`, `crypto`). The implications:

- No `npm install` step on the runner — the action starts in <2 seconds.
- No `Dockerfile` build cache to invalidate when GitHub bumps runner images.
- Works identically on macOS and Linux self-hosted runners.
- We can audit the entire bearer-key flow in one file (`src/index.js`).

A Docker container action would also have worked, but the typical cold-start
penalty (image pull + container start: ~30 s on a fresh runner) is most of our
budget for "optimize a 17 MB splat" PRs.

## Inputs

| Name | Required | Default | Description |
| --- | --- | --- | --- |
| `api-key` | yes | — | Bearer token. Set as a repo secret `SPLATFORGE_API_KEY`. Get one at https://splatforge.dev. |
| `api-url` | no | `https://splatforge-api.fly.dev` | Override for self-hosted SplatForge Cloud deployments. |
| `preset` | no | `web-mobile` | One of `lossless-repack` / `web-mobile` / `size-min`. |
| `target` | no | *(changed files)* | Path or comma-separated globs. Overrides PR-diff detection. |
| `regression-threshold` | no | `1.0` | Maximum allowed output/input ratio. `0.6` = output must be ≤60% of input. `1.0` disables the gate. |
| `comment` | no | `true` | Post (and update on re-runs) a sticky PR comment. |
| `timeout-seconds` | no | `270` | Per-job polling timeout. |

## Outputs

| Name | Description |
| --- | --- |
| `fidelity-score` | Aggregate fidelity score across optimized scenes (0-100, higher = better). |
| `compression-ratio` | Mean output/input byte ratio across optimized scenes (0-1, lower = better). |
| `output-url` | JSON array of output `.glb` URLs. |
| `report-url` | URL to the first job's JSON record (full per-scene report). |

> Fidelity scoring today is `100 · (1 − ratio)` (i.e. a byte-savings proxy). PSNR
> / SSIM-based scoring lands when the Cloud `/v1/jobs/:id/report` endpoint
> ships — the action will automatically read it without a workflow change.

## Security

- `api-key` is registered with `::add-mask::` on entry — any subsequent
  occurrence in *any* line of workflow output is replaced with `***` before
  GitHub displays it.
- The key is never written to `GITHUB_OUTPUT`, never echoed via `set-output`,
  and error messages are scrubbed for the literal key value before logging.
- Calls to the SplatForge API use `Authorization: Bearer <key>`; the GitHub
  API uses the workflow's `GITHUB_TOKEN`, scoped to PR-comment writes.

## Free tier vs paid tier

Both tiers go through the same API endpoint — the gating is server-side.

- **Free tier (OSS repos / hobby use):** 100 optimizer jobs per month per key.
  The deterministic CPU pipeline. Outputs are publicly accessible by URL (the
  PR comment links the `.glb` directly).
- **Paid tier:** higher rate limit, the differentiable A100 repack at the
  same output-URL surface, and private result URLs scoped to a Stripe-attached
  team. The action does not need to know which tier is active — it just
  forwards the bearer token.

## Idempotency

Every job is created with a label of the form `gh:<short-sha>:<file>`. This
key is surfaced in the Job JSON and is intended to let a future API endpoint
(`GET /v1/jobs?label=...`) detect retries on the same SHA and short-circuit
to the prior result. Today the action will create a new job on each rerun;
billing impact for OSS PRs is zero because optimizer runs are free up to the
100/month tier limit.

## Example workflow

See [`example/.github/workflows/optimize.yml`](./example/.github/workflows/optimize.yml).
The minimal recipe:

```yaml
name: optimize
on:
  pull_request:
    paths: ['**/*.ply', '**/*.splat', '**/*.spz']
permissions:
  pull-requests: write   # for the sticky comment
  contents: read
jobs:
  splatforge:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # needed for `git diff base...head` to work
      - uses: splatforge/optimize-action@v1
        with:
          api-key: ${{ secrets.SPLATFORGE_API_KEY }}
          preset: web-mobile
          regression-threshold: '0.6'
```

## Local development

```bash
# Smoke against the live Fly API.
SPLATFORGE_API_KEY=sk_... ./scripts/test-locally.sh
```

The smoke test simulates the same environment Actions provides (sets the
`INPUT_*` / `GITHUB_*` env vars, points at a tiny bonsai sample, runs
`src/index.js` directly under Node 20). It exits non-zero on any failure.

## Rate limits

The free tier is **100 jobs per month per API key**. A typical OSS repo
opens 5-20 splat PRs/month, so a single key is enough for most projects.
Hitting the cap returns `429 Too Many Requests`, which the action
surfaces in the PR comment.
