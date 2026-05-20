# Catetus Optimize — example repo

This folder is a self-contained example you can copy into any GitHub repo.

```
example/
├── .github/workflows/optimize.yml  ← drop this in `.github/workflows/`
├── scenes/scene.ply                 ← any splat you want optimized on PR
└── README.md
```

## Setup

1. Copy `.github/workflows/optimize.yml` to your repo's `.github/workflows/` folder.
2. Get an API key from https://catetus.com (free tier is 100 jobs/month).
3. In your repo: **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `CATETUS_API_KEY`
   - Value: `sk_...` (the key from step 2)
4. Open a PR that touches a `.ply` / `.splat` / `.spz` / `.ksplat` file.
5. The action runs, optimizes the splat, and posts a fidelity badge on the PR.

## Trying the example

```bash
git clone <this-repo>
# Open a PR that modifies scenes/scene.ply (e.g. swap in a different .ply).
```

The included `scenes/scene.ply` is a 16-byte placeholder so the action's
discovery layer triggers without committing a multi-MB binary to git history.
Replace it with a real Gaussian-splat `.ply` to actually exercise the
optimizer. Public test scenes: https://huggingface.co/datasets/dylanebert/3dgs

## How the gate works

The workflow as-shipped uses `regression-threshold: 0.6`. That means the
average output size across all changed splats must be ≤60% of the average
input size. Tighter? Set `0.4`. Disable? Set `1.0`.
