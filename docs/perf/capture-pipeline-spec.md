# Capture pipeline spec — photos → COLMAP → 3DGS → compressed

**Status**: initialization (2026-05-15). Pricing curve, dispatch wiring,
and Modal app scaffold landed in `feat/capture-pipeline`; the three
inner stages (COLMAP, gsplat training, encode dispatch) are TODO stubs
that fail loudly until implemented.

## Why this exists

Catetus today is a hosted compression API: customers upload an
existing 3D Gaussian Splat PLY, get back a `.mgs2` / `.lodge` /
codec-specific bitstream. The product gap vs Polycam / Luma is they
ship multi-photo *capture* as well as compression. This pipeline closes
that loop:

```
N photos.zip
  → POST /v1/jobs (preset=capture-and-compress)
  → API persists Job + Stripe-meter `capture_runs`
  → public worker forwards to private `catetus-capture` Modal app
    ├── Stage 1: COLMAP sparse reconstruction              (~5 min, CPU)
    ├── Stage 2: 3DGS training (gsplat, MVP 7k iters)      (~15-25 min, A100)
    └── Stage 3: encode via inner preset (default codec-gs-mixed) (~2 min, CPU)
  → upload compressed artifact to Vercel Blob
  → POST {status, output_url, metrics} back to API callback_url
```

The output bitstream is identical in shape to what a customer would
get if they had submitted a hand-curated PLY to `codec-gs-mixed`
directly — same `.mgs2`, same viewer, same SDK. The capture pipeline
is a thin front-end on the existing encoder graph.

## Architecture

### Components

| Component | Where | Status |
|---|---|---|
| `capture-and-compress` preset registration | `apps/api/src/pricing.rs` | LANDED |
| Public-worker dispatch + tests | `apps/worker/worker.py`, `apps/worker/test_worker_dispatch.py` | LANDED |
| Private Modal app `catetus-capture` | `catetus-private/apps/capture/capture_modal.py` | SCAFFOLD |
| COLMAP stage | `_run_colmap()` in capture_modal.py | TODO |
| gsplat training stage | `_run_gsplat_training()` | TODO |
| Encode dispatch | `_encode_via_inner_preset()` | TODO |
| Blob upload helper | `_upload_artifact()` | TODO |

### API surface

The pipeline reuses the existing `/v1/jobs` URL-mode entrypoint with
two changes the API layer surfaces in a follow-up:

```http
POST /v1/jobs
{
  "preset": "capture-and-compress",
  "input_kind": "photos",          // NEW; existing kinds: "splat", "scan"
  "source_url": "https://blob.../photos.zip",
  "inner_preset": "codec-gs-mixed", // optional; default codec-gs-mixed
  "training_iters": 7000            // optional; default 7000 (MVP fast path)
}
```

The `input_kind` field is purely informational on the API side — the
preset name already signals "this is a photos job" to the dispatcher.
We add it so the job-row carries the originating asset shape for
analytics + future preview rendering.

### Private Modal app contract

The public worker forwards to `CATETUS_CAPTURE_URL` (Modal secret;
configured at deploy time). The downstream `/enqueue` accepts:

```json
{
  "job_id": "...",
  "preset": "capture-and-compress",
  "blob_url": "https://blob.../photos.zip",
  "filename": "photos.zip",
  "callback_url": "https://api.catetus.com/v1/jobs/<id>/status",
  "inner_preset": "codec-gs-mixed",   // optional, default codec-gs-mixed
  "training_iters": 7000              // optional, default 7000
}
```

and replies:

```json
{ "queued": true, "error": null }
```

When the pipeline finishes (succeed or fail), the Modal app POSTs the
terminal callback directly to the API — the public worker is never on
the data path.

```json
{
  "status": "succeeded",
  "output_url": "https://blob.../scene.mgs2",
  "metrics": {
    "n_photos": 47,
    "photos_zip_bytes": 248123456,
    "colmap_seconds": 412.3,
    "training_seconds": 1180.0,
    "training_iters": 7000,
    "trained_ply_bytes": 412345678,
    "encode_seconds": 121.7,
    "artifact_bytes": 8765432,
    "compute_seconds": 1734.6
  }
}
```

## Pricing

`apps/api/src/pricing.rs::preset_compute_curve("capture-and-compress")`
returns `(2400.0, 2.4)` — i.e. 2400 s base + 2.4 s/MB on the photos.zip
payload. Anchored to a synthetic typical scene (50 photos, ~5 MB each
= ~250 MB zip → ~3000 s ≈ $3.00 compute + flat fee).

Bands the test in `apps/api/tests/pricing.rs::capture_and_compress_preset_registered_with_heavy_curve`
locks down:

| Photo zip size | Expected wall-clock | Compute $ (PER_COMPUTE_SECOND_CENTS=0.1) |
|---|---|---|
| 100 MB (~20 photos) | ~2640 s | $2.64 |
| 250 MB (~50 photos) | ~3000 s | $3.00 |
| 500 MB (~100 photos) | ~3600 s | $3.60 |

These are pass-through quotes against the meter; retail price ($5-10
per scene per spec) layers operator margin on top. Re-anchor once real
captures land on the deployed Modal app.

## Compute composition

Breaks down of a typical 50-photo / ~250 MB scene on an A100 / 16-CPU
Modal container:

| Stage | Wall-clock (typical) | Bottleneck |
|---|---|---|
| Download photos.zip | ~10 s | network |
| Extract photos | ~5 s | disk |
| COLMAP sparse | 300-600 s | CPU SIFT + matching |
| 3DGS training (7k iters MVP) | 900-1500 s | A100 |
| Encode (codec-gs-mixed inner) | 95-150 s | CPU HEVC |
| Upload artifact | ~5 s | network |
| **Total** | **~1500-2300 s** | training-bound |

FastGS / DashGaussian variants of the training step (~200-300 s for
the same quality) slot in once their wheels are validated on the
Modal image, dropping the total toward ~1000 s. Pricing curve doesn't
change — the quote is the ceiling and customers get billed against
actual meter seconds.

## Failure modes

Each documented case must surface a structured error to the API so the
frontend shows actionable text instead of "internal error".

| Failure | Trigger | API error |
|---|---|---|
| Too few photos | `n_photos < 8` after zip extract | 400 "need at least 8 photos" |
| COLMAP registers <50% of photos | sparse `images.bin` underflow | 422 "too few features matched; try slower / more overlapping photos" |
| Sparse model <100 points | textureless scene | 422 "scene lacks texture; needs detail/contrast surfaces" |
| Corrupt JPEG | COLMAP segv on bad file | 400 "corrupt image: `<filename>`" |
| Training diverges | PSNR < 12 dB at iter 1000 (~5% of real captures) | 500 "training failed to converge; resubmit with `training_iters=30000` for quality tier" |
| Training OOM | splat count > 2M | 500 "scene too large; split into smaller captures" |
| Output too small | `artifact_bytes < 1024` | 500 "encode produced empty output; report job_id" |

## MVP rollout

**Milestone 1 (this commit, LANDED)**: scaffold, pricing curve, dispatch wiring.
  * Public worker recognises `capture-and-compress` preset.
  * Healthz reports `preset_dispatch_configured["capture-and-compress"]`.
  * Pricing-preview returns a quote in the documented band.
  * Modal app scaffold ships with COLMAP / gsplat in the image, stages stubbed.

**Milestone 2**: implement `_run_colmap()` → `colmap automatic_reconstructor`.
  * Smoke test on a 20-photo Mip-NeRF360 sub-capture.
  * Verify sparse/0/ exists, image registration > 80%.

**Milestone 3**: implement `_run_gsplat_training()` with 7k iters.
  * Smoke test against the COLMAP output from milestone 2.
  * Verify final PLY > 1 MB, < 500 MB.

**Milestone 4**: implement `_encode_via_inner_preset()` →
  * dispatch to existing `catetus-codec-gs-mixed` Modal app
    (same forwarding shape the public worker uses).
  * Verify the .mgs2 round-trips through the existing decoder.

**Milestone 5**: implement `_upload_artifact()` and end-to-end on a
  curated dataset; flip `healthz.scaffold_only=False`; set
  `CATETUS_CAPTURE_URL` on the public worker.

**Milestone 6 (post-launch)**: replace MVP training with FastGS /
  DashGaussian for the 30-min target; add `quality` tier with 30k iters
  for hero-asset use.

## Out of scope (this initialization)

* COLMAP-on-Modal Docker image with CUDA SIFT (the apt build is the
  MVP; CUDA SIFT is milestone 2.5 when a customer hits the throughput
  ceiling).
* Mobile-side photo capture UI (separate iOS workstream).
* Re-running the pipeline end-to-end. The Modal app is scaffold-only —
  deploying it now would error on the first real invocation.
* Re-anchoring the pricing curve against real captures. Synthetic
  numbers are in place; tune in milestone 5 once the pipeline runs
  end-to-end on a curated dataset.
