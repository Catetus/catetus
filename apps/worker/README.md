# `splatforge-worker` — Modal Function for hosted optimize

A single Modal app exposing `/enqueue` + `/healthz` web endpoints plus a
container-backed `run_optimize` function that runs the pinned `splatforge`
CLI against a splat blob and POSTs the result to the API's status webhook.

## Quick start (local dev)

```bash
# Auth (one-time, opens browser):
python3 -m modal setup

# Hot-reload dev — Modal mounts your local file into a remote container and
# rebuilds the image only when image-defining commands change.
python3 -m modal serve apps/worker/worker.py
```

`modal serve` prints two URLs:

```
https://<workspace>--splatforge-worker-enqueue-dev.modal.run
https://<workspace>--splatforge-worker-healthz-dev.modal.run
```

The `/enqueue` URL is what `splatforge-api` should set as
`SPLATFORGE_MODAL_URL` (just the host; the path is appended internally).

## Production deploy

```bash
python3 -m modal deploy apps/worker/worker.py
```

The Modal image bakes the `splatforge` CLI at the git tag in
`SPLATFORGE_REF` (default `v0.1.1`). To pin a different revision:

```bash
SPLATFORGE_REF=v0.1.2 python3 -m modal deploy apps/worker/worker.py
```

After deploy, the published URLs are:

```
https://<workspace>--splatforge-worker-enqueue.modal.run
https://<workspace>--splatforge-worker-healthz.modal.run
```

## Cost expectations

Per-job (1M-splat scene, `web-mobile` preset):

- Image: 2 CPU × 4 GB × ~30 s = ~$0.01.
- Volume storage: ~0.1 GB-month/job → < $0.01.

Roughly 100 free invocations per dollar.

The image build is the only expensive step:

- Cold build (Rust toolchain + splatforge from source): ~3 min × 2 CPU.
- Cached rebuild: < 10 s.

## Function surface

| Function       | Cost class | Purpose                                           |
| -------------- | ---------- | ------------------------------------------------- |
| `enqueue`      | 0.25 CPU   | FastAPI web endpoint; spawns `run_optimize`       |
| `healthz`      | 0.25 CPU   | Liveness probe                                    |
| `run_optimize` | 2 CPU      | Pulls blob, runs CLI, POSTs result to API webhook |

## Webhook contract

After a successful (or failed) `run_optimize`, the worker POSTs to:

```
<SPLATFORGE_API_URL>/v1/jobs/<job_id>/status
```

with a JSON body matching `JobResult` in `apps/api/src/store.rs`:

```json
{
  "status": "succeeded" | "failed",
  "ratio": 22.81,
  "bytes_in": 273000000,
  "bytes_out": 12000000,
  "report": { /* analyze JSON */ },
  "spz_path": "/data/.../scene.spz",
  "gltf_path": "/data/.../scene.gltf"
}
```

The API is responsible for picking up the artifact bytes from the worker
(re-upload to Blob with a public ACL, or stream proxy). See
[`apps/api/src/store.rs`](../api/src/store.rs).
