#!/usr/bin/env python3
"""
SplatBench v3 matrix dispatcher.

Given (preset, scene_id, commit), call the Modal endpoint for the preset
with the scene PLY URL, wait for the terminal callback (or poll the result),
and emit a single result JSON to benches/timeseries/<commit>/<preset>__<scene>.json.

This script is invoked by .github/workflows/splatbench-v3.yml as one matrix
entry. It is intentionally self-contained: no project deps beyond `modal`,
`requests`, and stdlib so it can run on any Python 3.11+ environment with
the GitHub Actions runner.

Environment:
  MODAL_TOKEN_ID, MODAL_TOKEN_SECRET — Modal API auth (repo secrets)
  SPLATBENCH_PRESET — preset value (one of the 7 registered)
  SPLATBENCH_SCENE  — scene id from benches/scenes/manifest.json
  GITHUB_SHA        — commit being benchmarked
  GITHUB_REF_NAME   — branch/ref (for trace metadata)
  SPLATBENCH_OUT_DIR — override output dir (default: benches/timeseries/<sha>)
  SPLATBENCH_DRY_RUN — if "1", emit a placeholder result without calling Modal
                        (used for smoke-testing the workflow on first deploy
                        before MODAL_TOKEN_* are configured).

Exit code is always 0 if the result JSON was written (even if the cell
failed inside Modal — the aggregator surfaces failure modes honestly
instead of breaking the workflow). Non-zero exit only on argument errors
or filesystem failures.
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "benches" / "scenes" / "manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def find_scene(manifest: dict, scene_id: str) -> dict:
    for s in manifest["scenes"]:
        if s["id"] == scene_id:
            return s
    raise SystemExit(f"unknown scene id: {scene_id!r} — see benches/scenes/manifest.json")


def stable_run_id(commit: str, preset: str, scene_id: str) -> str:
    """Idempotent per-cell run id so re-running on the same commit reuses
    deterministic identifiers (handy for Modal trace correlation)."""
    h = hashlib.sha256(f"{commit}|{preset}|{scene_id}".encode()).hexdigest()[:16]
    return f"sbv3-{h}"


def dispatch_modal(preset: str, scene: dict, run_id: str) -> dict:
    """Call the deployed Modal endpoint for `preset` and return the metrics
    dict the endpoint POSTs back. Falls through to a structured failure
    record if Modal auth is missing or the endpoint errors."""
    try:
        import modal  # noqa: F401  (deferred import — Modal SDK is optional locally)
    except ImportError:
        return {
            "ok": False,
            "error": "modal-sdk-not-installed",
            "note": "pip install modal in the workflow env",
        }

    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        return {
            "ok": False,
            "error": "modal-credentials-missing",
            "note": "MODAL_TOKEN_ID + MODAL_TOKEN_SECRET repo secrets required",
        }

    # Map preset -> (modal app name, function name). The deployed apps own
    # their own preset-specific encoders; this dispatcher only needs to know
    # how to invoke them and parse the result envelope.
    PRESET_TO_MODAL = {
        "web-mobile":               ("splatforge-worker",       "run_optimize"),
        "size-min":                 ("splatforge-worker",       "run_optimize"),
        "hacpp-lzma":               ("splatforge-hacpp-lzma",   "encode"),
        "fcgs-instant":             ("splatforge-fcgs",         "encode"),
        "splatforge-qat-scaffold":  ("splatforge-qat-scaffold", "encode"),
        "splatforge-qat-bundle":    ("splatforge-qat-bundle",   "encode"),
        "splatforge-qat-3dgs":      ("splatforge-qat-3dgs",     "encode"),
    }
    if preset not in PRESET_TO_MODAL:
        return {"ok": False, "error": f"unknown-preset:{preset}"}

    app_name, fn_name = PRESET_TO_MODAL[preset]

    try:
        import modal
        fn = modal.Function.lookup(app_name, fn_name)
        # Common envelope: every endpoint accepts (preset, blob_url,
        # filename, run_id) and returns {output_url, ply_save_pct,
        # delta_psnr_db, delta_ssim, delta_lpips, wall_secs, modal_cost_cents}.
        # New endpoints should mirror this shape; legacy endpoints that
        # take a different signature go through a worker-shim function in
        # apps/worker/worker.py that adapts.
        result = fn.remote(
            preset=preset,
            blob_url=scene["sourceUrl"],
            filename=f"{scene['id']}.ply",
            run_id=run_id,
        )
        if not isinstance(result, dict):
            return {"ok": False, "error": "non-dict-result", "raw": repr(result)[:512]}
        result.setdefault("ok", True)
        return result
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": "modal-exception", "exception": repr(e)[:512]}


def main() -> int:
    preset = os.environ.get("SPLATBENCH_PRESET")
    scene_id = os.environ.get("SPLATBENCH_SCENE")
    commit = os.environ.get("GITHUB_SHA", "local-dev")
    ref = os.environ.get("GITHUB_REF_NAME", "local-dev")
    dry_run = os.environ.get("SPLATBENCH_DRY_RUN", "") == "1"

    if not preset or not scene_id:
        print("ERROR: SPLATBENCH_PRESET and SPLATBENCH_SCENE must be set", file=sys.stderr)
        return 2

    manifest = load_manifest()
    if preset not in manifest["presets"]:
        print(f"ERROR: preset {preset!r} not in manifest.presets", file=sys.stderr)
        return 2
    scene = find_scene(manifest, scene_id)

    run_id = stable_run_id(commit, preset, scene_id)
    t_start = time.time()

    if dry_run:
        result = {
            "ok": True,
            "dry_run": True,
            "note": "MODAL_TOKEN_* not configured — emitting placeholder cell",
            "ply_save_pct": None,
            "delta_psnr_db": None,
            "delta_ssim": None,
            "delta_lpips": None,
            "wall_secs": 0,
            "modal_cost_cents": 0,
            "output_url": None,
        }
    else:
        result = dispatch_modal(preset, scene, run_id)

    cell = {
        "schema": "splatforge.splatbench-v3.cell/0.1",
        "commit": commit,
        "ref": ref,
        "preset": preset,
        "scene_id": scene_id,
        "scene_class": scene.get("class"),
        "scene_dataset": scene.get("dataset"),
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_dispatcher_secs": round(time.time() - t_start, 2),
        "result": result,
    }

    out_dir = Path(
        os.environ.get("SPLATBENCH_OUT_DIR")
        or (REPO_ROOT / "benches" / "timeseries" / commit)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{preset}__{scene_id}.json"
    out_path.write_text(json.dumps(cell, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(REPO_ROOT)} (ok={result.get('ok')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
