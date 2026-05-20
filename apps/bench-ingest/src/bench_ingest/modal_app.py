"""Modal entrypoint for SplatBench v2 ingest.

Mirrors the local CLI's ``measure`` subcommand so the same scene can be
reproduced either on a developer laptop or on Modal infrastructure without
schema drift. The Modal path is the convenient one for large batches —
network egress from HuggingFace + Inria mirrors is fast on Modal, and the
optimize pipeline runs CPU-only so we can pick the cheapest tier.

Note: This module is imported lazily by the CLI. ``modal`` is NOT a
required dependency of the local harness.

Usage::

    modal run apps/bench-ingest/src/bench_ingest/modal_app.py::measure_remote \\
        --scene-id counter_mipnerf360_iter7k \\
        --ply-url "https://huggingface.co/.../point_cloud.ply" \\
        --license "MipNeRF360, CC-BY-4.0" \\
        --class indoor-scene \\
        --origin "MipNeRF360 / dylanebert HF mirror"

After the run, pull the result locally with::

    modal volume get splatbench-ingest splatbench-v2.json benches/reports/
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import modal  # type: ignore[import-not-found]

CATETUS_REF = os.environ.get("CATETUS_REF", "main")
CATETUS_REPO = os.environ.get(
    "CATETUS_REPO", "https://github.com/Catetus/catetus.git"
)

# Reuse the same base recipe as the optimize worker so we can hit the
# warm-image cache. If you change this image you'll pay one cold rebuild.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "git", "pkg-config", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable",
        f"git clone --depth 1 --branch {CATETUS_REF} {CATETUS_REPO} /opt/catetus",
        "/root/.cargo/bin/cargo build --release "
        "--manifest-path /opt/catetus/Cargo.toml -p catetus-cli",
        "ln -s /opt/catetus/target/release/catetus /usr/local/bin/catetus",
    )
    .pip_install("requests==2.32.3")
    # The harness lives in this same repo at apps/bench-ingest/src; copy it
    # into the image so the Modal Function can `import bench_ingest`.
    .add_local_python_source("bench_ingest")
)

app = modal.App("splatbench-v2-ingest", image=image)

volume = modal.Volume.from_name("splatbench-ingest", create_if_missing=True)

OUT_PATH = "/data/splatbench-v2.json"


def _download(url: str, dst: Path) -> None:
    """Stream-download a PLY to disk with a 30s read timeout."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "splatbench-v2-ingest"})
    with urllib.request.urlopen(req, timeout=120) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


@app.function(
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/data": volume},
)
def measure_remote(
    scene_id: str,
    ply_url: str,
    license: str,
    cls: str,
    origin: str,
    presets: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Download a PLY, run analyze + all presets, append to the shared JSON."""
    from bench_ingest.measure import (  # local import — only in the image
        DEFAULT_PRESETS,
        measure_scene,
        merge_row,
        row_is_complete,
    )

    chosen = tuple(presets) if presets else DEFAULT_PRESETS
    out_path = Path(OUT_PATH)

    # Load existing rows from the volume so multiple modal invocations build
    # up the same JSON without clobbering each other.
    if out_path.exists():
        doc = json.loads(out_path.read_text())
    else:
        doc = {
            "schema": "catetus.splatbench.v2/0.1",
            "name": "SplatBench v2 — real-photo corpus",
            "presets": list(chosen),
            "rows": [],
        }

    existing = next((r for r in doc.get("rows", []) if r.get("id") == scene_id), None)
    if existing and not force and row_is_complete(existing, chosen):
        return {"status": "skip", "scene_id": scene_id, "reason": "already-complete"}

    work = Path("/data/work") / scene_id
    work.mkdir(parents=True, exist_ok=True)
    ply = work / "input.ply"
    if not ply.exists() or force:
        _download(ply_url, ply)

    row = measure_scene(
        scene_id=scene_id,
        ply=ply,
        source="real",
        cls=cls,
        origin=origin,
        license=license,
        presets=chosen,
    )

    doc["rows"] = merge_row(doc.get("rows", []), row)
    doc["presets"] = list(chosen)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    volume.commit()
    return {
        "status": "ok",
        "scene_id": scene_id,
        "presetRuns": row.presetRuns,
        "bytesIn": row.bytesIn,
    }


@app.local_entrypoint()
def main(
    scene_id: str,
    ply_url: str,
    license: str = "unknown",
    cls: str = "real-scene",
    origin: str = "manual",
    force: bool = False,
) -> None:
    """``modal run ... measure_remote`` thin wrapper."""
    result = measure_remote.remote(
        scene_id=scene_id,
        ply_url=ply_url,
        license=license,
        cls=cls,
        origin=origin,
        force=force,
    )
    print(json.dumps(result, indent=2, default=str))
