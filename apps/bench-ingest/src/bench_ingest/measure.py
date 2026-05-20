"""Core measurement loop for SplatBench v2.

Given a local PLY file and a list of presets, runs ``catetus optimize``
once per preset, captures wall-clock + output bytes, and emits a row in the
v0 leaderboard schema (extended with a ``presetRuns`` map so we can support
more than the two original web-mobile/size-min columns without breaking
existing consumers).

The runner is intentionally CPU-only: ``catetus optimize`` performs
deterministic quantization passes and runs comfortably on a laptop. The
Modal entrypoint exists for batch convenience and reproducibility — it does
not change measured numbers, only where they are computed.

This module has zero non-stdlib imports so the local CLI works without a
virtualenv.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

# Presets we currently ship in the catetus CLI. Validated against
# `crates/catetus-optimize/src/presets.rs` at the v2 cutoff. Two of these
# (`web-mobile`, `size-min`) are the v0 leaderboard columns; the other two
# (`web-desktop`, `quest-browser`) are the natural targets for the
# real-photo expansion since they exercise different fan-out / quantization
# settings.
#
# The task spec referenced ``codec-gs-mixed`` and ``fcgs-instant``; those
# are not shipped presets in this repo's CLI, so the harness does not list
# them. Adding a preset is a one-line change in ``DEFAULT_PRESETS`` once the
# CLI registers it.
DEFAULT_PRESETS: tuple[str, ...] = (
    "web-mobile",
    "size-min",
    "web-desktop",
    "quest-browser",
)


@dataclass
class PresetRun:
    preset: str
    outputBytes: int
    ratio: float
    wallMs: int
    ok: bool = True
    stderr: str | None = None


@dataclass
class SceneRow:
    """Schema-compatible with `benches/reports/splatbench-v0.json` rows.

    The v0 schema has ``webMobileSpzBytes`` / ``webMobileRatio`` /
    ``sizeMinSpzBytes`` / ``sizeMinRatio`` as top-level fields. We mirror
    those into the row from the matching ``PresetRun`` so the Astro
    leaderboard continues to render without code changes. Any additional
    presets land under ``presetRuns`` for v2-aware consumers.
    """

    id: str
    source: str
    cls: str
    origin: str
    license: str
    splatCount: int
    bytesIn: int
    shDegree: int
    hash: str
    analyzeMs: int
    presetRuns: dict[str, dict[str, Any]] = field(default_factory=dict)

    # v0-schema mirrors — populated by `_apply_v0_mirrors` after measurement.
    webMobileSpzBytes: int = 0
    webMobileRatio: float = 0.0
    sizeMinSpzBytes: int = 0
    sizeMinRatio: float = 0.0

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        # `cls` → `class` to match v0 field name (Python reserved word).
        out["class"] = out.pop("cls")
        return out


def _which_catetus() -> str:
    """Locate the catetus CLI binary.

    Prefers ``$CATETUS_BIN``, then ``$PATH``, then the in-repo release
    build at ``target/release/catetus``. Raises if none are usable.
    """
    env = os.environ.get("CATETUS_BIN")
    if env and Path(env).exists():
        return env
    on_path = shutil.which("catetus")
    if on_path:
        return on_path
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "target" / "release" / "catetus"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError(
        "catetus CLI not found. Set $CATETUS_BIN, install it on PATH, "
        "or run `cargo build --release -p catetus-cli`."
    )


def _run_analyze(catetus: str, ply: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    proc = subprocess.run(
        [catetus, "analyze", str(ply)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    wall_ms = int((time.monotonic() - t0) * 1000)
    report = json.loads(proc.stdout)
    report["_analyzeMs"] = wall_ms
    return report


def _sum_output_bytes(manifest_path: Path) -> int:
    """Sum the glTF manifest plus all sidecar buffer files it references.

    The catetus optimize CLI emits a glTF JSON manifest with
    ``buffers[].uri`` pointing at sibling ``.bin`` files (typically under
    ``buffers/chunk_XXXX.bin``). The Modal worker bundles those into a
    single .glb at upload time, but the raw CLI output is multi-file. The
    user-observable transferred bytes are manifest + all buffers, so
    that's what we report.
    """
    total = manifest_path.stat().st_size
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:  # pragma: no cover - manifest unreadable, return what we have
        return total
    for buf in manifest.get("buffers", []) or []:
        uri = buf.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        sidecar = (manifest_path.parent / uri).resolve()
        if sidecar.exists():
            total += sidecar.stat().st_size
        else:
            # Fall back to the byteLength reported in the manifest if
            # the sidecar is missing for some reason (e.g. test fixture).
            total += int(buf.get("byteLength", 0))
    return total


def _run_optimize(
    catetus: str, ply: Path, preset: str, out_dir: Path
) -> PresetRun:
    """Run the canonical SplatBench pipeline for one preset.

    The published v0 numbers (e.g. bonsai web-mobile = 22.81×) measure the
    SPZ-packed output, not the intermediate glTF + sidecar buffers. The
    two-step pipeline is:

      1. ``catetus optimize --preset <name>`` → glTF manifest + sidecar
         ``.bin`` buffers.
      2. ``catetus convert --to spz`` → single ``.spz`` file.

    The reported ``outputBytes`` is the size of the ``.spz`` file. Wall
    time covers both steps so it's comparable to the worker's
    end-to-end latency.
    """
    work = out_dir / preset
    work.mkdir(parents=True, exist_ok=True)
    gltf = work / "scene.gltf"
    spz = work / "scene.spz"

    t0 = time.monotonic()
    try:
        subprocess.run(
            [catetus, "optimize", "--preset", preset, "-o", str(gltf), str(ply)],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        subprocess.run(
            [catetus, "convert", "--to", "spz", "-o", str(spz), str(gltf)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        return PresetRun(
            preset=preset,
            outputBytes=0,
            ratio=0.0,
            wallMs=int((time.monotonic() - t0) * 1000),
            ok=False,
            stderr=(e.stderr or "")[-1024:],
        )
    wall_ms = int((time.monotonic() - t0) * 1000)
    out_bytes = spz.stat().st_size if spz.exists() else 0
    in_bytes = ply.stat().st_size
    ratio = round(in_bytes / out_bytes, 2) if out_bytes else 0.0
    return PresetRun(
        preset=preset,
        outputBytes=out_bytes,
        ratio=ratio,
        wallMs=wall_ms,
    )


def _apply_v0_mirrors(row: SceneRow) -> None:
    wm = row.presetRuns.get("web-mobile")
    if wm and wm.get("ok"):
        row.webMobileSpzBytes = int(wm["outputBytes"])
        row.webMobileRatio = float(wm["ratio"])
    sm = row.presetRuns.get("size-min")
    if sm and sm.get("ok"):
        row.sizeMinSpzBytes = int(sm["outputBytes"])
        row.sizeMinRatio = float(sm["ratio"])


def measure_scene(
    scene_id: str,
    ply: Path,
    *,
    source: str = "real",
    cls: str,
    origin: str,
    license: str,
    presets: Iterable[str] = DEFAULT_PRESETS,
    work_dir: Path | None = None,
) -> SceneRow:
    """Run analyze + all presets on a local PLY. Returns a `SceneRow`."""
    catetus = _which_catetus()
    ply = ply.resolve()
    if not ply.exists():
        raise FileNotFoundError(ply)

    analyze = _run_analyze(catetus, ply)

    row = SceneRow(
        id=scene_id,
        source=source,
        cls=cls,
        origin=origin,
        license=license,
        splatCount=int(analyze.get("splatCount") or analyze.get("splat_count") or 0),
        bytesIn=ply.stat().st_size,
        shDegree=int(analyze.get("shDegree") or analyze.get("sh_degree") or 0),
        hash=str(analyze.get("hash") or analyze.get("blake3") or ""),
        analyzeMs=int(analyze["_analyzeMs"]),
    )

    cleanup_work = work_dir is None
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="bench-ingest-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        for preset in presets:
            run = _run_optimize(catetus, ply, preset, work_dir)
            row.presetRuns[preset] = {
                "outputBytes": run.outputBytes,
                "ratio": run.ratio,
                "wallMs": run.wallMs,
                "ok": run.ok,
                **({"stderr": run.stderr} if run.stderr else {}),
            }
    finally:
        if cleanup_work and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    _apply_v0_mirrors(row)
    return row


# ---------------------------------------------------------- idempotent merge


def merge_row(existing: list[dict[str, Any]], row: SceneRow) -> list[dict[str, Any]]:
    """Insert / replace a scene row in a list of v2-schema rows by ``id``."""
    new = row.to_json()
    out: list[dict[str, Any]] = []
    replaced = False
    for r in existing:
        if r.get("id") == new["id"]:
            out.append(new)
            replaced = True
        else:
            out.append(r)
    if not replaced:
        out.append(new)
    return out


def row_is_complete(row: dict[str, Any], presets: Iterable[str]) -> bool:
    """True if ``row`` already has a successful run for every preset."""
    runs = row.get("presetRuns") or {}
    for p in presets:
        r = runs.get(p)
        if not r or not r.get("ok") or not r.get("outputBytes"):
            return False
    return True
