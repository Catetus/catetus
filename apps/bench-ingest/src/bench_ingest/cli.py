"""CLI shim for the SplatBench v2 ingest harness.

Two subcommands:

- ``measure`` — measure a single local PLY, append/replace the row in
  ``benches/reports/splatbench-v2.json``.
- ``batch``   — read ``benches/scenes/real/manifest.json`` (or any manifest
  in the same schema) and measure every scene whose row is not yet complete.

Both are idempotent: re-running with the same inputs and a fully-populated
JSON is a no-op unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .measure import (
    DEFAULT_PRESETS,
    measure_scene,
    merge_row,
    row_is_complete,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO_ROOT / "benches" / "reports" / "splatbench-v2.json"
DEFAULT_MANIFEST = REPO_ROOT / "benches" / "scenes" / "real" / "manifest.json"


def _load_existing(out_path: Path) -> dict:
    if not out_path.exists():
        return {
            "schema": "catetus.splatbench.v2/0.1",
            "name": "SplatBench v2 — real-photo corpus",
            "description": (
                "Real-photo 3DGS scenes measured through the catetus "
                "optimize CLI. Source of truth for the v2 expansion; "
                "merged into splatbench-v0.json by sync-splatbench.mjs."
            ),
            "presets": list(DEFAULT_PRESETS),
            "rows": [],
        }
    return json.loads(out_path.read_text())


def _save(out_path: Path, doc: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")


def cmd_measure(args: argparse.Namespace) -> int:
    presets = tuple(args.presets) if args.presets else DEFAULT_PRESETS
    out_path = Path(args.out).resolve()
    doc = _load_existing(out_path)

    # Idempotence check before doing real work.
    rows = doc.get("rows", [])
    existing = next((r for r in rows if r.get("id") == args.scene_id), None)
    if existing and not args.force and row_is_complete(existing, presets):
        print(
            f"[bench-ingest] {args.scene_id} already complete for "
            f"{','.join(presets)} — skipping (use --force to re-measure)."
        )
        return 0

    row = measure_scene(
        scene_id=args.scene_id,
        ply=Path(args.ply),
        source=args.source,
        cls=args.cls,
        origin=args.origin,
        license=args.license,
        presets=presets,
    )
    doc["rows"] = merge_row(rows, row)
    doc["presets"] = list(presets)
    _save(out_path, doc)
    print(f"[bench-ingest] wrote {args.scene_id} → {out_path}")
    for preset, run in row.presetRuns.items():
        status = "OK " if run.get("ok") else "FAIL"
        print(
            f"  {status} {preset:18s} {run.get('outputBytes', 0):>12,} B "
            f"ratio={run.get('ratio'):>6.2f}× wall={run.get('wallMs'):>6,} ms"
        )
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    presets = tuple(args.presets) if args.presets else DEFAULT_PRESETS
    out_path = Path(args.out).resolve()
    doc = _load_existing(out_path)
    rows = doc.get("rows", [])

    failures: list[str] = []
    for entry in manifest.get("scenes", []):
        scene_id = entry["id"]
        ply = (manifest_path.parent / entry["filename"]).resolve()
        if not ply.exists():
            print(f"[bench-ingest] {scene_id}: missing {ply} — skipping")
            continue

        existing = next((r for r in rows if r.get("id") == scene_id), None)
        if existing and not args.force and row_is_complete(existing, presets):
            print(f"[bench-ingest] {scene_id}: complete, skipping")
            continue

        try:
            row = measure_scene(
                scene_id=scene_id,
                ply=ply,
                source="real",
                cls=entry.get("class", "real-scene"),
                origin=entry.get("sourceUrl", "unknown"),
                license=entry.get("license", "unknown"),
                presets=presets,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[bench-ingest] {scene_id}: FAIL ({e})")
            failures.append(scene_id)
            continue

        rows = merge_row(rows, row)
        doc["rows"] = rows
        doc["presets"] = list(presets)
        _save(out_path, doc)  # checkpoint after every scene
        print(f"[bench-ingest] {scene_id}: wrote {len(row.presetRuns)} preset rows")

    if failures:
        print(f"[bench-ingest] DONE with failures: {','.join(failures)}", file=sys.stderr)
        return 2
    print("[bench-ingest] batch complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench-ingest")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="Measure a single local PLY.")
    m.add_argument("--scene-id", required=True)
    m.add_argument("--ply", required=True)
    m.add_argument("--source", default="real", choices=["real", "synthetic"])
    m.add_argument("--class", dest="cls", required=True)
    m.add_argument("--origin", required=True)
    m.add_argument("--license", required=True)
    m.add_argument("--presets", nargs="+")
    m.add_argument("--out", default=str(DEFAULT_OUT))
    m.add_argument("--force", action="store_true")
    m.set_defaults(func=cmd_measure)

    b = sub.add_parser("batch", help="Measure every scene in a manifest.")
    b.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    b.add_argument("--presets", nargs="+")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_batch)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
