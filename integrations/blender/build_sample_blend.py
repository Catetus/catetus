"""Generate the demo SAMPLE.blend file.

Run with:

    blender --background --python integrations/blender/build_sample_blend.py

(or from inside Blender: ``Text Editor → Open → build_sample_blend.py → Run``)

Produces ``integrations/blender/SAMPLE.blend`` — a scene with one Empty
labelled "Bonsai Splat" with the ``catetus_source`` /
``catetus_glb`` custom properties pre-filled, plus a camera positioned
so the Catetus N-panel is visible on open. The Empty's stamped paths
point at the bonsai sample inside ``fixtures/`` so a user with the repo
checked out can hit **Optimize** without browsing for a file.

The file is intentionally tiny (~10 KB). It is committed to the repo so
end-users can `git clone` the demo without rebuilding it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover — bpy is only available inside Blender
    sys.stderr.write(
        "This script must be run via Blender:\n"
        "  blender --background --python build_sample_blend.py\n"
    )
    sys.exit(2)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SAMPLE_PATH = SCRIPT_DIR / "SAMPLE.blend"

# Point at the bonsai sample committed to the repo. We use Blender's
# `//` relative-path convention so the stamp resolves regardless of where
# the user has the repo checked out — `bpy.path.abspath` expands it
# against the open .blend's directory. The optimize operator's call to
# `bpy.path.abspath` makes this transparent.
DEMO_SOURCE = "//../../fixtures/splats/bonsai-7k.ply"


def reset_scene() -> None:
    """Start from a clean scene so re-running the generator is idempotent."""

    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_empty() -> None:
    bpy.ops.object.empty_add(type="SPHERE", radius=0.5, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "Bonsai Splat"
    obj["catetus_source"] = DEMO_SOURCE
    obj["catetus_glb"] = DEMO_SOURCE  # same file — the import op will convert lazily


def make_camera_and_light() -> None:
    bpy.ops.object.camera_add(location=(4, -4, 3), rotation=(1.1, 0, 0.785))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))


def write_sample() -> None:
    reset_scene()
    make_empty()
    make_camera_and_light()
    # Default preset stamp on the scene — the panel reads ``scene.catetus``
    # if the add-on is enabled, and falls back gracefully if it isn't.
    bpy.context.scene["catetus_default_preset"] = "web-mobile"
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save uncompressed so the file header is plain ASCII (BLENDER-vNNN) and
    # the file opens identically on every LTS release. The size delta on a
    # scene with one Empty + camera + light is ~80 KB vs ~10 KB; trivial.
    bpy.ops.wm.save_as_mainfile(filepath=str(SAMPLE_PATH), compress=False)
    print(f"wrote {SAMPLE_PATH} ({os.path.getsize(SAMPLE_PATH)} bytes)")


if __name__ == "__main__":
    write_sample()
