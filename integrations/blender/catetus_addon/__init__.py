"""Catetus Blender add-on.

Wraps the ``catetus`` CLI and Catetus Cloud API so Blender users can
import a Gaussian-Splat file (.ply / .spz / .glb), optimize it for a target
device preset, and either keep the output locally or submit it to Catetus
Cloud — all from an N-panel inside the 3D viewport.

The add-on does NOT bundle the CLI. It auto-detects the user's installed
``catetus`` binary (homebrew / cargo-install / explicit path) and surfaces
a clear error with an install hint if it cannot be found.

Blender entry points: every module that registers classes exposes
``register()`` / ``unregister()`` and is wired up below.
"""

from __future__ import annotations

import importlib

bl_info = {
    "name": "Catetus",
    "description": (
        "Production-grade Gaussian Splat optimizer. Import .ply / .spz / .glb, "
        "optimize for web / mobile / Quest / VisionOS, and ship as glb."
    ),
    "author": "Catetus contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Catetus",
    "warning": "",
    "doc_url": "https://github.com/Catetus/catetus/tree/main/integrations/blender",
    "tracker_url": "https://github.com/Catetus/catetus/issues",
    "support": "COMMUNITY",
    "category": "Import-Export",
}


# Submodules are imported lazily so ``addon_utils.modules()`` can introspect
# ``bl_info`` above without running registration side-effects.
_MODULES = (
    "preferences",
    "installer",
    "operators",
    "panels",
)


def _resolved_modules():
    """Return imported / reloaded submodule objects in load order."""

    resolved = []
    for name in _MODULES:
        full = f"{__package__}.{name}"
        if full in importlib.sys.modules:
            module = importlib.reload(importlib.sys.modules[full])
        else:
            module = importlib.import_module(full)
        resolved.append(module)
    return resolved


def register() -> None:
    for module in _resolved_modules():
        if hasattr(module, "register"):
            module.register()


def unregister() -> None:
    # Unregister in reverse to satisfy Blender's class-dependency rules
    # (panels reference operators reference property groups reference prefs).
    for module in reversed(_resolved_modules()):
        if hasattr(module, "unregister"):
            try:
                module.unregister()
            except Exception:  # pragma: no cover — Blender-only path
                # Unregistration must never raise during teardown — Blender
                # will leak handlers otherwise.
                pass


if __name__ == "__main__":  # pragma: no cover
    register()
