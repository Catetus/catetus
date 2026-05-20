"""N-panel UI for the Catetus add-on.

Lives in the 3D viewport sidebar under the **Catetus** tab. Layout:

  ┌─ Catetus ─────────────────────┐
  │ CLI: catetus 0.1.2 (PATH)     │   <- green when detected, red when not
  │ [Import Splat]                   │
  │ Preset: [web-mobile ▼]           │
  │ [Optimize]      [Submit to Cloud]│
  │ ┌──────────────────────────────┐ │
  │ │ optimize: encoding-gltf      │ │
  │ │ ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░  47 %  │ │   <- live during modal ops
  │ └──────────────────────────────┘ │
  │ Last output: bonsai.web-mobile.glb│
  │ [Open .glb] [Fidelity] [Share ↗] │
  └──────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Panel, PropertyGroup

from . import installer
from .operators import STATUS
from .preferences import PRESETS, get_prefs


class CatetusSceneProps(PropertyGroup):
    """Scene-scoped properties surfaced in the panel.

    Stored on ``bpy.types.Scene`` so they persist across operator runs
    and live alongside the .blend file (so re-opening a scene remembers
    "I was working on the web-mobile preset"). The API key explicitly
    does NOT live here — it stays in add-on prefs, which are per-user.
    """

    preset: EnumProperty(
        name="Preset",
        items=PRESETS,
        default="web-mobile",
    )
    compress: BoolProperty(
        name="zstd-compress buffers",
        description="Emit .bin.zst sidecars after optimize (halves file size)",
        default=False,
    )
    cloud_label: StringProperty(
        name="Label",
        description="Optional human-readable label stamped on the cloud job",
        default="",
    )


class CATETUS_PT_main(Panel):
    bl_idname = "CATETUS_PT_main"
    bl_label = "Catetus"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Catetus"

    def draw(self, context):
        layout = self.layout
        prefs = get_prefs(context)
        scn = context.scene.catetus

        # CLI status header — green check or red error with install hint.
        header = layout.box()
        if prefs.cli_path and prefs.cli_version:
            row = header.row()
            row.label(text=prefs.cli_version, icon="CHECKMARK")
            row.operator("catetus.detect_cli", icon="FILE_REFRESH", text="")
        else:
            col = header.column(align=True)
            col.label(text="catetus CLI not found", icon="ERROR")
            col.label(text=installer.install_hint())
            col.operator("catetus.detect_cli", icon="VIEWZOOM", text="Detect")

        # Import row — wraps the file selector.
        layout.operator(
            "catetus.import_splat",
            text="Import Splat (.ply / .spz / .glb)",
            icon="IMPORT",
        )

        # Active selection summary — surfaces *what* the next op acts on.
        active = context.active_object
        if active is not None and active.get("catetus_source"):
            src = Path(str(active["catetus_source"]))
            sel = layout.box()
            sel.label(text=f"Source: {src.name}", icon="FILE_3D")

        # Preset chooser + the two big buttons.
        layout.separator()
        layout.prop(scn, "preset")
        row = layout.row(align=True)
        row.scale_y = 1.4
        op_opt = row.operator(
            "catetus.optimize_local",
            text="Optimize",
            icon="MODIFIER",
        )
        op_opt.preset = scn.preset
        op_opt.compress = scn.compress
        op_sub = row.operator(
            "catetus.submit_to_cloud",
            text="Submit to Cloud",
            icon="EXPORT",
        )
        op_sub.preset = scn.preset
        op_sub.label = scn.cloud_label

        layout.prop(scn, "compress")
        layout.prop(scn, "cloud_label")

        # Progress / status — drives off the shared STATUS dict the modal
        # operators publish into. Blender does not ship a stock progress
        # widget; we fake one with the slider's value-as-bar visual.
        if STATUS.get("running") or STATUS.get("message"):
            prog_box = layout.box()
            msg = STATUS.get("message") or ""
            prog_box.label(text=msg[:60])
            frac = float(STATUS.get("progress") or 0.0)
            # The dummy WM property below is a 0..1 float we only use as
            # a visualization channel — clicking it does nothing.
            sub = prog_box.row()
            sub.enabled = False
            context.window_manager.catetus_progress = frac
            sub.prop(
                context.window_manager,
                "catetus_progress",
                text=f"{int(frac * 100)}%",
                slider=True,
            )

        # Output viewer — buttons enabled only when relevant.
        layout.separator()
        out_path = STATUS.get("last_output") or ""
        if out_path:
            out_box = layout.box()
            out_box.label(text=f"Output: {Path(str(out_path)).name}", icon="FILE_TICK")
            row = out_box.row(align=True)
            row.operator(
                "catetus.fetch_fidelity",
                text="Fidelity Report",
                icon="VIEWZOOM",
            )
            row.operator(
                "catetus.open_share_link",
                text="Open Share Link",
                icon="URL",
            )


class CATETUS_PT_cloud(Panel):
    """Cloud-specific status sub-panel; collapses by default."""

    bl_idname = "CATETUS_PT_cloud"
    bl_label = "Cloud Job"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Catetus"
    bl_parent_id = "CATETUS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        prefs = get_prefs(context)
        layout.label(
            text="API URL:" if prefs.api_url else "API URL not set",
            icon="WORLD" if prefs.api_url else "ERROR",
        )
        layout.label(text=prefs.api_url or "(set in add-on prefs)")
        layout.label(
            text="API key configured" if prefs.api_key else "No API key — anonymous",
            icon="LOCKED" if prefs.api_key else "UNLOCKED",
        )
        share = STATUS.get("last_share_url") or ""
        if share:
            layout.label(text="Last share URL:")
            layout.label(text=str(share)[:48] + ("..." if len(str(share)) > 48 else ""))


_CLASSES = (CatetusSceneProps, CATETUS_PT_main, CATETUS_PT_cloud)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.catetus = bpy.props.PointerProperty(type=CatetusSceneProps)
    # Progress visualization channel — read-only float on the WM.
    bpy.types.WindowManager.catetus_progress = bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0
    )


def unregister() -> None:
    try:
        del bpy.types.WindowManager.catetus_progress
    except (AttributeError, RuntimeError):
        pass
    try:
        del bpy.types.Scene.catetus
    except (AttributeError, RuntimeError):
        pass
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
