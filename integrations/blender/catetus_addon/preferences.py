"""Add-on preferences.

Two concerns live here:

1. **CLI path** — auto-detected via :mod:`installer`, but always overridable.
   Stored in Blender's per-user add-on prefs, so it survives restarts but
   is not committed to the .blend file (which is critical — .blend files
   are shared assets and should never embed absolute user paths).

2. **API key** — the single biggest UX risk per the brief. Most Blender
   users do not have shell env vars set, so we cannot rely on
   ``$CATETUS_API_KEY``. We surface a password-masked text field in
   preferences, store it in Blender's user prefs (which already live
   under per-user ``config/userpref.blend`` — chmod 600 by default on
   POSIX), and additionally write it to ``~/.config/catetus/api_key``
   (chmod 600) on save so the CLI can pick it up via env without the
   user having to fiddle with ``.bashrc``.

The "additionally" wiring matters: when the operators run, they need
``CATETUS_API_KEY`` in the child process environment regardless of
whether Blender was launched from a terminal. We resolve that in
:func:`api_key_for_env` — preference field beats env var beats config
file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import AddonPreferences, Operator

from . import installer

# Per-user config dir, follows XDG on Linux, falls back to ~/.config on macOS
# (which Blender also uses), and ``%APPDATA%/Catetus`` on Windows.
def _config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming")))
        return base / "Catetus"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "catetus"
    return Path.home() / ".config" / "catetus"


def _api_key_file() -> Path:
    return _config_dir() / "api_key"


PRESETS = [
    ("web-mobile", "Web — Mobile", "Smallest GLB for phones / cellular networks"),
    ("web-desktop", "Web — Desktop", "Balanced quality+size for desktop browsers"),
    ("quest-browser", "Quest Browser", "Tuned for Meta Quest 2/3 in-headset WebGL"),
    ("visionos-preview", "VisionOS", "Apple Vision Pro preview-quality glTF"),
    ("thumbnail-preview", "Thumbnail", "Tiny preview for grids / cards"),
    ("quality-max", "Quality (Max)", "Lossless repack — biggest, exact quality"),
    ("size-min", "Size (Min)", "Aggressive — drops dim splats + quantizes hard"),
    ("lossless-repack", "Lossless Repack", "Round-trips byte-identical quality"),
]


def _save_api_key_to_disk(key: str) -> None:
    """Persist the API key to ``~/.config/catetus/api_key`` (chmod 600).

    We accept the small redundancy with Blender's own userpref storage so
    the same key works for users who later install the CLI standalone and
    invoke it from a terminal — they get continuity without having to
    re-paste anywhere.
    """

    if not key:
        # Empty input means "clear" — remove the on-disk file too, otherwise
        # a stale key would silently override the now-empty preference.
        try:
            _api_key_file().unlink()
        except FileNotFoundError:
            pass
        return
    target = _api_key_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(key.strip() + "\n", encoding="utf-8")
    if os.name != "nt":
        try:
            os.chmod(target, 0o600)
        except OSError:
            # Not fatal — the file is in a per-user dir already.
            pass


def _load_api_key_from_disk() -> str:
    try:
        return _api_key_file().read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def api_key_for_env(prefs: "CatetusPreferences") -> str:
    """Resolve the effective API key for a CLI subprocess invocation.

    Order: preference text field > env var > on-disk config file. Returning
    an empty string is fine — the API allows anonymous low-quota requests
    and the operator surfaces a clearer error on 401 than a missing key.
    """

    if prefs and prefs.api_key:
        return prefs.api_key
    env = os.environ.get("CATETUS_API_KEY")
    if env:
        return env
    return _load_api_key_from_disk()


def _on_api_key_update(self: "CatetusPreferences", _context: bpy.types.Context) -> None:
    # Persisting on every keystroke would chmod-thrash; bpy invokes update
    # callbacks on edit-commit, so this fires once per real change.
    _save_api_key_to_disk(self.api_key or "")


class CATETUS_OT_detect_cli(Operator):
    """Re-run CLI auto-detection and pin the result into preferences.

    Surfaced as a button on the prefs panel and the N-panel "not found" card.
    """

    bl_idname = "catetus.detect_cli"
    bl_label = "Detect catetus CLI"
    bl_description = (
        "Search PATH and well-known install locations for the catetus "
        "binary. Updates the preference if a working CLI is found."
    )
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):  # noqa: D401
        prefs = context.preferences.addons[__package__].preferences
        info = installer.detect_cli(prefs.cli_path or None)
        if info is None:
            self.report(
                {"ERROR"},
                "catetus CLI not found. " + installer.install_hint(),
            )
            return {"CANCELLED"}
        prefs.cli_path = info.path
        prefs.cli_version = info.version
        prefs.cli_source = info.source
        self.report(
            {"INFO"},
            f"Found {info.version} via {info.source}",
        )
        return {"FINISHED"}


class CatetusPreferences(AddonPreferences):
    bl_idname = __package__

    cli_path: StringProperty(
        name="catetus CLI",
        description=(
            "Absolute path to the catetus binary. Leave empty for auto-detect "
            "(PATH + well-known locations). Click 'Detect catetus CLI' to refresh."
        ),
        subtype="FILE_PATH",
        default="",
    )
    cli_version: StringProperty(name="Detected CLI Version", default="")
    cli_source: StringProperty(name="Resolution Source", default="")

    api_url: StringProperty(
        name="API URL",
        description="Catetus Cloud endpoint",
        default="https://catetus-api.fly.dev",
    )

    api_key: StringProperty(
        name="API Key",
        description=(
            "Personal Catetus Cloud API key. Mirrored to "
            "~/.config/catetus/api_key (chmod 600) so the CLI can find it"
        ),
        subtype="PASSWORD",  # masks the field in the UI
        default="",
        update=_on_api_key_update,
    )

    default_preset: EnumProperty(
        name="Default Preset",
        description="Preset selected on fresh open",
        items=PRESETS,
        default="web-mobile",
    )

    auto_open_output: BoolProperty(
        name="Open output folder after optimize",
        description="Reveal the optimized .glb in your file manager when the run finishes",
        default=True,
    )

    def draw(self, _context: bpy.types.Context) -> None:
        layout = self.layout

        box = layout.box()
        box.label(text="Catetus CLI", icon="CONSOLE")
        row = box.row(align=True)
        row.prop(self, "cli_path", text="")
        row.operator(CATETUS_OT_detect_cli.bl_idname, icon="VIEWZOOM", text="")
        if self.cli_version:
            box.label(text=f"Detected: {self.cli_version} ({self.cli_source})")
        else:
            box.label(
                text="Not detected — click the magnifier or paste an absolute path.",
                icon="ERROR",
            )
            box.label(text=f"Install: {installer.install_hint()}")

        cloud = layout.box()
        cloud.label(text="Catetus Cloud", icon="WORLD")
        cloud.prop(self, "api_url")
        cloud.prop(self, "api_key")
        cloud.label(
            text=(
                "API key is stored in Blender prefs + mirrored to "
                "~/.config/catetus/api_key (chmod 600)."
            ),
            icon="LOCKED",
        )

        defaults = layout.box()
        defaults.label(text="Defaults", icon="PREFERENCES")
        defaults.prop(self, "default_preset")
        defaults.prop(self, "auto_open_output")


_CLASSES = (CATETUS_OT_detect_cli, CatetusPreferences)


def get_prefs(context: Optional[bpy.types.Context] = None) -> CatetusPreferences:
    ctx = context or bpy.context
    return ctx.preferences.addons[__package__].preferences


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    # First-run auto-detect, but only if no path is stored yet — we never
    # silently overwrite a user-set preference.
    try:
        prefs = get_prefs()
        if not prefs.cli_path:
            info = installer.detect_cli(None)
            if info is not None:
                prefs.cli_path = info.path
                prefs.cli_version = info.version
                prefs.cli_source = info.source
    except (AttributeError, KeyError):
        # During Blender's startup the addon prefs may not be live yet —
        # the panel's poll will re-detect when the user opens the N-panel.
        pass


def unregister() -> None:
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
