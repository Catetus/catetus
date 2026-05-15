"""Blender operators that drive the SplatForge CLI + Cloud API.

All long-running operators follow Blender's **modal operator** pattern:

* ``execute`` / ``invoke`` spawns a worker thread (or a subprocess plus
  a reader thread) and registers a ``wm.event_timer_add`` ticker.
* ``modal`` runs on the main thread at every timer tick and drains a
  ``queue.Queue`` populated by the worker. This is the only place we
  touch ``bpy`` state — Blender's data model is NOT thread-safe.
* ``cancel`` tears down the timer and signals the worker to stop.

This keeps the UI responsive — submitting a 200 MB scene to SplatForge
Cloud no longer locks the viewport.

Five operators are exposed:

* ``splatforge.import_splat``       — file-browser import (.ply/.spz/.glb)
* ``splatforge.optimize_local``     — run ``splatforge optimize``
* ``splatforge.submit_to_cloud``    — POST /v1/jobs + upload + poll
* ``splatforge.fetch_fidelity``     — run ``splatforge fidelity``
* ``splatforge.open_share_link``    — open the last cloud URL in a browser
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator

from . import installer
from .preferences import PRESETS, api_key_for_env, get_prefs


# --- shared state -----------------------------------------------------------

# Buffered status messages from background workers, drained by the panel's
# modal poll. Kept on a module attribute so the panel and the operator can
# share it without juggling Blender properties (which would re-render the
# whole UI on every push).
STATUS: dict[str, object] = {
    "phase": "",          # short tag, e.g. "uploading"
    "message": "",        # human-readable line shown under the buttons
    "progress": 0.0,      # 0..1 for the progress bar
    "last_output": "",    # filesystem path of most recent optimize output
    "last_share_url": "", # most recent cloud /jobs/<id> share URL
    "running": False,     # whether any modal operator is active
}


def _reveal_in_file_manager(path: Path) -> None:
    """Open the parent folder in the host file manager (best-effort).

    We intentionally fail silently — the operator already succeeded by the
    time this runs; if file-manager invocation breaks (no $DISPLAY, sandboxed
    Flatpak Blender, …) we'd rather not turn a success into an error.
    """

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError:
        pass


def _ensure_cli(operator: Operator) -> Optional[str]:
    """Resolve the splatforge CLI path or report an actionable error.

    Returns the binary path, or ``None`` and reports the error to the
    operator (so the caller can ``return {'CANCELLED'}``).
    """

    prefs = get_prefs()
    info = installer.detect_cli(prefs.cli_path or None)
    if info is None:
        operator.report(
            {"ERROR"},
            f"splatforge CLI not found. {installer.install_hint()}",
        )
        return None
    # Persist resolution so the next op skips the probe.
    prefs.cli_path = info.path
    prefs.cli_version = info.version
    prefs.cli_source = info.source
    return info.path


def _child_env(prefs) -> dict[str, str]:
    """Build the env dict for CLI subprocesses.

    We replicate the API key into the subprocess env so the CLI's
    ``submit`` path picks it up even when Blender was launched from a
    GUI shortcut with no shell env inheritance.
    """

    env = os.environ.copy()
    api_key = api_key_for_env(prefs)
    if api_key:
        env["SPLATFORGE_API_KEY"] = api_key
    if prefs.api_url:
        env["SPLATFORGE_API_URL"] = prefs.api_url
    return env


# --- 1. import_splat --------------------------------------------------------


class SPLATFORGE_OT_import_splat(Operator):
    """Import a Gaussian-Splat file.

    .glb files are loaded with Blender's native glTF importer (they are
    valid glTF 2.0 containers with the ``KHR_gaussian_splatting`` extension).
    .ply / .spz / .usda / .usdc files are first converted to .glb via
    ``splatforge convert`` and then imported the same way — this means the
    splat appears in the scene outliner with the file basename as the
    object name, which is what KIRI Engine's add-on does and what
    Blender users expect.

    We deliberately import as a mesh placeholder (Blender 4.2 has no
    native splat shader); the original file path is stamped on the
    object's custom properties so subsequent operators (optimize, submit)
    know what bytes to ship.
    """

    bl_idname = "splatforge.import_splat"
    bl_label = "Import Splat (.ply / .spz / .glb)"
    bl_description = (
        "Import a Gaussian-Splat file. Auto-converts non-glb formats via the "
        "splatforge CLI."
    )
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(
        default="*.ply;*.spz;*.glb;*.gltf;*.usda;*.usdc",
        options={"HIDDEN"},
    )

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, _context):
        src = Path(bpy.path.abspath(self.filepath))
        if not src.is_file():
            self.report({"ERROR"}, f"file not found: {src}")
            return {"CANCELLED"}

        ext = src.suffix.lower()
        glb_path = src
        if ext != ".glb":
            cli = _ensure_cli(self)
            if cli is None:
                return {"CANCELLED"}
            # Write the converted glb next to the source so the user can
            # locate it again — and so re-importing the same file is a
            # no-op rather than re-running conversion.
            glb_path = src.with_suffix(".glb")
            if not glb_path.exists():
                prefs = get_prefs()
                try:
                    subprocess.run(
                        [cli, "convert", str(src), "--to", "glb", "-o", str(glb_path)],
                        check=True,
                        env=_child_env(prefs),
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as e:
                    self.report({"ERROR"}, f"convert failed: {e.stderr or e}")
                    return {"CANCELLED"}

        try:
            bpy.ops.import_scene.gltf(filepath=str(glb_path))
        except RuntimeError as e:
            self.report({"ERROR"}, f"glTF import failed: {e}")
            return {"CANCELLED"}

        # Stamp the source on every freshly-imported object so the optimize
        # operator can find the bytes again without re-prompting.
        for obj in bpy.context.selected_objects:
            obj["splatforge_source"] = str(src)
            obj["splatforge_glb"] = str(glb_path)

        STATUS["message"] = f"Imported {src.name}"
        self.report({"INFO"}, STATUS["message"])
        return {"FINISHED"}


# --- 2. optimize_local ------------------------------------------------------


def _selected_source_path() -> Optional[Path]:
    """Find the splat source path stamped on the active object (or any
    selected one).

    The stamped path may use Blender's ``//`` relative-path convention
    (see ``SAMPLE.blend``); we run it through ``bpy.path.abspath`` so the
    same .blend works on every checkout regardless of repo location.
    """

    for obj in [bpy.context.active_object, *bpy.context.selected_objects]:
        if obj is None:
            continue
        src = obj.get("splatforge_source") or obj.get("splatforge_glb")
        if src:
            return Path(bpy.path.abspath(str(src)))
    return None


class SPLATFORGE_OT_optimize_local(Operator):
    """Run ``splatforge optimize`` against the selected splat file.

    Modal so the UI stays responsive — the optimize pipeline streams
    ``PROGRESS frac=… stage=…`` lines on stdout, which we parse and
    surface on the panel's progress bar.
    """

    bl_idname = "splatforge.optimize_local"
    bl_label = "Optimize"
    bl_description = (
        "Run the splatforge CLI with the chosen preset on the imported splat. "
        "Writes a .glb next to the source and a JSON optimize report."
    )
    bl_options = {"REGISTER"}

    preset: EnumProperty(items=PRESETS, default="web-mobile")
    compress: BoolProperty(
        name="zstd-compress buffers",
        default=False,
        description="Emit .bin.zst sidecars (halves on-disk size, zero quality cost)",
    )

    _timer = None
    _proc: Optional[subprocess.Popen] = None
    _q: Optional["queue.Queue[str]"] = None
    _reader: Optional[threading.Thread] = None
    _out_path: Optional[Path] = None
    _progress_re = re.compile(r"^PROGRESS frac=([0-9.]+) stage=(\S+)")

    def execute(self, context):
        src = _selected_source_path()
        if src is None:
            self.report(
                {"ERROR"},
                "No splat source stamped on selection. Use 'Import Splat' first.",
            )
            return {"CANCELLED"}
        cli = _ensure_cli(self)
        if cli is None:
            return {"CANCELLED"}

        # Optimize always emits a glb sibling — predictable output path lets
        # us auto-open it and feeds the cloud-submit op without re-prompting.
        out = src.with_name(f"{src.stem}.{self.preset}.glb")
        self._out_path = out

        args = [
            cli,
            "optimize",
            str(src),
            "--preset",
            self.preset,
            "-o",
            str(out),
            "--progress",
        ]
        if self.compress:
            args.append("--compress")

        prefs = get_prefs()
        STATUS["phase"] = "optimize"
        STATUS["message"] = f"Optimizing {src.name} → {out.name}"
        STATUS["progress"] = 0.0
        STATUS["running"] = True

        try:
            self._proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_child_env(prefs),
            )
        except OSError as e:
            self._reset()
            self.report({"ERROR"}, f"spawn failed: {e}")
            return {"CANCELLED"}

        self._q = queue.Queue()
        self._reader = threading.Thread(
            target=self._pump, args=(self._proc, self._q), daemon=True
        )
        self._reader.start()

        wm = context.window_manager
        # 200 ms tick — quick enough that the progress bar feels live, slow
        # enough that we're not redrawing the panel every frame.
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    @staticmethod
    def _pump(proc: subprocess.Popen, q: "queue.Queue[str]") -> None:
        # Reading on the worker thread means a 5-minute optimize never blocks
        # the Blender main thread, which would freeze the entire UI.
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line.rstrip("\n"))
        proc.wait()
        q.put(f"__exit__ {proc.returncode}")

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            while True:
                line = self._q.get_nowait()
                if line.startswith("__exit__"):
                    code = int(line.split()[1])
                    return self._finish(code)
                m = self._progress_re.match(line)
                if m:
                    STATUS["progress"] = float(m.group(1))
                    STATUS["message"] = f"optimize: {m.group(2)}"
                else:
                    # Surface stderr-via-stdout lines so the user can read
                    # failure context inline in the panel.
                    STATUS["message"] = line[-160:]
        except queue.Empty:
            pass
        # Force the N-panel to redraw so the bar advances even when the
        # mouse is stationary.
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _finish(self, returncode: int):
        STATUS["progress"] = 1.0
        STATUS["running"] = False
        if returncode != 0:
            STATUS["message"] = f"optimize failed (exit {returncode})"
            self.report({"ERROR"}, STATUS["message"])
            self._reset()
            return {"CANCELLED"}
        STATUS["last_output"] = str(self._out_path) if self._out_path else ""
        STATUS["message"] = f"Done → {self._out_path.name if self._out_path else 'optimize'}"
        self.report({"INFO"}, STATUS["message"])
        prefs = get_prefs()
        if prefs.auto_open_output and self._out_path:
            _reveal_in_file_manager(self._out_path)
        self._reset()
        return {"FINISHED"}

    def _reset(self):
        if self._timer is not None:
            try:
                bpy.context.window_manager.event_timer_remove(self._timer)
            except (RuntimeError, AttributeError):
                pass
            self._timer = None
        self._proc = None
        self._q = None
        self._reader = None

    def cancel(self, _context):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass
        self._reset()
        STATUS["running"] = False
        STATUS["message"] = "cancelled"
        return {"CANCELLED"}


# --- 3. submit_to_cloud -----------------------------------------------------


class SPLATFORGE_OT_submit_to_cloud(Operator):
    """Submit the selected splat to SplatForge Cloud.

    Non-blocking: a worker thread does the POST + upload + polling, and
    drips status messages back through ``STATUS`` for the panel.

    We use stdlib ``urllib`` rather than ``requests`` because Blender
    does not bundle ``requests`` and pip-installing into Blender's Python
    is something we DO NOT want to ask end-users to do.
    """

    bl_idname = "splatforge.submit_to_cloud"
    bl_label = "Submit to Cloud"
    bl_description = (
        "Upload the selected splat to SplatForge Cloud, run the chosen preset "
        "on managed compute, and return a shareable URL"
    )
    bl_options = {"REGISTER"}

    preset: EnumProperty(items=PRESETS, default="web-mobile")
    label: StringProperty(name="Label", default="")

    _timer = None
    _worker: Optional[threading.Thread] = None
    _q: Optional["queue.Queue[dict]"] = None
    _stop_flag: Optional[threading.Event] = None

    def execute(self, context):
        src = _selected_source_path()
        if src is None:
            self.report({"ERROR"}, "No splat source stamped on selection.")
            return {"CANCELLED"}
        prefs = get_prefs()
        if not prefs.api_key:
            # Empty key still proceeds — many endpoints accept anonymous —
            # but warn the user via STATUS so they can paste a key if 401s.
            STATUS["message"] = "warning: no API key in prefs; submitting anonymously"

        STATUS["phase"] = "submit"
        STATUS["progress"] = 0.0
        STATUS["running"] = True

        self._q = queue.Queue()
        self._stop_flag = threading.Event()
        self._worker = threading.Thread(
            target=self._run_submit,
            args=(
                str(src),
                self.preset,
                self.label,
                prefs.api_url,
                prefs.api_key or os.environ.get("SPLATFORGE_API_KEY", ""),
                self._q,
                self._stop_flag,
            ),
            daemon=True,
        )
        self._worker.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    # The worker NEVER touches bpy. Everything it produces flows through `q`.
    @staticmethod
    def _run_submit(
        src_path: str,
        preset: str,
        label: str,
        api_url: str,
        api_key: str,
        q: "queue.Queue[dict]",
        stop_flag: threading.Event,
    ) -> None:
        api = api_url.rstrip("/") if api_url else "https://splatforge-api.fly.dev"
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        src = Path(src_path)

        def _post(path: str, body: bytes, ctype: str) -> dict:
            req = urllib.request.Request(
                f"{api}{path}",
                data=body,
                method="POST",
                headers={**headers, "content-type": ctype},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")

        def _get(path: str) -> dict:
            req = urllib.request.Request(f"{api}{path}", headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")

        try:
            q.put({"message": "creating job", "progress": 0.05})
            body = {
                "preset": preset,
                "filename": src.name,
                "size_bytes": src.stat().st_size,
            }
            if label:
                body["label"] = label
            create = _post(
                "/v1/jobs", json.dumps(body).encode("utf-8"), "application/json"
            )
            job_id = create.get("id")
            if not job_id:
                q.put({"error": f"missing job id in create response: {create}"})
                return

            q.put({"message": f"uploading {src.name}", "progress": 0.15, "job_id": job_id})
            with src.open("rb") as fh:
                payload = fh.read()
            _post(
                f"/v1/jobs/{job_id}/upload",
                payload,
                "application/octet-stream",
            )

            # Polling loop: 5-second cadence (matches the CLI default) with
            # an early exit on stop_flag so the Blender modal can cancel.
            q.put({"message": "queued", "progress": 0.25, "job_id": job_id})
            deadline = time.time() + 900  # 15 min cap to match CLI
            while time.time() < deadline:
                if stop_flag.is_set():
                    q.put({"error": "cancelled"})
                    return
                # Sleep in short slices so cancellation is responsive.
                for _ in range(50):
                    if stop_flag.is_set():
                        q.put({"error": "cancelled"})
                        return
                    time.sleep(0.1)
                pj = _get(f"/v1/jobs/{job_id}")
                status = pj.get("status", "unknown")
                phase = pj.get("phase") or status
                # Bias the progress bar to never go past 0.95 until done —
                # the optimizer reports its own fraction inside the API
                # response when it streams progress through.
                frac = float(pj.get("progress") or 0.0)
                frac = max(0.25, min(0.95, frac if frac > 0 else 0.5))
                q.put(
                    {
                        "message": f"cloud: {phase}",
                        "progress": frac,
                        "job_id": job_id,
                    }
                )
                if status in ("done", "succeeded"):
                    out = pj.get("output_url")
                    if not out:
                        q.put({"error": "done but no output_url"})
                        return
                    share = f"{api}/v1/jobs/{job_id}"
                    q.put(
                        {
                            "done": True,
                            "output_url": out,
                            "share_url": share,
                            "progress": 1.0,
                            "message": "cloud: done",
                        }
                    )
                    return
                if status in ("error", "failed"):
                    q.put({"error": pj.get("error") or "cloud job failed"})
                    return
            q.put({"error": "cloud timeout (15 min)"})
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            q.put({"error": f"HTTP {e.code}: {body[:200]}"})
        except urllib.error.URLError as e:
            q.put({"error": f"network: {e.reason}"})
        except Exception as e:  # last-resort — never let a bg thread die silently
            q.put({"error": f"{type(e).__name__}: {e}"})

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            while True:
                msg = self._q.get_nowait()
                if "error" in msg:
                    STATUS["message"] = f"submit failed: {msg['error']}"
                    STATUS["running"] = False
                    self.report({"ERROR"}, STATUS["message"])
                    self._reset()
                    return {"CANCELLED"}
                if "progress" in msg:
                    STATUS["progress"] = float(msg["progress"])
                if "message" in msg:
                    STATUS["message"] = str(msg["message"])
                if msg.get("done"):
                    STATUS["last_share_url"] = str(msg.get("share_url", ""))
                    STATUS["last_output"] = str(msg.get("output_url", ""))
                    STATUS["running"] = False
                    self.report(
                        {"INFO"},
                        f"Cloud done: {msg.get('share_url')}",
                    )
                    self._reset()
                    return {"FINISHED"}
        except queue.Empty:
            pass
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _reset(self):
        if self._timer is not None:
            try:
                bpy.context.window_manager.event_timer_remove(self._timer)
            except (RuntimeError, AttributeError):
                pass
            self._timer = None
        self._worker = None
        self._q = None
        self._stop_flag = None

    def cancel(self, _context):
        if self._stop_flag is not None:
            self._stop_flag.set()
        STATUS["running"] = False
        STATUS["message"] = "cancelled"
        self._reset()
        return {"CANCELLED"}


# --- 4. fetch_fidelity_report -----------------------------------------------


class SPLATFORGE_OT_fetch_fidelity(Operator):
    """Render a fidelity report comparing the optimized output to the source.

    Wraps ``splatforge fidelity --baseline <source> <optimized>``. We point
    the report at a ``reports/fidelity/<stem>`` folder next to the source.
    """

    bl_idname = "splatforge.fetch_fidelity"
    bl_label = "Fidelity Report"
    bl_description = (
        "Run the deterministic 8-orbit fidelity diff (ΔE94 / SSIM / pixelmatch) "
        "against the baseline. Writes report.json next to the optimized file."
    )
    bl_options = {"REGISTER"}

    def execute(self, _context):
        src = _selected_source_path()
        if src is None:
            self.report({"ERROR"}, "No splat source stamped on selection.")
            return {"CANCELLED"}
        last_out = STATUS.get("last_output") or ""
        candidate = Path(last_out) if last_out else None
        if candidate is None or not candidate.exists() or candidate.suffix not in (".glb", ".gltf"):
            self.report(
                {"ERROR"},
                "No optimized output found. Run 'Optimize' first.",
            )
            return {"CANCELLED"}
        cli = _ensure_cli(self)
        if cli is None:
            return {"CANCELLED"}
        out_dir = candidate.with_suffix("").with_name(f"{candidate.stem}-fidelity")
        out_dir.mkdir(parents=True, exist_ok=True)
        args = [
            cli,
            "fidelity",
            str(candidate),
            "--baseline",
            str(src),
            "-o",
            str(out_dir),
        ]
        try:
            r = subprocess.run(
                args,
                capture_output=True,
                text=True,
                env=_child_env(get_prefs()),
                timeout=600,
            )
        except OSError as e:
            self.report({"ERROR"}, f"spawn failed: {e}")
            return {"CANCELLED"}
        except subprocess.TimeoutExpired:
            self.report({"ERROR"}, "fidelity timed out after 10 min")
            return {"CANCELLED"}
        if r.returncode != 0:
            STATUS["message"] = "fidelity exceeded threshold"
            self.report({"WARNING"}, (r.stderr or r.stdout or "fidelity failed")[-200:])
            _reveal_in_file_manager(out_dir / "report.json")
            return {"CANCELLED"}
        STATUS["message"] = f"fidelity ok → {out_dir.name}"
        self.report({"INFO"}, STATUS["message"])
        _reveal_in_file_manager(out_dir / "report.json")
        return {"FINISHED"}


# --- 5. open_share_link -----------------------------------------------------


class SPLATFORGE_OT_open_share_link(Operator):
    """Open the last successful cloud job's share URL in the system browser."""

    bl_idname = "splatforge.open_share_link"
    bl_label = "Open Share Link"
    bl_description = "Open the SplatForge Cloud job page in your browser"

    @classmethod
    def poll(cls, _context):
        return bool(STATUS.get("last_share_url") or STATUS.get("last_output"))

    def execute(self, _context):
        url = STATUS.get("last_share_url") or STATUS.get("last_output") or ""
        if not url:
            self.report({"WARNING"}, "no share URL available yet")
            return {"CANCELLED"}
        try:
            webbrowser.open(str(url))
        except webbrowser.Error as e:
            self.report({"ERROR"}, f"could not open browser: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


# --- registration -----------------------------------------------------------


_CLASSES = (
    SPLATFORGE_OT_import_splat,
    SPLATFORGE_OT_optimize_local,
    SPLATFORGE_OT_submit_to_cloud,
    SPLATFORGE_OT_fetch_fidelity,
    SPLATFORGE_OT_open_share_link,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
