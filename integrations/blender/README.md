# SplatForge for Blender

Optimize Gaussian-Splat assets (`.ply` / `.spz` / `.glb`) for a target device
straight from Blender. Wraps the production `splatforge` CLI and the
SplatForge Cloud API in a clean N-panel.

KIRI Engine ships a Blender add-on that takes you from photos to a textured
mesh. This add-on covers the half KIRI does not: take a Gaussian splat and
turn it into a shippable web/mobile/VR asset.

| Capability | Local CLI | Cloud (managed compute) |
|---|---|---|
| `.ply` / `.spz` / `.glb` import | yes | n/a |
| `web-mobile` / `web-desktop` / `quest-browser` / `visionos` / `size-min` / `quality-max` presets | yes | yes |
| Optimize | yes (in-process CLI) | yes (uploads → polls) |
| Fidelity report (8-orbit ΔE94 / SSIM) | yes | n/a yet |
| Output share URL | n/a | yes |

## Requirements

* Blender **4.2 LTS** (or newer LTS — the add-on does not use experimental APIs).
* `splatforge` CLI installed somewhere on disk. The add-on auto-detects it
  on PATH and at well-known install dirs (homebrew, `~/.cargo/bin`,
  `~/.local/bin`, `%LOCALAPPDATA%\Programs\splatforge`, `C:\ProgramData\chocolatey\bin`).
  If detection fails you can paste an absolute path in the add-on preferences.

  Install the CLI:

  ```sh
  # macOS — Apple-silicon or Intel homebrew
  brew install splatforge

  # Linux / macOS via cargo
  cargo install splatforge-cli

  # Windows — grab splatforge.exe from
  # https://github.com/montabano1/SplatForge/releases and put it on PATH
  ```

## Install

1. Download `splatforge_addon-<version>.zip` from
   [GitHub Releases](https://github.com/montabano1/SplatForge/releases)
   (or build it locally — see *Build the zip* below).
2. In Blender: **Edit ▸ Preferences ▸ Add-ons ▸ Install…** and pick the .zip.
3. Tick the **SplatForge** entry to enable it.
4. Expand the add-on row to set your **API key** (optional — only needed for
   *Submit to Cloud*) and confirm the **splatforge CLI** field shows a
   green "Detected" line. If it does not, click the magnifier icon to
   re-probe, or paste an absolute path.

## Where to find it

In any 3D viewport press **N** to open the sidebar, then click the
**SplatForge** tab. You will see:

```
┌─ SplatForge ─────────────────────────────┐
│ ✓ splatforge 0.1.2 (PATH)                │
│ [ Import Splat (.ply / .spz / .glb) ]    │
│ Source: bonsai.ply                       │
│ Preset: [ web-mobile          ▼ ]        │
│ [    Optimize    ] [ Submit to Cloud ]   │
│ ☐ zstd-compress buffers                  │
│ Label:                                   │
│ ┌────────────────────────────────────┐   │
│ │ optimize: encoding-gltf            │   │
│ │ ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░ 47%      │   │
│ └────────────────────────────────────┘   │
│ Output: bonsai.web-mobile.glb            │
│ [ Fidelity Report ] [ Open Share Link ]  │
│ ▸ Cloud Job                              │
└──────────────────────────────────────────┘
```

## Workflow

1. **Import Splat** — file-select your `.ply` / `.spz` / `.glb`. Non-glb
   files are auto-converted via `splatforge convert` so Blender's stock
   glTF importer can place them in the outliner. The source path is
   stamped on each imported object's custom properties.
2. **Pick a preset** — `web-mobile` is the default; pick `quest-browser`
   for in-headset WebGL, `size-min` to ruthlessly cut bytes, or
   `quality-max` for the lossless repack.
3. **Optimize** — runs `splatforge optimize` and streams progress into the
   panel's progress bar. Output lands next to the source as
   `<name>.<preset>.glb`. With "open output folder after optimize"
   enabled (default) your file manager reveals it.
4. **Submit to Cloud** — same preset, but on managed compute. Uses your
   `SPLATFORGE_API_KEY` (from the add-on prefs, env var, or
   `~/.config/splatforge/api_key`). The poller runs on a worker thread
   so the Blender UI stays responsive — submit a 200 MB scene and keep
   modelling. Result is a shareable URL.
5. **Fidelity Report** — diffs the optimized output against the source
   via the deterministic 8-orbit ΔE94 / SSIM / pixelmatch harness.
   Writes `report.json` and frame PNGs to
   `<name>.<preset>-fidelity/` and opens the folder.
6. **Open Share Link** — opens the most recent cloud job page in your
   default browser.

## API key handling

This was called out in the brief as the biggest UX risk. The reality:

* Blender users almost never have `SPLATFORGE_API_KEY` set in their shell
  environment — Blender often launches from a GUI shortcut that inherits
  nothing.
* So the add-on offers a **password-masked text field** in
  *Edit ▸ Preferences ▸ Add-ons ▸ SplatForge*. On save it is:
  * persisted in Blender's per-user `userpref.blend` (chmod 600 on POSIX
    by Blender itself),
  * **and** mirrored to `~/.config/splatforge/api_key` (`%APPDATA%\SplatForge\api_key`
    on Windows) with chmod 600 so the standalone CLI picks the same key
    up.
* At call time, the resolution order is: **prefs field → env var → on-disk
  config**. The key is injected into every CLI subprocess's environment so
  GUI-launched Blender still gets authenticated submits.

## Build the zip

```sh
./integrations/blender/dist/build.sh
# → writes integrations/blender/dist/splatforge_addon-<version>.zip
```

The script:

* extracts the version from `bl_info` in `__init__.py` (single source of
  truth — no duplication),
* stages a clean copy that excludes `__pycache__`, `.DS_Store`, etc.,
* zips it with the `splatforge_addon/` directory at the zip root (the
  layout Blender 4.2's add-on installer expects),
* verifies the layout by inspecting the zip contents post-build.

CI runs this on every push to `integrations/blender/**` (see
`.github/workflows/blender-addon.yml`) and attaches the zip to GitHub
release events automatically.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `splatforge CLI not found` red banner | Click the magnifier icon to re-probe, or paste an absolute path in *Add-on Preferences ▸ splatforge CLI*. If you installed via cargo, the path is usually `~/.cargo/bin/splatforge`. |
| `splatforge CLI not found` on Windows | Download `splatforge.exe` from Releases, put it next to your project or somewhere on `PATH`, then re-probe. |
| Cloud submit returns `HTTP 401` | Open *Add-on Preferences ▸ API Key* and paste your key. The error message will surface in the panel's status line. |
| Optimize freezes the UI | It shouldn't — the operator is modal. If it does, capture Blender's stderr (`blender --debug`) and file an issue. |
| Fidelity errors with `node not found` | The fidelity harness shells out to Node.js 20+ for the deterministic rendering step. Install Node and ensure it's on `PATH`. |
| Re-importing a `.ply` looks identical | The first import wrote a `.glb` sibling for Blender to load; subsequent imports reuse it. Delete the sibling to force re-conversion. |

## Sample scene

`SAMPLE.blend` ships with a placeholder splat object that has the
`splatforge_source` / `splatforge_glb` custom properties set to the
bonsai sample bundled with the SplatBench corpus. Open it, switch to the
*SplatForge* sidebar tab, and click **Optimize** — you get a feel for
the full pipeline without having to drag a file in first.
