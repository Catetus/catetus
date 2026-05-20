"""Locate the user's ``catetus`` CLI binary.

We deliberately do NOT bundle the CLI — it ships ~30 MB of Rust and would
need triplet-specific builds for every Blender host. Instead, on first
panel render (and on the explicit "Detect CLI" button) we probe the usual
suspects, in priority order:

  1. The path stored in add-on preferences, if non-empty AND executable.
  2. ``$CATETUS_CLI`` env var.
  3. ``shutil.which("catetus")`` — covers any binary on ``$PATH`` already.
     This is what 95% of users (homebrew, cargo-install, manual /usr/local
     install) will hit.
  4. Platform-specific well-known install dirs:
       - macOS:  ``/opt/homebrew/bin/catetus`` (Apple Silicon brew)
                 ``/usr/local/bin/catetus``    (Intel brew, manual)
                 ``~/.cargo/bin/catetus``      (cargo install)
       - Linux:  ``/usr/local/bin/catetus``
                 ``~/.cargo/bin/catetus``
                 ``~/.local/bin/catetus``
       - Windows: ``%USERPROFILE%\\.cargo\\bin\\catetus.exe``
                  ``%LOCALAPPDATA%\\Programs\\catetus\\catetus.exe``
                  ``C:\\ProgramData\\chocolatey\\bin\\catetus.exe``

If the CLI is not found, the panel renders an install hint that maps the
detected platform to the easiest install path (brew on macOS, cargo on
Linux, the GitHub Releases zip on Windows).

Returns ``None`` when nothing usable was found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CLI_NAME = "catetus.exe" if sys.platform.startswith("win") else "catetus"


@dataclass(frozen=True)
class CliInfo:
    """A resolved CLI binary location + version string."""

    path: str
    version: str  # e.g. "catetus 0.1.2" — the first line of `--version`
    source: str  # human-readable: "pref" / "env" / "PATH" / "homebrew" / etc.


def _platform_search_paths() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [
            Path("/opt/homebrew/bin") / CLI_NAME,
            Path("/usr/local/bin") / CLI_NAME,
            home / ".cargo" / "bin" / CLI_NAME,
        ]
    if sys.platform.startswith("linux"):
        return [
            Path("/usr/local/bin") / CLI_NAME,
            home / ".cargo" / "bin" / CLI_NAME,
            home / ".local" / "bin" / CLI_NAME,
        ]
    if sys.platform.startswith("win"):
        candidates = [
            home / ".cargo" / "bin" / CLI_NAME,
        ]
        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            candidates.append(Path(localappdata) / "Programs" / "catetus" / CLI_NAME)
        candidates.append(Path("C:/ProgramData/chocolatey/bin") / CLI_NAME)
        return candidates
    return []


def _probe(candidate: Optional[str], source: str) -> Optional[CliInfo]:
    """Confirm a candidate path is executable and report its version.

    A failed ``--version`` probe is treated as "not the right binary" —
    we'd rather fall through to the next candidate than surface a half-
    working CLI to the user.
    """

    if not candidate:
        return None
    p = Path(candidate)
    # `shutil.which` already checked the executable bit on POSIX, but for
    # the explicit-pref case the user might have pasted a directory or a
    # non-executable path. Guard both.
    if not p.is_file():
        return None
    if not os.access(p, os.X_OK):
        return None
    try:
        # 5-second timeout: a hung CLI here would freeze the Blender UI on
        # every panel poll. The real CLI prints version + returns in <50 ms.
        out = subprocess.run(
            [str(p), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    first_line = (out.stdout or out.stderr).strip().splitlines()[:1]
    version = first_line[0] if first_line else "catetus"
    return CliInfo(path=str(p), version=version, source=source)


def detect_cli(pref_path: Optional[str] = None) -> Optional[CliInfo]:
    """Locate the catetus CLI. Returns ``None`` if not found."""

    # 1. Explicit user preference wins.
    info = _probe(pref_path, "preference")
    if info is not None:
        return info

    # 2. Environment override — useful on CI / nightly-build machines.
    env_path = os.environ.get("CATETUS_CLI")
    info = _probe(env_path, "env:CATETUS_CLI")
    if info is not None:
        return info

    # 3. PATH lookup — catches homebrew, apt, cargo-install, manual installs.
    which = shutil.which(CLI_NAME)
    info = _probe(which, "PATH")
    if info is not None:
        return info

    # 4. Platform-specific well-known locations (in case Blender is launched
    # from a Finder/Explorer shortcut that does not inherit the shell PATH).
    for cand in _platform_search_paths():
        info = _probe(str(cand), f"well-known:{cand.parent}")
        if info is not None:
            return info

    return None


def install_hint() -> str:
    """A short single-line install hint for the current platform.

    Surfaced in the panel when ``detect_cli`` returns None — the most common
    failure mode (Blender launched from a GUI shortcut without ``~/.cargo/bin``
    on PATH) is fixed by either pasting the path into preferences or running
    one of these commands in a terminal.
    """

    if sys.platform == "darwin":
        return "brew install catetus   (or: cargo install catetus-cli)"
    if sys.platform.startswith("linux"):
        return "cargo install catetus-cli   (or: download the release tarball)"
    if sys.platform.startswith("win"):
        return (
            "Download catetus.exe from "
            "https://github.com/Catetus/catetus/releases and add it to PATH"
        )
    return "cargo install catetus-cli"
