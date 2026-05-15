#!/usr/bin/env bash
# Package the SplatForge add-on into a Blender-installable .zip.
#
# Blender 4.2 expects exactly one of these two layouts inside the zip:
#   (a) A single top-level directory whose name matches the package
#       (`splatforge_addon/`) containing an `__init__.py` with `bl_info`.
#   (b) A bare `__init__.py` at the zip root (single-file add-on).
#
# We use layout (a). Blender installs by copying the directory verbatim
# into `~/Library/Application Support/Blender/4.2/scripts/addons/` (macOS)
# or the platform equivalent.
#
# Usage:
#   ./integrations/blender/dist/build.sh            # writes ./dist/splatforge_addon-<version>.zip
#   OUT=/tmp/foo.zip ./build.sh                     # explicit output path
#
# Exit codes:
#   0  zip built successfully
#   1  pyc cleanup / packaging failed
#   2  Python or zip binary missing
set -euo pipefail

# Resolve the repo-relative paths regardless of where the script is invoked from
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ADDON_DIR="$( cd -- "$SCRIPT_DIR/.." && pwd )"
SRC_DIR="$ADDON_DIR/splatforge_addon"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: $SRC_DIR not found — run from a SplatForge checkout" >&2
    exit 2
fi

# Extract the version from bl_info — single source of truth, no duplication.
VERSION="$(
    python3 - <<PY
import ast, pathlib
src = pathlib.Path("$SRC_DIR/__init__.py").read_text()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for tgt in node.targets:
            if getattr(tgt, "id", "") == "bl_info":
                d = ast.literal_eval(node.value)
                print(".".join(str(x) for x in d["version"]))
                raise SystemExit
PY
)"

if [[ -z "$VERSION" ]]; then
    echo "ERROR: could not parse version from bl_info" >&2
    exit 1
fi

OUT="${OUT:-$SCRIPT_DIR/splatforge_addon-${VERSION}.zip}"
mkdir -p "$(dirname "$OUT")"

# Stage in a temp dir to avoid shipping accidental files (.DS_Store, __pycache__).
STAGE="$(mktemp -d -t splatforge-addon.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

# rsync excludes match the install rules Blender enforces — bytecode in the
# zip would just be deleted on first run and bloats the artifact.
rsync -a \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".DS_Store" \
    --exclude=".pytest_cache" \
    "$SRC_DIR/" "$STAGE/splatforge_addon/"

# Stamp the install-time README inside the package so a user who installs
# from the .zip can still read the docs from inside Blender's prefs panel.
if [[ -f "$ADDON_DIR/README.md" ]]; then
    cp "$ADDON_DIR/README.md" "$STAGE/splatforge_addon/README.md"
fi

# Remove any pre-existing target so `zip` does not silently append.
rm -f "$OUT"

(
    cd "$STAGE"
    # `zip -r -q` is universal; `python -m zipfile -c` is the fallback for
    # CI runners that ship without /usr/bin/zip (rare, but Alpine does).
    if command -v zip >/dev/null 2>&1; then
        zip -r -q "$OUT" splatforge_addon
    else
        python3 -m zipfile -c "$OUT" splatforge_addon
    fi
)

# Sanity: enforce layout (a) by inspecting the zip contents — guards against
# someone refactoring this script and accidentally shipping a flat zip.
python3 - "$OUT" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
assert any(n.startswith("splatforge_addon/__init__.py") for n in names), \
    f"zip missing splatforge_addon/__init__.py — got: {names[:5]}"
print(f"zip ok: {len(names)} entries")
PY

echo "wrote $OUT"
