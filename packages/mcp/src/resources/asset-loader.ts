// asset-loader.ts — load static resource assets from src/resources/assets/ at runtime.
//
// At build time (tsc), the assets/ directory is NOT compiled (it contains JSON/MD,
// not TS). The package.json `files` field includes `dist/`; we copy assets into
// dist/ during build via a postbuild step (see scripts/copy-assets.mjs). At runtime,
// we resolve assets relative to this module's URL.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolution order:
//   1. <this-file-dir>/assets/...                  (when running tsx from src/)
//   2. <this-file-dir>/../resources/assets/...     (when running from dist/)
// The src/ layout puts assets at ./assets relative to this file; tsc copies them
// alongside, so the same relative path works in both modes.
function resolveAsset(relPath: string): string {
  return join(__dirname, "assets", relPath);
}

export function loadTextAsset(relPath: string): string {
  return readFileSync(resolveAsset(relPath), "utf8");
}

export function loadJsonAsset<T = unknown>(relPath: string): T {
  return JSON.parse(loadTextAsset(relPath)) as T;
}

// Used by `resources/list` to set a `size` hint on the resource descriptor.
export function assetByteLength(relPath: string): number {
  return Buffer.byteLength(loadTextAsset(relPath), "utf8");
}
