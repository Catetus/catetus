#!/usr/bin/env node
// copy-resource-assets.mjs
//
// Copies src/resources/assets/** into dist/resources/assets/** so the
// runtime asset-loader (which resolves relative to the compiled module URL)
// can find them after `tsc` has emitted dist/.
//
// Wire this into the npm package as a `postbuild` step:
//   "scripts": { "build": "tsc", "postbuild": "node scripts/copy-resource-assets.mjs" }
//
// Owned by: implementer C (resources subsystem). Implementer B's package.json
// should reference this script in the build pipeline.

import { cp, mkdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(__dirname, "..");

const SRC = join(pkgRoot, "src", "resources", "assets");
const DST = join(pkgRoot, "dist", "resources", "assets");

async function main() {
  if (!existsSync(SRC)) {
    console.error(`[copy-resource-assets] source directory missing: ${SRC}`);
    process.exit(1);
  }
  await mkdir(dirname(DST), { recursive: true });
  await cp(SRC, DST, { recursive: true, force: true });
  const stats = await stat(DST);
  console.log(`[copy-resource-assets] copied ${SRC} -> ${DST} (dir mtime ${stats.mtime.toISOString()})`);
}

main().catch((err) => {
  console.error("[copy-resource-assets] failed:", err);
  process.exit(1);
});
