// Regenerates fixtures/bonsai-tiny.qat.ply from the synthetic reference encoder.
// Self-contained: bundles buildFixture.ts via esbuild and writes the bytes.
//
// Run from any one of the plugin package roots so node_modules/esbuild is
// resolvable, e.g.:
//   cd apps/codec/threejs-qat
//   node ../fixtures/generate-bonsai-tiny.mjs

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";
import * as esbuild from "esbuild";

const __dirname = dirname(fileURLToPath(import.meta.url));
// __dirname is .../apps/codec/fixtures regardless of cwd, because import.meta.url
// is fixed to this script's source path.
const tsPath = resolvePath(__dirname, "buildFixture.ts");

const result = await esbuild.build({
  entryPoints: [tsPath],
  bundle: true,
  format: "esm",
  platform: "node",
  write: false,
  target: "es2020",
  absWorkingDir: __dirname,
});

const code = result.outputFiles[0].text;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
const mod = await import(moduleUrl);

const fx = mod.buildSyntheticFixture(256);
const outPath = resolvePath(__dirname, "bonsai-tiny.qat.ply");
writeFileSync(outPath, fx.bytes);
console.log(`wrote ${outPath} (${fx.bytes.byteLength} bytes, ${fx.N} anchors)`);
