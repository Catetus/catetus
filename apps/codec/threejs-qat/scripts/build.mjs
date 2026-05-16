// Build script: bundles src/index.ts to dist/esm/index.js (ESM) and
// dist/cjs/index.cjs (CJS). Three is left as an external peer dep.
import { build } from "esbuild";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

const common = {
  entryPoints: ["src/index.ts"],
  bundle: true,
  platform: "neutral",
  target: "es2020",
  external: ["three"],
  sourcemap: true,
  logLevel: "info",
};

for (const out of ["dist/esm/index.js", "dist/cjs/index.cjs"]) {
  mkdirSync(dirname(out), { recursive: true });
}

await build({ ...common, format: "esm", outfile: "dist/esm/index.js" });
await build({ ...common, format: "cjs", outfile: "dist/cjs/index.cjs" });
