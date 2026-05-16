import { build } from "esbuild";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

const common = {
  entryPoints: ["src/index.ts"],
  bundle: true,
  platform: "neutral",
  target: "es2020",
  external: ["cesium"],
  sourcemap: true,
  logLevel: "info",
};

for (const out of ["dist/esm/index.js", "dist/cjs/index.cjs"]) {
  mkdirSync(dirname(out), { recursive: true });
}

await build({ ...common, format: "esm", outfile: "dist/esm/index.js" });
await build({ ...common, format: "cjs", outfile: "dist/cjs/index.cjs" });
