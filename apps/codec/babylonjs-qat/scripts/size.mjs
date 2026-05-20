import { gzipSync } from "node:zlib";
import { readFileSync } from "node:fs";
for (const f of ["dist/esm/index.js", "dist/cjs/index.cjs"]) {
  try {
    const raw = readFileSync(f);
    const gz = gzipSync(raw);
    console.log(`${f}: raw=${raw.byteLength}B gzip=${gz.byteLength}B`);
  } catch (e) {
    console.log(`${f}: <missing>`);
  }
}
