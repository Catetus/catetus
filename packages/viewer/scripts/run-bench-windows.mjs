#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
//
// Windows-side bench driver — mirrors bench/run-bench.mjs but:
//   - calls tsc via `node node_modules/typescript/bin/tsc` (works on Windows
//     where the `tsc` shim is `tsc.CMD` and spawnSync without shell fails).
//   - launches playwright with `channel: 'chromium'` so we use full
//     chromium (with WebGPU/D3D12) rather than the headless-shell which
//     has a stripped renderer pipeline.
//   - captures the FULL `__bench` object (results + adapter + limits)
//     to results.json so we can verify the real GPU is in use.
//
// Run: node packages/viewer/scripts/run-bench-windows.mjs

import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const benchDir = join(root, 'bench');
const distBench = join(root, 'dist-bench');

await mkdir(distBench, { recursive: true });

const tscBinJs = join(root, 'node_modules', 'typescript', 'bin', 'tsc');
const tscCfgPath = join(root, 'tsconfig.bench.json');
const tscRes = spawnSync(process.execPath, [tscBinJs, '-p', tscCfgPath], {
  cwd: root,
  stdio: 'inherit',
});
if (tscRes.status !== 0) {
  console.error('bench: tsc failed');
  process.exit(tscRes.status ?? 1);
}

const PORT = Number(process.env.BENCH_PORT ?? 4318);
const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.json': 'application/json',
  '.wgsl': 'text/plain',
};
const server = createServer(async (req, res) => {
  try {
    let p = decodeURIComponent((req.url ?? '/').split('?')[0]);
    if (p === '/' || p === '') p = '/index.html';
    let abs;
    if (p.startsWith('/dist-bench/')) {
      abs = join(root, p);
    } else {
      abs = join(benchDir, p);
    }
    const ext = (abs.match(/\.[a-z]+$/)?.[0] ?? '').toLowerCase();
    const data = await readFile(abs);
    res.writeHead(200, {
      'Content-Type': MIME[ext] ?? 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(data);
  } catch (err) {
    res.writeHead(404);
    res.end(`not found: ${err.message}`);
  }
});
await new Promise((r) => server.listen(PORT, r));
console.error(`bench: serving on http://127.0.0.1:${PORT}/`);

const visualPwt = resolve(root, '..', '..', 'tests', 'visual', 'node_modules', 'playwright-core', 'index.js');
const candidates = ['playwright-core', pathToFileURL(visualPwt).href];
let chromium;
const errs = [];
for (const c of candidates) {
  try {
    const mod = await import(c);
    chromium = mod.chromium ?? mod.default?.chromium;
    if (chromium) break;
    errs.push(`${c}: imported but .chromium missing`);
  } catch (err) {
    errs.push(`${c}: ${err.message}`);
  }
}
if (!chromium) {
  console.error('bench: playwright-core unavailable. Tried:');
  for (const e of errs) console.error('  ' + e);
  server.close();
  process.exit(2);
}

const args = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--ignore-gpu-blocklist',
  '--no-sandbox',
  '--enable-webgpu-developer-features',
  '--use-webgpu-adapter=d3d12',
  '--enable-gpu-rasterization',
  '--use-gl=angle',
  '--use-angle=d3d11',
];
const browser = await chromium.launch({
  headless: process.env.BENCH_HEADED ? false : true,
  channel: 'chromium', // full chromium, not headless-shell
  args,
});

let result;
try {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  page.on('console', (msg) => process.stderr.write(`[page] ${msg.text()}\n`));
  await page.goto(`http://127.0.0.1:${PORT}/index.html`);
  // Wait for the full __bench object, not just .results.
  await page.waitForFunction(
    () => /** @type {any} */ (globalThis).__bench && ((globalThis.__bench.results && globalThis.__bench.results.length >= 2) || globalThis.__bench.error),
    null,
    { timeout: 600_000 },
  );
  result = await page.evaluate(() => /** @type {any} */ (globalThis).__bench);
} finally {
  await browser.close();
  server.close();
}

const outPath = join(benchDir, 'results.json');
await writeFile(outPath, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
console.error(`bench: wrote ${outPath}`);
if (result?.error) process.exit(3);
