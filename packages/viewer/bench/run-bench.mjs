#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
//
// Bench driver. Compiles `bench/compute-decode.bench.ts`, starts a tiny
// static server, launches a headless Chromium via playwright-core with
// WebGPU enabled, navigates to the bench page, polls `window.__bench`, and
// prints the JSON to stdout.
//
// Run: `pnpm --filter @splatforge/viewer run bench`
//
// Output: a JSON document with per-scale decode/sort/total frame timings.
// Writes the same JSON to `bench/results.json`.

import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const distBench = join(root, 'dist-bench');

await mkdir(distBench, { recursive: true });

// 1. Compile bench TS → dist-bench/*.js via tsc.
const tscBin = join(root, 'node_modules', '.bin', 'tsc');
const tscCfg = {
  compilerOptions: {
    target: 'ES2022',
    module: 'ESNext',
    moduleResolution: 'Bundler',
    lib: ['ES2022', 'DOM', 'DOM.Iterable'],
    types: ['@webgpu/types'],
    strict: true,
    outDir: 'dist-bench',
    rootDir: '.',
    esModuleInterop: true,
    skipLibCheck: true,
    resolveJsonModule: true,
    isolatedModules: true,
  },
  include: ['bench/**/*.ts', 'src/**/*.ts'],
  exclude: ['src/__tests__/**/*', 'src/streaming/**/*', 'node_modules', 'dist'],
};
await writeFile(join(root, 'tsconfig.bench.json'), JSON.stringify(tscCfg, null, 2));
const tscRes = spawnSync(tscBin, ['-p', join(root, 'tsconfig.bench.json')], {
  cwd: root,
  stdio: 'inherit',
});
if (tscRes.status !== 0) {
  console.error('bench: tsc failed');
  process.exit(tscRes.status ?? 1);
}

// 2. Tiny static server.
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
      abs = join(here, p);
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

// 3. Headless Chromium with WebGPU. playwright-core may live under the
// visual-tests workspace rather than the viewer package; we search both.
let chromium;
{
  const visualPwt = resolve(root, '..', '..', 'tests', 'visual', 'node_modules', 'playwright-core', 'index.js');
  const candidates = ['playwright-core', pathToFileURL(visualPwt).href];
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
}

const browser = await chromium.launch({
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--use-vulkan=swiftshader',
    '--ignore-gpu-blocklist',
    '--no-sandbox',
  ],
});
let result;
try {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  page.on('console', (msg) => process.stderr.write(`[page] ${msg.text()}\n`));
  await page.goto(`http://127.0.0.1:${PORT}/index.html`);
  result = await page.waitForFunction(
    () => /** @type {any} */ (globalThis).__bench && (globalThis.__bench.results || globalThis.__bench.error),
    null,
    { timeout: 480_000 },
  ).then((h) => h.jsonValue());
} finally {
  await browser.close();
  server.close();
}

const outPath = join(here, 'results.json');
await writeFile(outPath, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
console.error(`bench: wrote ${outPath}`);
if (result.error) process.exit(3);
