#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
//
// Bench driver. Compiles `bench/compute-decode.bench.ts` (+ real-scene.bench.ts),
// starts a tiny static server, launches a headless Chromium via playwright-core
// with WebGPU enabled, navigates to the bench page, polls `window.__bench`, and
// prints the JSON to stdout.
//
// If SF_BENCH_PLY_DIR is set, ALSO runs real-scene.html on the discovered
// scene set and merges the result under `realScene` in results.json.
//
// Run: `pnpm --filter @splatforge/viewer run bench`
//
// Output: a JSON document with per-scale decode/sort/total frame timings.
// Writes the same JSON to `bench/results.json`.

import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const distBench = join(root, 'dist-bench');

await mkdir(distBench, { recursive: true });

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

// Discover real-scene scenes.
const sceneDir = process.env.SF_BENCH_PLY_DIR ?? '';
let sceneList = [];
if (sceneDir) {
  try {
    const entries = await readdir(sceneDir);
    const bins = entries.filter((e) => e.endsWith('.bin'));
    for (const b of bins) {
      const base = b.slice(0, -4);
      if (entries.includes(`${base}.meta.json`)) sceneList.push(base);
    }
    console.error(`bench: found ${sceneList.length} real scene(s) in ${sceneDir}`);
  } catch (err) {
    console.error(`bench: SF_BENCH_PLY_DIR unreadable: ${err.message}`);
  }
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
    if (p === '/scenes/index.json') {
      const data = Buffer.from(JSON.stringify(sceneList));
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(data); return;
    }
    if (p.startsWith('/scenes/')) {
      if (!sceneDir) { res.writeHead(404); res.end('SF_BENCH_PLY_DIR not set'); return; }
      abs = join(sceneDir, p.slice('/scenes/'.length));
    } else if (p.startsWith('/dist-bench/')) {
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

const platformArgs =
  process.platform === 'darwin'
    ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer', '--use-angle=metal']
    : ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer', '--use-vulkan=swiftshader'];
const browser = await chromium.launch({
  headless: process.env.BENCH_HEADED ? false : true,
  args: [...platformArgs, '--ignore-gpu-blocklist', '--no-sandbox'],
});
const skipSynth = process.env.SF_SKIP_SYNTH === '1';
let result = {};
try {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  page.on('console', (msg) => process.stderr.write(`[page] ${msg.text()}\n`));
  if (!skipSynth) {
    await page.goto(`http://127.0.0.1:${PORT}/index.html`);
    const synth = await page.waitForFunction(
      () => /** @type {any} */ (globalThis).__bench && (globalThis.__bench.results || globalThis.__bench.error),
      null,
      { timeout: 480_000 },
    ).then((h) => h.jsonValue());
    Object.assign(result, synth);
  }
  if (sceneList.length > 0) {
    const page2 = await ctx.newPage();
    page2.on('console', (msg) => process.stderr.write(`[page-real] ${msg.text()}\n`));
    await page2.goto(`http://127.0.0.1:${PORT}/real-scene.html`);
    const real = await page2.waitForFunction(
      () => /** @type {any} */ (globalThis).__benchRealScene && (globalThis.__benchRealScene.results || globalThis.__benchRealScene.error),
      null,
      { timeout: 900_000 },
    ).then((h) => h.jsonValue());
    result.realScene = real;
  }
} finally {
  await browser.close();
  server.close();
}

const outPath = join(here, 'results.json');
await writeFile(outPath, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
console.error(`bench: wrote ${outPath}`);
if (result.error) process.exit(3);
