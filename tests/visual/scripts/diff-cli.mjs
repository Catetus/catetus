#!/usr/bin/env node
/**
 * `catetus diff` helper (SPEC-0009).
 *
 * The Rust CLI dispatches to this script. It is invoked as:
 *
 *   node diff-cli.mjs \
 *     --before <path> --after <path> \
 *     --out <dir> --threshold 0.03 \
 *     --cli <catetus binary path> \
 *     [--camera-path orbit-8] [--frame-size 512x512] [--renderer webgpu|webgl2]
 *
 * Flow:
 *   1. Validate inputs.
 *   2. If `before`/`after` is a `.ply` or `.spz`, shell out to `${cli}
 *      convert --to gltf` to produce a temp .gltf the viewer can load.
 *   3. Try to dynamic-import `playwright-core`. If it isn't installed, emit
 *      a *degraded* but well-formed `diff.json` + `diff.html` and exit 0
 *      (this is the expected behaviour in CI sandboxes without a browser).
 *   4. Otherwise launch headless Chromium, load the harness page for each
 *      asset, capture 8 deterministic orbit frames, and pixelmatch them
 *      frame-by-frame.
 *   5. Always emit `diff.json` and `diff.html` under `--out`.
 *
 * Exit code:
 *   0 - all good, or degraded path
 *   1 - threshold exceeded
 *   2 - usage / IO error
 */
import { parseArgs } from 'node:util';
import { spawnSync } from 'node:child_process';
import {
  mkdirSync,
  writeFileSync,
  readFileSync,
  existsSync,
  mkdtempSync,
  copyFileSync,
  readdirSync,
  statSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve, basename, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..');

/* ----------------------------------------------------------------- args */

function parseCliArgs(argv) {
  const { values } = parseArgs({
    args: argv,
    options: {
      before: { type: 'string' },
      after: { type: 'string' },
      out: { type: 'string', default: 'reports/diff' },
      threshold: { type: 'string', default: '0.03' },
      cli: { type: 'string', default: '' },
      'camera-path': { type: 'string', default: 'orbit-8' },
      'frame-size': { type: 'string', default: '512x512' },
      renderer: { type: 'string', default: 'webgpu' },
      help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: false,
    strict: true,
  });
  if (values.help) {
    printUsage();
    process.exit(0);
  }
  if (!values.before || !values.after) {
    printUsage();
    process.exit(2);
  }
  return {
    before: values.before,
    after: values.after,
    out: values.out,
    threshold: Number(values.threshold),
    cli: values.cli,
    cameraPath: values['camera-path'],
    frameSize: values['frame-size'],
    renderer: values.renderer,
  };
}

function printUsage() {
  process.stderr.write(
    'usage: diff-cli.mjs --before FILE --after FILE [--out DIR] [--threshold N]\n' +
      '                    [--cli BINARY] [--camera-path P] [--frame-size WxH] [--renderer R]\n',
  );
}

/* --------------------------------------------------------- diff math */

/**
 * Aggregate per-frame ratios into max/mean/p95. Pure function — exported so
 * the unit test under tests/visual/scripts/diff-cli.test.mjs can import it.
 *
 * @param {number[]} perFrame - per-frame pixel-diff ratios in 0..1
 * @returns {{ max:number, mean:number, p95:number, count:number }}
 */
export function aggregateMetrics(perFrame) {
  if (!perFrame || perFrame.length === 0) {
    return { max: 0, mean: 0, p95: 0, count: 0 };
  }
  const max = Math.max(...perFrame);
  const mean = perFrame.reduce((s, x) => s + x, 0) / perFrame.length;
  const sorted = [...perFrame].sort((a, b) => a - b);
  const i = (sorted.length - 1) * 0.95;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  const p95 = lo === hi ? sorted[lo] : sorted[lo] * (hi - i) + sorted[hi] * (i - lo);
  return { max, mean, p95, count: perFrame.length };
}

/**
 * Compare two PNG buffers and return the fraction of pixels that differ.
 * Lazy-imports pngjs+pixelmatch so the module can be loaded in environments
 * where they aren't installed.
 *
 * @returns {Promise<{ratio:number, diffPng:Buffer|null}>}
 */
export async function pixelDiffPair(beforeBuf, afterBuf) {
  let PNG, pixelmatch;
  try {
    ({ PNG } = await import('pngjs'));
    pixelmatch = (await import('pixelmatch')).default;
  } catch {
    return { ratio: 0, diffPng: null };
  }
  const a = PNG.sync.read(beforeBuf);
  const b = PNG.sync.read(afterBuf);
  if (a.width !== b.width || a.height !== b.height) {
    return { ratio: 1, diffPng: null };
  }
  const out = new PNG({ width: a.width, height: a.height });
  const changed = pixelmatch(a.data, b.data, out.data, a.width, a.height, {
    threshold: 0.1,
    includeAA: false,
  });
  return { ratio: changed / (a.width * a.height), diffPng: PNG.sync.write(out) };
}

/* --------------------------------------------------------- conversion */

/**
 * If `path` is a `.ply` or `.spz`, shell out to the catetus binary to
 * convert it to a `.gltf` in `tmpDir`. Otherwise return `path` unchanged.
 */
function ensureGltf(path, tmpDir, cli) {
  const ext = extname(path).toLowerCase();
  if (ext === '.gltf' || ext === '.glb') return path;
  if (ext !== '.ply' && ext !== '.spz') {
    throw new Error(`unsupported input extension: ${ext}`);
  }
  if (!cli) {
    throw new Error('cannot convert without a --cli catetus binary path');
  }
  const out = join(tmpDir, basename(path).replace(/\.(ply|spz)$/i, '.gltf'));
  const res = spawnSync(cli, ['convert', path, '--to', 'gltf', '--out', out], {
    stdio: 'inherit',
  });
  if (res.status !== 0) {
    throw new Error(`catetus convert failed for ${path} (exit ${res.status})`);
  }
  return out;
}

/* -------------------------------------------------------- report HTML */

/**
 * Render `diff.html`. Prefers `@catetus/report-ui` if available (built
 * output exists), otherwise falls back to a minimal-but-valid inline
 * template so the helper degrades gracefully.
 */
async function renderReportHtml(data) {
  // Try the published package first.
  try {
    const mod = await import('@catetus/report-ui');
    if (typeof mod.renderDiffReport === 'function') {
      return mod.renderDiffReport(data);
    }
  } catch {
    /* fall through */
  }
  // Try the relative dist path.
  const dist = resolve(ROOT, '../../packages/report-ui/dist/index.js');
  if (existsSync(dist)) {
    try {
      const mod = await import(dist);
      if (typeof mod.renderDiffReport === 'function') {
        return mod.renderDiffReport(data);
      }
    } catch {
      /* fall through */
    }
  }
  return inlineFallbackHtml(data);
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function inlineFallbackHtml(data) {
  const passed = data.metrics.mean <= data.threshold;
  const pct = (v) => `${(v * 100).toFixed(2)}%`;
  const status = data.status ?? (passed ? 'pass' : 'fail');
  const banner = data.status === 'degraded'
    ? `<p style="background:#facc1522;border:1px solid #facc15;padding:8px;border-radius:6px;">
         degraded mode: ${esc(data.reason || 'no frames captured')}
       </p>`
    : '';
  const frames = (data.frames || [])
    .map(
      (f) => `<details>
  <summary>Frame ${esc(String(f.index).padStart(4, '0'))} — diff ${esc(pct(f.diffRatio))}</summary>
</details>`,
    )
    .join('\n');
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Catetus diff — ${esc(data.asset || 'report')}</title>
<style>body{font-family:ui-monospace,Menlo,monospace;background:#0b0d10;color:#d7dde4;padding:24px}</style>
</head><body>
<h1>Catetus visual diff — ${esc(data.asset || 'report')} (${esc(status.toUpperCase())})</h1>
${banner}
<p>mean ${esc(pct(data.metrics.mean))} · max ${esc(pct(data.metrics.max))} · p95 ${esc(pct(data.metrics.p95))} · threshold ${esc(pct(data.threshold))}</p>
${frames}
</body></html>
`;
}

/* --------------------------------------------------------- capture */

/**
 * Capture the 8 orbit frames for a single asset. Requires playwright-core.
 * Returns `[{ index, png: Buffer }]`.
 */
async function captureFrames(playwright, baseUrl, srcQueryPath, renderer) {
  const browser = await playwright.chromium.launch({
    headless: true,
    args: ['--enable-unsafe-webgpu', '--use-vulkan=swiftshader', '--enable-features=Vulkan'],
  });
  try {
    const ctx = await browser.newContext({ viewport: { width: 512, height: 512 } });
    const page = await ctx.newPage();
    const url = `${baseUrl}/page.html?src=${encodeURIComponent(srcQueryPath)}&renderer=${renderer}&seed=42`;
    await page.goto(url, { waitUntil: 'load' });
    // Big real scenes (1M+ splats) need a long budget on the CPU-rasterized
    // SwiftShader path. Override via CATETUS_DIFF_TIMEOUT_MS if needed.
    // The previous 60_000 ms hardcoded value tripped at 60s on 1M+-splat
    // scenes (bonsai_iter7000, bicycle_iter7000) and produced no useful
    // metrics in the baseline-splat-transform experiment — see
    // experiments/baseline-splat-transform/STATUS.md "PIVOT" entry.
    // For 1M+-splat scenes that the browser path can't handle in any
    // reasonable time, point CATETUS_DIFF_HELPER at
    // experiments/w3-fidelity-harness/code/cpu-fidelity.mjs — that runs
    // in ~5s on 1M+ splats without Playwright/SwiftShader.
    const diffTimeout = Number(process.env.CATETUS_DIFF_TIMEOUT_MS || 600_000);
    await page.waitForFunction(
      () => {
        const sf = window.__sf;
        return sf && (sf.ready === true || sf.error !== null);
      },
      null,
      { timeout: diffTimeout },
    );
    const frames = await page.evaluate(() => {
      const sf = window.__sf;
      if (sf.error) throw new Error(`viewer error: ${sf.error.code} ${sf.error.message}`);
      return sf.frames.map((f) => ({ index: f.index, dataUrl: f.dataUrl }));
    });
    return frames.map((f) => ({
      index: f.index,
      png: Buffer.from(f.dataUrl.split('base64,')[1], 'base64'),
    }));
  } finally {
    await browser.close();
  }
}

/**
 * Start an in-process static file server. Identical routing to
 * harness/server.mjs but lifetime-controlled.
 */
async function startServer({ assetsRoot, port }) {
  const { createServer } = await import('node:http');
  const { readFile } = await import('node:fs/promises');
  const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.gltf': 'model/gltf+json',
    '.glb': 'model/gltf-binary',
    '.bin': 'application/octet-stream',
    '.ply': 'application/octet-stream',
    '.spz': 'application/octet-stream',
  };
  const VIEWER_DIST = resolve(ROOT, '../../packages/viewer/dist');
  const PAGE_HTML = resolve(ROOT, 'harness/page.html');
  return new Promise((resolveServer) => {
    const server = createServer(async (req, res) => {
      const url = (req.url || '/').split('?')[0];
      const send = (body, ext) => {
        res.writeHead(200, {
          'content-type': MIME[ext.toLowerCase()] ?? 'application/octet-stream',
          'cache-control': 'no-store',
          'access-control-allow-origin': '*',
        });
        res.end(body);
      };
      try {
        if (url === '/' || url === '/page.html') return send(await readFile(PAGE_HTML), '.html');
        if (url.startsWith('/viewer/')) {
          return send(
            await readFile(resolve(VIEWER_DIST, url.slice('/viewer/'.length))),
            extname(url),
          );
        }
        if (url.startsWith('/fixtures/')) {
          return send(
            await readFile(resolve(assetsRoot, url.slice('/fixtures/'.length))),
            extname(url),
          );
        }
        res.writeHead(404);
        res.end();
      } catch (err) {
        if (err && err.code === 'ENOENT') {
          res.writeHead(404);
          res.end();
          return;
        }
        res.writeHead(500);
        res.end(String(err));
      }
    });
    server.listen(port, '127.0.0.1', () => resolveServer(server));
  });
}

/* --------------------------------------------------------- main */

/**
 * Write the diff.json + diff.html pair. Centralized so every exit path
 * lands on valid output.
 */
async function emitReport(outDir, payload) {
  mkdirSync(outDir, { recursive: true });
  // Only embed `generatedAt` when explicitly requested — keeps default
  // output deterministic byte-for-byte.
  if (process.env.CATETUS_INCLUDE_TIMESTAMPS === '1') {
    payload.generatedAt = new Date().toISOString();
  }
  writeFileSync(resolve(outDir, 'diff.json'), JSON.stringify(payload, null, 2) + '\n');
  const html = await renderReportHtml(payload);
  writeFileSync(resolve(outDir, 'diff.html'), html);
}

async function main() {
  const opts = parseCliArgs(process.argv.slice(2));
  if (!existsSync(opts.before)) {
    console.error(`missing: ${opts.before}`);
    process.exit(2);
  }
  if (!existsSync(opts.after)) {
    console.error(`missing: ${opts.after}`);
    process.exit(2);
  }

  // Try playwright-core. If it isn't installed, emit a degraded report and
  // exit 0 — the diff command should still produce valid artifacts.
  let playwright = null;
  try {
    playwright = await import('playwright-core');
  } catch {
    /* degraded */
  }

  const basePayload = {
    schema: 'catetus.diff/1',
    asset: `${basename(opts.before)} -> ${basename(opts.after)}`,
    before: opts.before,
    after: opts.after,
    cameraPath: opts.cameraPath,
    frameSize: opts.frameSize,
    threshold: opts.threshold,
    renderer: opts.renderer,
  };

  if (!playwright) {
    await emitReport(opts.out, {
      ...basePayload,
      status: 'degraded',
      reason: 'playwright-core not installed; install it under tests/visual/ to enable rendering',
      passed: true,
      metrics: { max: 0, mean: 0, p95: 0 },
      perFrame: [],
      frames: [],
      frameCount: 0,
    });
    process.stderr.write(
      `[diff] degraded (no playwright-core) -> ${resolve(opts.out)}\n` +
        `[diff] install playwright-core under tests/visual to enable full rendering\n`,
    );
    process.exit(0);
  }

  // Full rendering path. Convert PLY/SPZ -> glTF first.
  const tmp = mkdtempSync(join(tmpdir(), 'catetus-diff-'));
  let beforeGltf, afterGltf;
  try {
    beforeGltf = ensureGltf(opts.before, tmp, opts.cli);
    afterGltf = ensureGltf(opts.after, tmp, opts.cli);
  } catch (err) {
    await emitReport(opts.out, {
      ...basePayload,
      status: 'error',
      reason: String(err && err.message ? err.message : err),
      passed: false,
      metrics: { max: 0, mean: 0, p95: 0 },
      perFrame: [],
      frames: [],
      frameCount: 0,
    });
    console.error(err);
    process.exit(2);
  }

  // Stage assets under a single fixtures root so the in-process server can
  // serve them as /fixtures/before/... and /fixtures/after/....
  mkdirSync(resolve(tmp, 'before'), { recursive: true });
  mkdirSync(resolve(tmp, 'after'), { recursive: true });
  copyFileSync(beforeGltf, resolve(tmp, 'before', basename(beforeGltf)));
  copyFileSync(afterGltf, resolve(tmp, 'after', basename(afterGltf)));
  for (const side of ['before', 'after']) {
    const gltfPath = side === 'before' ? beforeGltf : afterGltf;
    // Legacy single-sibling .bin (used by older fixtures).
    const bin = gltfPath.replace(/\.gltf$/i, '.bin');
    if (existsSync(bin)) copyFileSync(bin, resolve(tmp, side, basename(bin)));
    // catetus-emitted chunked layout: `<gltfDir>/buffers/chunk_XXXX.bin`.
    // Stage the whole directory so URIs like `buffers/chunk_0000.bin` resolve.
    const gltfDir = dirname(gltfPath);
    const buffersDir = resolve(gltfDir, 'buffers');
    if (existsSync(buffersDir) && statSync(buffersDir).isDirectory()) {
      const sideBuffers = resolve(tmp, side, 'buffers');
      mkdirSync(sideBuffers, { recursive: true });
      for (const entry of readdirSync(buffersDir)) {
        copyFileSync(resolve(buffersDir, entry), resolve(sideBuffers, entry));
      }
    }
  }

  const port = 4000 + Math.floor(Math.random() * 1000);
  const server = await startServer({ assetsRoot: tmp, port });
  try {
    const baseUrl = `http://127.0.0.1:${port}`;
    const beforeFrames = await captureFrames(
      playwright,
      baseUrl,
      `/fixtures/before/${basename(beforeGltf)}`,
      opts.renderer,
    );
    const afterFrames = await captureFrames(
      playwright,
      baseUrl,
      `/fixtures/after/${basename(afterGltf)}`,
      opts.renderer,
    );

    const outFrames = resolve(opts.out, 'frames');
    mkdirSync(resolve(outFrames, 'before'), { recursive: true });
    mkdirSync(resolve(outFrames, 'after'), { recursive: true });
    mkdirSync(resolve(outFrames, 'diff'), { recursive: true });

    const perFrame = [];
    const reportFrames = [];
    const N = Math.min(beforeFrames.length, afterFrames.length);
    for (let i = 0; i < N; i++) {
      const idx = String(i + 1).padStart(4, '0');
      writeFileSync(resolve(outFrames, 'before', `${idx}.png`), beforeFrames[i].png);
      writeFileSync(resolve(outFrames, 'after', `${idx}.png`), afterFrames[i].png);
      const { ratio, diffPng } = await pixelDiffPair(beforeFrames[i].png, afterFrames[i].png);
      if (diffPng) {
        writeFileSync(resolve(outFrames, 'diff', `${idx}.png`), diffPng);
      }
      perFrame.push(ratio);
      reportFrames.push({
        index: i + 1,
        beforePng: `data:image/png;base64,${beforeFrames[i].png.toString('base64')}`,
        afterPng: `data:image/png;base64,${afterFrames[i].png.toString('base64')}`,
        diffPng: diffPng ? `data:image/png;base64,${diffPng.toString('base64')}` : '',
        diffRatio: ratio,
      });
    }

    const metrics = aggregateMetrics(perFrame);
    const passed = metrics.max <= opts.threshold;
    await emitReport(opts.out, {
      ...basePayload,
      status: passed ? 'pass' : 'fail',
      passed,
      metrics: { max: metrics.max, mean: metrics.mean, p95: metrics.p95 },
      perFrame,
      frames: reportFrames,
      frameCount: N,
    });
    process.stderr.write(
      `[diff] ${passed ? 'PASS' : 'FAIL'} mean=${metrics.mean.toFixed(4)} max=${metrics.max.toFixed(4)} -> ${resolve(opts.out)}\n`,
    );
    process.exit(passed ? 0 : 1);
  } catch (err) {
    await emitReport(opts.out, {
      ...basePayload,
      status: 'error',
      reason: String(err && err.message ? err.message : err),
      passed: false,
      metrics: { max: 0, mean: 0, p95: 0 },
      perFrame: [],
      frames: [],
      frameCount: 0,
    });
    console.error(err);
    process.exit(1);
  } finally {
    server.close();
  }
}

// Only run main if this file is executed directly (not imported as a module
// by the unit test).
const invokedDirectly = (() => {
  try {
    return resolve(process.argv[1] || '') === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
})();
if (invokedDirectly) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
