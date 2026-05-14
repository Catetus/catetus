#!/usr/bin/env node
/**
 * SplatBench v0.1.1 fidelity runner.
 *
 * For each scene in the SplatBench corpus, optimize at three presets and
 * compare 8 deterministic orbit-8 frames between the baseline (lossless-repack)
 * and each non-baseline preset. Metrics reported per (scene, preset):
 *
 *   - `pixelMatch`  fraction of pixels above pngjs/pixelmatch threshold
 *   - `deltaE94`    mean CIE Delta-E 1994 across the canvas, scaled to 0..1
 *                   by dividing by 100 (PRD threshold reads as "3%" → 0.03)
 *   - `ssimLoss`    `1 - SSIM` so all metrics agree on "lower is better"
 *
 * Pass criteria: mean Delta-E94 (0..1) < 0.03 AND max Delta-E94 < 0.08.
 *
 * Output: `benches/reports/fidelity-v0.json` and per-scene PNG dumps under
 * `benches/reports/frames/`. The leaderboard regenerator (`splatbench-update.mjs`)
 * consumes the JSON.
 */
import { spawnSync } from 'node:child_process';
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, basename, dirname, join } from 'node:path';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';
import playwright from 'playwright-core';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
// Script lives at tests/visual/scripts/, repo root is three levels up.
const ROOT = resolve(__dirname, '..', '..', '..');

const CLI = resolve(ROOT, 'target/release/splatforge');
const VIEWER_DIST = resolve(ROOT, 'packages/viewer/dist');
const HARNESS_PAGE = resolve(ROOT, 'tests/visual/harness/page.html');
const SCENES_DIR = resolve('/tmp/sbench/scenes');
const REPORTS_DIR = resolve(ROOT, 'benches/reports');
const FRAMES_DIR = resolve(REPORTS_DIR, 'frames');

const BASELINE = 'lossless-repack';
const PRESETS = ['lossless-repack', 'web-mobile', 'size-min'];
const PASS_MEAN = 0.03;
const PASS_MAX = 0.08;

/** Stable scene ordering: real anchors first, then synthetics. */
const SCENE_ORDER = [
  { file: 'bonsai.ply', id: 'bonsai_mipnerf360_iter7k' },
  { file: 'bicycle.ply', id: 'bicycle_mipnerf360_iter7k' },
  { file: 'splatbench_product_proxy.ply', id: 'splatbench_product_proxy' },
  { file: 'splatbench_indoor_proxy.ply', id: 'splatbench_indoor_proxy' },
  { file: 'splatbench_floater_proxy.ply', id: 'splatbench_floater_proxy' },
  { file: 'splatbench_outdoor_proxy.ply', id: 'splatbench_outdoor_proxy' },
  { file: 'splatbench_dense_proxy.ply', id: 'splatbench_dense_proxy' },
];

/* -------------------------------------------------- color-difference math */

/** sRGB byte (0..255) → linear-light component. */
function srgbToLinear(c) {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/** D65 sRGB → CIE Lab (1976). */
function rgbToLab(r, g, b) {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);
  // Linear sRGB → XYZ (D65).
  const X = lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375;
  const Y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.072175;
  const Z = lr * 0.0193339 + lg * 0.119192 + lb * 0.9503041;
  // D65 reference white.
  const xn = X / 0.95047;
  const yn = Y / 1.0;
  const zn = Z / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27) * t / 116 + 16 / 116);
  const fx = f(xn);
  const fy = f(yn);
  const fz = f(zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/**
 * CIE Delta-E 1994 (graphic-arts weighting). Range typically 0..100; values
 * below ~2.3 are imperceptible to a trained observer, ~10 is a clear shift.
 */
function deltaE94(lab1, lab2) {
  const [L1, a1, b1] = lab1;
  const [L2, a2, b2] = lab2;
  const dL = L1 - L2;
  const C1 = Math.sqrt(a1 * a1 + b1 * b1);
  const C2 = Math.sqrt(a2 * a2 + b2 * b2);
  const dC = C1 - C2;
  const da = a1 - a2;
  const db = b1 - b2;
  const dH2 = Math.max(0, da * da + db * db - dC * dC);
  const kL = 1, K1 = 0.045, K2 = 0.015;
  const sL = 1, sC = 1 + K1 * C1, sH = 1 + K2 * C1;
  const term1 = (dL / (kL * sL)) ** 2;
  const term2 = (dC / sC) ** 2;
  const term3 = dH2 / (sH * sH);
  return Math.sqrt(term1 + term2 + term3);
}

/**
 * Per-pixel mean ΔE94 between two RGBA buffers of the same size. Pixels are
 * compared in sRGB byte space; ΔE94 is a perceptual lightness/chroma metric.
 */
function meanDeltaE94(a, b) {
  let sum = 0;
  const pixels = a.length / 4;
  for (let i = 0; i < a.length; i += 4) {
    const labA = rgbToLab(a[i], a[i + 1], a[i + 2]);
    const labB = rgbToLab(b[i], b[i + 1], b[i + 2]);
    sum += deltaE94(labA, labB);
  }
  return sum / pixels;
}

/**
 * Mean structural similarity over 8×8 blocks on the luma channel. Returns
 * SSIM in 0..1; higher is more similar. Implementation is naive (uniform
 * window, no Gaussian) but sufficient to catch structural changes the
 * pixel-only metric misses.
 */
function blockSSIM(a, b, width, height) {
  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  const luma = (buf, x, y) => {
    const i = (y * width + x) * 4;
    return 0.299 * buf[i] + 0.587 * buf[i + 1] + 0.114 * buf[i + 2];
  };
  const block = 8;
  let total = 0;
  let count = 0;
  for (let by = 0; by + block <= height; by += block) {
    for (let bx = 0; bx + block <= width; bx += block) {
      let muA = 0, muB = 0;
      for (let y = by; y < by + block; y++) {
        for (let x = bx; x < bx + block; x++) {
          muA += luma(a, x, y);
          muB += luma(b, x, y);
        }
      }
      const N = block * block;
      muA /= N; muB /= N;
      let varA = 0, varB = 0, cov = 0;
      for (let y = by; y < by + block; y++) {
        for (let x = bx; x < bx + block; x++) {
          const la = luma(a, x, y) - muA;
          const lb = luma(b, x, y) - muB;
          varA += la * la;
          varB += lb * lb;
          cov += la * lb;
        }
      }
      varA /= N; varB /= N; cov /= N;
      const num = (2 * muA * muB + C1) * (2 * cov + C2);
      const den = (muA * muA + muB * muB + C1) * (varA + varB + C2);
      total += num / den;
      count++;
    }
  }
  return count === 0 ? 1 : total / count;
}

/** Aggregate {max, mean, p95} over per-frame samples. */
function aggregate(samples) {
  if (samples.length === 0) return { max: 0, mean: 0, p95: 0 };
  const max = Math.max(...samples);
  const mean = samples.reduce((s, x) => s + x, 0) / samples.length;
  const sorted = [...samples].sort((x, y) => x - y);
  const idx = (sorted.length - 1) * 0.95;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const p95 = lo === hi ? sorted[lo] : sorted[lo] * (hi - idx) + sorted[hi] * (idx - lo);
  return { max, mean, p95 };
}

/* -------------------------------------------------- splatforge orchestration */

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  if (res.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} exited with status ${res.status}`);
  }
}

/**
 * Optimize `ply` at `preset` into `outDir`. The buffers/ directory is written
 * as a sibling of the .gltf.
 */
function optimize(ply, preset, outDir) {
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
  const gltf = join(outDir, 'scene.gltf');
  run(CLI, ['optimize', ply, '--preset', preset, '--out', gltf]);
  return gltf;
}

/* -------------------------------------------------- static server */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.gltf': 'model/gltf+json',
  '.bin': 'application/octet-stream',
};

function startServer({ scenesRoot, port }) {
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
        if (url === '/' || url === '/page.html') {
          return send(await readFile(HARNESS_PAGE), '.html');
        }
        if (url.startsWith('/viewer/')) {
          return send(
            await readFile(resolve(VIEWER_DIST, url.slice('/viewer/'.length))),
            '.' + url.split('.').pop(),
          );
        }
        if (url.startsWith('/scenes/')) {
          return send(
            await readFile(resolve(scenesRoot, url.slice('/scenes/'.length))),
            '.' + url.split('.').pop(),
          );
        }
        res.writeHead(404).end();
      } catch (err) {
        if (err && err.code === 'ENOENT') {
          res.writeHead(404).end();
        } else {
          res.writeHead(500).end(String(err));
        }
      }
    });
    server.listen(port, '127.0.0.1', () => resolveServer(server));
  });
}

/**
 * Render 8 orbit-8 frames for `gltfRelativePath` (under `/scenes/`) into
 * `framesOutDir`. Returns the list of PNG buffers (in-memory) so a caller can
 * compute metrics without re-reading disk.
 */
async function renderFrames(baseUrl, gltfRelativePath, framesOutDir, renderer) {
  // SBENCH_CHROME_FLAGS lets the hardware-accel rerun (apps/fidelity-gpu/run.py)
  // force ANGLE/Vulkan flags without the script having to know about Modal.
  const extraFlags = (process.env.SBENCH_CHROME_FLAGS || '')
    .split(/\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const headless = process.env.SBENCH_HEADLESS === '0' ? false : true;
  const browser = await playwright.chromium.launch({
    headless,
    args: ['--no-sandbox', ...extraFlags],
  });
  try {
    const ctx = await browser.newContext({ viewport: { width: 512, height: 512 } });
    const page = await ctx.newPage();
    const url = `${baseUrl}/page.html?src=${encodeURIComponent(`/scenes/${gltfRelativePath}`)}&renderer=${renderer}&seed=42`;
    await page.goto(url, { waitUntil: 'load' });
    // Big real scenes (3.6M splats) need a long budget on the CPU-rasterized
    // SwiftShader path. Override via SBENCH_RENDER_TIMEOUT_MS if needed.
    const renderTimeout = Number(process.env.SBENCH_RENDER_TIMEOUT_MS || 1_800_000);
    await page.waitForFunction(
      () => window.__sf && (window.__sf.ready === true || window.__sf.error !== null),
      null,
      { timeout: renderTimeout },
    );
    const result = await page.evaluate(() => {
      const sf = window.__sf;
      if (sf.error) throw new Error(`viewer error: ${sf.error.code} ${sf.error.message}`);
      return sf.frames.map((f) => ({ index: f.index, dataUrl: f.dataUrl }));
    });
    mkdirSync(framesOutDir, { recursive: true });
    const out = [];
    for (const f of result) {
      const png = Buffer.from(f.dataUrl.split('base64,')[1], 'base64');
      const filename = `${String(f.index).padStart(4, '0')}.png`;
      writeFileSync(resolve(framesOutDir, filename), png);
      out.push({ index: f.index, png });
    }
    return out;
  } finally {
    await browser.close();
  }
}

/* -------------------------------------------------- main */

async function main() {
  if (!existsSync(CLI)) {
    throw new Error(`splatforge binary not found at ${CLI}. Run cargo build --release first.`);
  }
  if (!existsSync(SCENES_DIR)) {
    throw new Error(`scenes directory missing: ${SCENES_DIR}`);
  }

  // Build a single staging area where every scene/preset is exposed under
  // /scenes/<scene_id>/<preset>/scene.gltf and /scenes/<scene_id>/<preset>/buffers/.
  const stage = mkdtempSync(join(tmpdir(), 'sbench-fidelity-'));
  console.error(`[fidelity] staging dir: ${stage}`);

  mkdirSync(FRAMES_DIR, { recursive: true });
  const port = 4500 + Math.floor(Math.random() * 500);
  const server = await startServer({ scenesRoot: stage, port });
  const baseUrl = `http://127.0.0.1:${port}`;

  const renderer = process.env.SBENCH_RENDERER || 'webgl2';
  const sceneFilter = process.env.SBENCH_SCENES
    ? new Set(process.env.SBENCH_SCENES.split(','))
    : null;

  // Load any prior partial run so we can resume after a crash without
  // re-rendering scenes that already completed.
  const outPath = resolve(REPORTS_DIR, 'fidelity-v0.json');
  let results;
  if (existsSync(outPath) && process.env.SBENCH_RESUME === '1') {
    try {
      results = JSON.parse(readFileSync(outPath, 'utf8'));
      console.error(`[fidelity] resuming from ${outPath} (${results.scenes.length} scenes carried over)`);
    } catch {
      results = null;
    }
  }
  if (!results) {
    results = {
      schema: 'splatforge.splatbench.fidelity/0.1',
      name: 'SplatBench v0 — fidelity',
      splatforgeVersion: '0.1.1',
      runDate: new Date().toISOString().slice(0, 10),
      renderer,
      frameSize: '512x512',
      cameraPath: 'orbit-8',
      baseline: BASELINE,
      metricThresholds: { meanDeltaE94: PASS_MEAN, maxDeltaE94: PASS_MAX },
      scenes: [],
      errors: [],
    };
  }
  if (!results.errors) results.errors = [];
  const doneIds = new Set(results.scenes.map((s) => s.id));
  const errorIds = new Set((results.errors || []).map((e) => e.id));

  /** Persist `results` so a later crash doesn't lose finished work. */
  function persist() {
    writeFileSync(outPath, JSON.stringify(results, null, 2) + '\n');
  }

  try {
    for (const scene of SCENE_ORDER) {
      if (sceneFilter && !sceneFilter.has(scene.id)) continue;
      if (doneIds.has(scene.id)) {
        console.error(`[fidelity] === ${scene.id} (cached, skipping) ===`);
        continue;
      }
      const ply = resolve(SCENES_DIR, scene.file);
      if (!existsSync(ply)) {
        console.error(`[fidelity] missing ${ply}, skipping`);
        continue;
      }
      console.error(`\n[fidelity] === ${scene.id} ===`);

      try {
        await processScene(scene, ply, stage, baseUrl, renderer, results);
        // Drop any prior error entry for this scene on success.
        results.errors = results.errors.filter((e) => e.id !== scene.id);
        persist();
      } catch (err) {
        const msg = String(err && err.message ? err.message : err);
        console.error(`[fidelity]   FAILED: ${msg}`);
        if (!errorIds.has(scene.id)) results.errors.push({ id: scene.id, message: msg });
        errorIds.add(scene.id);
        persist();
        // Don't abort; move on to the next scene.
      }
    }
  } finally {
    server.close();
    rmSync(stage, { recursive: true, force: true });
  }

  persist();
  console.error(`\n[fidelity] wrote ${outPath}`);
}

/* Per-scene worker, extracted so the main loop can wrap it in try/catch. */
async function processScene(scene, ply, stage, baseUrl, renderer, results) {

      // Optimize all 3 presets up front so we can render each into the same
      // server view without re-staging mid-flight.
      const sceneStage = resolve(stage, scene.id);
      const presetGltfs = {};
      for (const preset of PRESETS) {
        const presetDir = resolve(sceneStage, preset);
        console.error(`[fidelity]   optimize ${preset}`);
        optimize(ply, preset, presetDir);
        presetGltfs[preset] = `${scene.id}/${preset}/scene.gltf`;
      }

      // Render each preset.
      const presetFrames = {};
      for (const preset of PRESETS) {
        const framesOut = resolve(FRAMES_DIR, scene.id, preset);
        console.error(`[fidelity]   render ${preset}`);
        presetFrames[preset] = await renderFrames(
          baseUrl,
          presetGltfs[preset],
          framesOut,
          renderer,
        );
      }

      // Compute metrics: each non-baseline preset vs baseline frame-by-frame.
      const baselineFrames = presetFrames[BASELINE];
      const sceneMetrics = { id: scene.id, presets: {} };
      for (const preset of PRESETS) {
        const frames = presetFrames[preset];
        const samples = { pixelMatch: [], deltaE94: [], ssimLoss: [] };
        for (let i = 0; i < Math.min(frames.length, baselineFrames.length); i++) {
          const a = PNG.sync.read(baselineFrames[i].png);
          const b = PNG.sync.read(frames[i].png);
          if (a.width !== b.width || a.height !== b.height) {
            samples.pixelMatch.push(1);
            samples.deltaE94.push(1);
            samples.ssimLoss.push(1);
            continue;
          }
          const diff = new PNG({ width: a.width, height: a.height });
          const changed = pixelmatch(a.data, b.data, diff.data, a.width, a.height, {
            threshold: 0.1,
            includeAA: false,
          });
          samples.pixelMatch.push(changed / (a.width * a.height));
          samples.deltaE94.push(meanDeltaE94(a.data, b.data) / 100);
          samples.ssimLoss.push(1 - blockSSIM(a.data, b.data, a.width, a.height));
        }
        const metrics = {
          pixelMatch: aggregate(samples.pixelMatch),
          deltaE94: aggregate(samples.deltaE94),
          ssimLoss: aggregate(samples.ssimLoss),
          perFrame: {
            pixelMatch: samples.pixelMatch,
            deltaE94: samples.deltaE94,
            ssimLoss: samples.ssimLoss,
          },
        };
        const passed =
          metrics.deltaE94.mean < PASS_MEAN && metrics.deltaE94.max < PASS_MAX;
        const status =
          preset === BASELINE
            ? 'baseline'
            : passed
              ? metrics.deltaE94.mean > PASS_MEAN * 0.66
                ? 'borderline'
                : 'pass'
              : 'fail';
        sceneMetrics.presets[preset] = { ...metrics, status, passed };
        console.error(
          `[fidelity]     ${preset}: ΔE94 mean=${metrics.deltaE94.mean.toFixed(4)} max=${metrics.deltaE94.max.toFixed(4)} → ${status}`,
        );
      }
      results.scenes.push(sceneMetrics);

      // Free the staged scene dir as soon as it's measured to keep disk usage low.
      rmSync(sceneStage, { recursive: true, force: true });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
