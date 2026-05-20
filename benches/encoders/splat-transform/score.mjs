#!/usr/bin/env node
/**
 * Fidelity scorer for splat-transform SOG output.
 *
 * Closes the honest-comparison gap on the splat-transform leaderboard column.
 * The bench harness today reports compression ratio only because SOG and
 * Catetus's glTF are different formats — and Catetus's viewer doesn't
 * speak SOG natively. This script decodes the SOG payload via the viewer's
 * `loader/sog` module, synthesizes a `KHR_gaussian_splatting` glTF that the
 * existing viewer pipeline consumes unchanged, then renders 8 canonical
 * orbit frames through Playwright and compares each frame against the
 * Catetus baseline render (lossless-repack of the same source PLY)
 * using the same ΔE94 / SSIM / pixelmatch math `tests/visual/scripts/splatbench-fidelity.mjs`
 * already uses for Catetus's own presets.
 *
 * CLI:
 *   node score.mjs --sog <output.sog> --ply <source.ply> --out <metrics.json> [opts]
 *
 * Options:
 *   --baseline <dir>      Directory of pre-rendered Catetus baseline PNGs (0000..0007.png).
 *                         If omitted, the script renders the baseline from <ply>.
 *   --frames-out <dir>    Where to dump per-frame PNGs (default: <out>.frames/).
 *   --keep-stage          Don't delete the synth glTF after scoring (useful for debug).
 *   --renderer webgl2|webgpu   Forced viewer backend. Default: webgl2 (deterministic).
 *
 * Determinism: same SOG bytes + same baseline PNGs + same renderer build →
 * same ΔE94 to the last LSB. We pin orbit-8 frames, 512×512 viewport, and
 * the source bbox (so a SOG that drops outliers doesn't accidentally reframe
 * the orbit and falsely "win" the comparison).
 */
import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createServer } from 'node:http';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// ESM-friendly absolute root path. Layout:
//   benches/encoders/splat-transform/score.mjs   <-- this file
//   benches/                                     <-- two levels up
//   .                                            <-- three levels up = repo root
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, '..', '..', '..');

/* --------------------------------------------------------------------- */
/* Dependency surface — fail loudly if the workspace hasn't been bootstrapped. */
/* --------------------------------------------------------------------- */

const VIEWER_DIST = resolve(ROOT, 'packages/viewer/dist');
const SOG_LOADER = resolve(VIEWER_DIST, 'loader/sog.js');
const HARNESS_PAGE = resolve(ROOT, 'tests/visual/harness/page.html');
const CATETUS_CLI = resolve(ROOT, 'target/release/catetus');

if (!existsSync(SOG_LOADER)) {
  console.error(`[score] missing ${SOG_LOADER}. Run \`pnpm -F @catetus/viewer build\` first.`);
  process.exit(2);
}
if (!existsSync(HARNESS_PAGE)) {
  console.error(`[score] missing harness page at ${HARNESS_PAGE}`);
  process.exit(2);
}

const { readSog, sogSceneToGltf } = await import(SOG_LOADER);

const sharpMod = await tryRequire('sharp');
if (!sharpMod) {
  console.error('[score] requires `sharp` for WebP decoding. Install via `pnpm -w add sharp`.');
  process.exit(2);
}
const playwrightMod = await tryRequire('playwright-core');
if (!playwrightMod) {
  console.error('[score] requires `playwright-core` from tests/visual.');
  process.exit(2);
}
const pngjsMod = await tryRequire('pngjs');
const pixelmatchMod = await tryRequire('pixelmatch');
if (!pngjsMod || !pixelmatchMod) {
  console.error('[score] requires `pngjs` + `pixelmatch` (live in tests/visual/node_modules).');
  process.exit(2);
}
const { PNG } = pngjsMod;
const pixelmatch = pixelmatchMod.default ?? pixelmatchMod;
const sharp = sharpMod.default ?? sharpMod;
const playwright = playwrightMod.default ?? playwrightMod;

/* --------------------------------------------------------------------- */
/* Color-difference math — kept byte-for-byte equivalent to               */
/* tests/visual/scripts/splatbench-fidelity.mjs so the splat-transform    */
/* numbers land on the same axis as the Catetus-preset numbers.        */
/* --------------------------------------------------------------------- */

function srgbToLinear(c) {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}
function rgbToLab(r, g, b) {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);
  const X = lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375;
  const Y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.072175;
  const Z = lr * 0.0193339 + lg * 0.119192 + lb * 0.9503041;
  const xn = X / 0.95047;
  const yn = Y / 1.0;
  const zn = Z / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t) / 116 + 16 / 116);
  const fx = f(xn);
  const fy = f(yn);
  const fz = f(zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
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
  const kL = 1,
    K1 = 0.045,
    K2 = 0.015;
  const sL = 1,
    sC = 1 + K1 * C1,
    sH = 1 + K2 * C1;
  const term1 = (dL / (kL * sL)) ** 2;
  const term2 = (dC / sC) ** 2;
  const term3 = dH2 / (sH * sH);
  return Math.sqrt(term1 + term2 + term3);
}
function meanDeltaE94(a, b) {
  let sum = 0;
  const pixels = a.length / 4;
  for (let i = 0; i < a.length; i += 4) {
    sum += deltaE94(rgbToLab(a[i], a[i + 1], a[i + 2]), rgbToLab(b[i], b[i + 1], b[i + 2]));
  }
  return sum / pixels;
}
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
      let muA = 0,
        muB = 0;
      for (let y = by; y < by + block; y++) {
        for (let x = bx; x < bx + block; x++) {
          muA += luma(a, x, y);
          muB += luma(b, x, y);
        }
      }
      const N = block * block;
      muA /= N;
      muB /= N;
      let varA = 0,
        varB = 0,
        cov = 0;
      for (let y = by; y < by + block; y++) {
        for (let x = bx; x < bx + block; x++) {
          const la = luma(a, x, y) - muA;
          const lb = luma(b, x, y) - muB;
          varA += la * la;
          varB += lb * lb;
          cov += la * lb;
        }
      }
      varA /= N;
      varB /= N;
      cov /= N;
      const num = (2 * muA * muB + C1) * (2 * cov + C2);
      const den = (muA * muA + muB * muB + C1) * (varA + varB + C2);
      total += num / den;
      count++;
    }
  }
  return count === 0 ? 1 : total / count;
}

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

/* --------------------------------------------------------------------- */
/* ZIP + WebP unpack                                                     */
/* --------------------------------------------------------------------- */

/**
 * Unzip a SOG archive into `outDir`. Uses the system `unzip` since SOGs
 * produced by splat-transform use the "store" method (no DEFLATE), so
 * pulling in a JS zip library would be overkill. Falls back to a hard
 * error if `unzip` is not on PATH — we'd rather break the bench loudly
 * than silently emit empty splats.
 */
function extractSog(sogPath, outDir) {
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
  const res = spawnSync('unzip', ['-o', '-q', sogPath, '-d', outDir]);
  if (res.status !== 0) {
    throw new Error(`unzip failed for ${sogPath}: ${res.stderr?.toString()}`);
  }
}

/**
 * `WebPDecoder` implementation that goes through `sharp`. Returned RGBA
 * matches the layout the SOG reader expects (premultiplied=false).
 */
function makeSharpDecoder() {
  return {
    async decode(bytes) {
      const img = sharp(bytes).ensureAlpha();
      const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
      return {
        rgba: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
        width: info.width,
        height: info.height,
      };
    },
  };
}

/* --------------------------------------------------------------------- */
/* Static server reused from splatbench-fidelity                          */
/* --------------------------------------------------------------------- */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.gltf': 'model/gltf+json',
  '.bin': 'application/octet-stream',
};

function startServer({ scenesRoot, port }) {
  return new Promise((res) => {
    const server = createServer(async (req, resp) => {
      const url = (req.url || '/').split('?')[0];
      const send = (body, ext) => {
        resp.writeHead(200, {
          'content-type': MIME[ext.toLowerCase()] ?? 'application/octet-stream',
          'cache-control': 'no-store',
          'access-control-allow-origin': '*',
        });
        resp.end(body);
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
        resp.writeHead(404).end();
      } catch (err) {
        if (err && err.code === 'ENOENT') resp.writeHead(404).end();
        else resp.writeHead(500).end(String(err));
      }
    });
    server.listen(port, '127.0.0.1', () => res(server));
  });
}

async function renderFrames(baseUrl, gltfRelativePath, framesOutDir, renderer, cameraBbox) {
  const browser = await playwright.chromium.launch({
    headless: true,
    args: ['--no-sandbox'],
  });
  try {
    const ctx = await browser.newContext({ viewport: { width: 512, height: 512 } });
    const page = await ctx.newPage();
    const bboxQuery = cameraBbox
      ? `&cameraBbox=${encodeURIComponent(JSON.stringify(cameraBbox))}`
      : '';
    const url = `${baseUrl}/page.html?src=${encodeURIComponent(
      `/scenes/${gltfRelativePath}`,
    )}&renderer=${renderer}&seed=42${bboxQuery}`;
    await page.goto(url, { waitUntil: 'load' });
    const renderTimeout = Number(process.env.SBENCH_RENDER_TIMEOUT_MS || 600_000);
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

/* --------------------------------------------------------------------- */
/* Baseline (Catetus lossless-repack) — only runs when --baseline is   */
/* not supplied; the bench harness pre-renders these once per scene to    */
/* keep per-encoder scoring cheap.                                        */
/* --------------------------------------------------------------------- */

function optimize(ply, preset, outDir) {
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
  const gltf = join(outDir, 'scene.gltf');
  const res = spawnSync(CATETUS_CLI, ['optimize', ply, '--preset', preset, '--out', gltf], {
    stdio: 'inherit',
  });
  if (res.status !== 0) throw new Error(`catetus optimize ${ply} exited ${res.status}`);
  return gltf;
}

function inputPlyBbox(ply) {
  const res = spawnSync(CATETUS_CLI, ['analyze', ply], { encoding: 'utf8' });
  if (res.status !== 0) return null;
  const j = JSON.parse(res.stdout);
  const bb = j && j.boundingBox;
  if (!bb || !Array.isArray(bb.min) || !Array.isArray(bb.max)) return null;
  return {
    min: [Number(bb.min[0]), Number(bb.min[1]), Number(bb.min[2])],
    max: [Number(bb.max[0]), Number(bb.max[1]), Number(bb.max[2])],
  };
}

/* --------------------------------------------------------------------- */
/* Main                                                                  */
/* --------------------------------------------------------------------- */

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--sog') out.sog = argv[++i];
    else if (a === '--ply') out.ply = argv[++i];
    else if (a === '--out') out.out = argv[++i];
    else if (a === '--baseline') out.baseline = argv[++i];
    else if (a === '--frames-out') out.framesOut = argv[++i];
    else if (a === '--renderer') out.renderer = argv[++i];
    else if (a === '--keep-stage') out.keepStage = true;
    else if (a === '--help' || a === '-h') {
      console.log(readFileSync(__filename, 'utf8').split('\n').slice(1, 30).join('\n'));
      process.exit(0);
    } else {
      console.error(`[score] unknown arg: ${a}`);
      process.exit(2);
    }
  }
  if (!out.sog || !out.ply || !out.out) {
    console.error('[score] required: --sog <path> --ply <path> --out <path>');
    process.exit(2);
  }
  out.renderer = out.renderer || 'webgl2';
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!existsSync(args.sog)) throw new Error(`sog not found: ${args.sog}`);
  if (!existsSync(args.ply)) throw new Error(`ply not found: ${args.ply}`);

  // Pin every render to the source-PLY bbox so a SOG that prunes outlier
  // geometry doesn't reframe the camera and accidentally "win" by orbiting
  // a tighter scene. Same rationale as splatbench-fidelity.mjs.
  let cameraBbox = inputPlyBbox(args.ply);
  if (!cameraBbox) {
    console.error('[score] could not compute source bbox; falling back to manifest bbox');
  }

  const stage = mkdtempSync(join(tmpdir(), 'sog-score-'));
  console.error(`[score] staging dir: ${stage}`);

  try {
    // 1. Decode SOG → synthetic glTF.
    const sogDir = join(stage, 'sog-extracted');
    extractSog(args.sog, sogDir);
    const decoder = makeSharpDecoder();
    const sceneFs = {
      async read(name) {
        return new Uint8Array(await readFile(join(sogDir, name)));
      },
    };
    const sogScene = await readSog(sceneFs, 'meta.json', decoder);
    console.error(`[score] decoded SOG: ${sogScene.splatCount} splats`);

    const sogSceneDir = join(stage, 'sog-scene');
    mkdirSync(sogSceneDir, { recursive: true });
    const { gltf, bin } = sogSceneToGltf(sogScene, 'scene.bin');
    writeFileSync(join(sogSceneDir, 'scene.gltf'), gltf);
    writeFileSync(join(sogSceneDir, 'scene.bin'), bin);

    // 2. Baseline render — reuse pre-rendered PNGs if supplied; otherwise run
    //    Catetus optimize → render.
    let baselineFrames;
    const port = 4500 + Math.floor(Math.random() * 500);
    const server = await startServer({ scenesRoot: stage, port });
    const baseUrl = `http://127.0.0.1:${port}`;

    try {
      if (args.baseline && existsSync(args.baseline)) {
        baselineFrames = readPngDir(args.baseline);
        console.error(`[score] using ${baselineFrames.length} pre-rendered baseline frames`);
      } else {
        if (!existsSync(CATETUS_CLI)) {
          throw new Error(
            `catetus CLI missing at ${CATETUS_CLI}; supply --baseline <dir> instead`,
          );
        }
        console.error('[score] rendering Catetus baseline (lossless-repack)…');
        const baselineDir = join(stage, 'baseline');
        optimize(args.ply, 'lossless-repack', baselineDir);
        // Move into the served scenes root.
        const baselineServe = join(stage, 'baseline-served');
        spawnSync('mv', [baselineDir, baselineServe], { stdio: 'inherit' });
        baselineFrames = await renderFrames(
          baseUrl,
          'baseline-served/scene.gltf',
          join(dirname(args.out), 'baseline-frames'),
          args.renderer,
          cameraBbox,
        );
      }

      // 3. Render SOG-synthesized glTF.
      const sogFrameDir = args.framesOut || `${args.out}.frames`;
      console.error('[score] rendering SOG-synthesized glTF…');
      const sogFrames = await renderFrames(
        baseUrl,
        'sog-scene/scene.gltf',
        sogFrameDir,
        args.renderer,
        cameraBbox,
      );

      // 4. Compute metrics.
      const samples = { pixelMatch: [], deltaE94: [], ssimLoss: [] };
      const n = Math.min(sogFrames.length, baselineFrames.length);
      for (let i = 0; i < n; i++) {
        const a = PNG.sync.read(baselineFrames[i].png);
        const b = PNG.sync.read(sogFrames[i].png);
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
      const out = {
        schema: 'catetus.splatbench.encoder-fidelity/0.1',
        sog: args.sog,
        ply: args.ply,
        renderer: args.renderer,
        frames: n,
        frameSize: '512x512',
        cameraPath: 'orbit-8',
        pixelMatch: aggregate(samples.pixelMatch),
        deltaE94: aggregate(samples.deltaE94),
        ssimLoss: aggregate(samples.ssimLoss),
        meanDeltaE94: aggregate(samples.deltaE94).mean,
        meanPixelMatch: aggregate(samples.pixelMatch).mean,
        meanSsimLoss: aggregate(samples.ssimLoss).mean,
        perFrame: samples,
      };
      mkdirSync(dirname(args.out), { recursive: true });
      writeFileSync(args.out, JSON.stringify(out, null, 2));
      console.error(
        `[score] wrote ${args.out} — ΔE94 mean=${out.deltaE94.mean.toFixed(4)} max=${out.deltaE94.max.toFixed(4)}`,
      );
    } finally {
      server.close();
    }
  } finally {
    if (!args.keepStage) rmSync(stage, { recursive: true, force: true });
  }
}

function readPngDir(dir) {
  const fs = require('node:fs');
  const files = fs
    .readdirSync(dir)
    .filter((f) => /^\d{4}\.png$/.test(f))
    .sort();
  return files.map((f, i) => ({ index: i + 1, png: fs.readFileSync(join(dir, f)) }));
}

/** Dynamic-import-or-resolve from one of the known node_modules roots. */
async function tryRequire(name) {
  const roots = [
    resolve(ROOT, 'tests/visual/node_modules'),
    resolve(ROOT, 'apps/web/node_modules'),
    resolve(ROOT, 'node_modules'),
  ];
  for (const r of roots) {
    const p = resolve(r, name);
    if (existsSync(p)) {
      try {
        return await import(p);
      } catch (e) {
        // ESM-import the package.json `main` resolution path:
        try {
          const pkg = JSON.parse(readFileSync(resolve(p, 'package.json'), 'utf8'));
          const main = pkg.module || pkg.main || 'index.js';
          return await import(resolve(p, main));
        } catch {
          throw e;
        }
      }
    }
  }
  return null;
}

main().catch((err) => {
  console.error('[score] failed:', err.stack || err.message || err);
  process.exit(1);
});
