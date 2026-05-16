#!/usr/bin/env node
/**
 * Screenshot every (scene, preset) GLB through the preview-hero viewer and
 * dump render.png next to the GLB. Honest stand-in for "what the customer
 * sees in their browser".
 *
 * Requires:
 *   - `pnpm build` already run (so apps/web/dist exists and preview-hero.html
 *     + the GLBs under public/bench-renders/<scene>/<preset>/scene.glb are
 *     served).
 *   - A local static server on http://127.0.0.1:4325 serving apps/web/dist.
 *
 * Output:
 *   apps/web/public/bench-renders/<scene>/<preset>/render.png
 *   apps/web/public/bench-renders/<scene>/<preset>/render.json   (byte size)
 *
 * Skips any (scene, preset) where scene.glb is missing.
 */
import { chromium } from 'playwright';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, statSync, writeFileSync, readdirSync, mkdirSync, copyFileSync } from 'node:fs';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..', '..', '..');   // repo root
const BENCH_PUB = resolve(__dirname, '..', 'public', 'bench-renders');
const BENCH_DIST = resolve(__dirname, '..', 'dist', 'bench-renders');
const BASE = process.env.SF_PREVIEW_URL ?? 'http://127.0.0.1:4325';
const FRAMING = process.env.SF_FRAMING ?? '0.55';
const VIEWPORT_W = 800;
const VIEWPORT_H = 600;
const DPR = 2;
// Hold the viewer for this long to let auto-orbit settle before snapping.
const SETTLE_MS = parseInt(process.env.SF_SETTLE_MS ?? '4500', 10);

function listScenePresetPairs() {
  if (!existsSync(BENCH_PUB)) return [];
  const out = [];
  for (const scene of readdirSync(BENCH_PUB)) {
    const sceneDir = resolve(BENCH_PUB, scene);
    if (!statSync(sceneDir).isDirectory()) continue;
    for (const preset of readdirSync(sceneDir)) {
      const presetDir = resolve(sceneDir, preset);
      if (!statSync(presetDir).isDirectory()) continue;
      const gltf = resolve(presetDir, 'scene.gltf');
      if (existsSync(gltf)) {
        out.push({ scene, preset, gltf, presetDir });
      }
    }
  }
  return out;
}

const pairs = listScenePresetPairs();
console.log(`Found ${pairs.length} (scene, preset) gltf manifests to screenshot.`);

// Ensure each scene's chunked output is mirrored under dist/ so the static
// server serves it. `astro build` only copies public/ files snapshotted at
// build time, so files added after the build aren't reflected automatically.
// scene.gltf references sidecar buffers under `buffers/`, so we copy the
// whole directory tree.
function copyDirRecursive(src, dest) {
  mkdirSync(dest, { recursive: true });
  for (const entry of readdirSync(src, { withFileTypes: true })) {
    const sp = resolve(src, entry.name);
    const dp = resolve(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(sp, dp);
    } else {
      copyFileSync(sp, dp);
    }
  }
}
for (const p of pairs) {
  const srcDir = resolve(BENCH_PUB, p.scene, p.preset);
  const distDir = resolve(BENCH_DIST, p.scene, p.preset);
  copyDirRecursive(srcDir, distDir);
}

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({
  viewport: { width: VIEWPORT_W, height: VIEWPORT_H },
  deviceScaleFactor: DPR,
});
const page = await ctx.newPage();

const results = [];
for (const p of pairs) {
  const src = `/bench-renders/${p.scene}/${p.preset}/scene.gltf`;
  const url = `${BASE}/preview-hero.html?src=${encodeURIComponent(src)}&framing=${FRAMING}`;
  const outPng = resolve(p.presetDir, 'render.png');
  const outJson = resolve(p.presetDir, 'render.json');
  const bytes = statSync(p.gltf).size;

  console.log(`[${p.scene}/${p.preset}] ${url}`);
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    // The viewer flips a status element to "live · auto-orbit" once the
    // splat scene is uploaded and orbiting. Wait for that — falls back to
    // a timeout-based settle if the element shape changed.
    try {
      await page.waitForFunction(
        () => {
          const el = document.querySelector('#state, #viewerState, .state, [data-state]');
          if (!el) return false;
          return /live|auto-orbit|ready|rendering/i.test(el.textContent || '');
        },
        { timeout: 25_000 },
      );
    } catch {
      console.log(`  (no state signal, continuing on settle-timer)`);
    }
    await page.waitForTimeout(SETTLE_MS);
    // Screenshot only the canvas, not the controls overlay.
    const canvas = page.locator('canvas').first();
    if (await canvas.count()) {
      await canvas.screenshot({ path: outPng });
    } else {
      await page.screenshot({ path: outPng, clip: { x: 0, y: 0, width: VIEWPORT_W, height: VIEWPORT_H } });
    }
    writeFileSync(outJson, JSON.stringify({ gltfBytes: bytes, framing: FRAMING, viewport: [VIEWPORT_W, VIEWPORT_H], dpr: DPR }, null, 2));
    results.push({ scene: p.scene, preset: p.preset, ok: true, bytes });
  } catch (e) {
    console.error(`  FAILED:`, e.message);
    results.push({ scene: p.scene, preset: p.preset, ok: false, error: e.message, bytes });
  }
}

await browser.close();
console.log(JSON.stringify(results, null, 2));
const failed = results.filter((r) => !r.ok);
process.exit(failed.length ? 1 : 0);
