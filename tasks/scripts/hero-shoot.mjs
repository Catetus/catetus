#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// Screenshot the hero after the WebGPU fast-path viewer has stabilized.
// Reports observed FPS over a 3-second window post-firstRender.
import { chromium } from 'playwright-core';
import { writeFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

// Discover an installed Chromium binary on disk. playwright-core@1.60.0
// expects a specific build revision; we accept whichever full-Chromium
// dir is present (chromium-* over chromium_headless_shell-*) so we get
// real WebGPU support.
function findChrome() {
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || join(homedir(), '.cache', 'ms-playwright');
  if (!existsSync(base)) return undefined;
  const dirs = readdirSync(base).filter((d) => d.startsWith('chromium-'));
  const candidates = [
    ['chrome-mac-arm64', 'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing'],
    ['chrome-mac', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'],
    ['chrome-mac-arm64', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'],
  ];
  for (const d of dirs) {
    for (const tail of candidates) {
      const p = join(base, d, ...tail);
      if (existsSync(p)) return p;
    }
  }
  return undefined;
}

const URL = process.env.HERO_URL || 'http://localhost:4188/';
const OUT = process.env.HERO_OUT || 'tasks/screenshots/hero-fast-after.png';
const exe = findChrome();
console.log('chrome exe:', exe);

const browser = await chromium.launch({
  headless: true,
  executablePath: exe,
  // WebGPU on Chromium headless needs these — same flags Vercel's
  // screenshot harness uses.
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--use-vulkan=swiftshader',
    '--ignore-gpu-blocklist',
    '--disable-vulkan-fallback-to-gl-for-testing',
  ],
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
page.on('console', (m) => {
  const t = m.type();
  if (t === 'error' || t === 'warn' || t === 'warning' || t === 'log') console.log(`[browser:${t}]`, m.text());
});
page.on('pageerror', (e) => console.log('[browser:pageerror]', e.message));

console.log('navigating', URL);
await page.goto(URL, { waitUntil: 'networkidle', timeout: 30_000 });

// Wait until the viewer reports either "live" or "static fallback" or "offline".
const status = await page.waitForFunction(
  () => {
    const s = document.querySelector('[data-hero-status]');
    if (!s) return null;
    const t = s.textContent || '';
    if (/live|fallback|offline/.test(t)) return t;
    return null;
  },
  { timeout: 20_000 },
);
const statusText = await status.jsonValue();
console.log('viewer status:', statusText);

// Sample FPS for ~3s. We can't trust the page's rAF (Chromium
// throttles backgrounded/headless rAFs). Instead, sample the canvas
// every 100ms and check whether pixels changed — auto-orbit yaws the
// camera continuously, so a non-stalled renderer produces a different
// frame each tick.
const diag = await page.evaluate(async () => {
  const canvas = document.querySelector('[data-hero-canvas]');
  if (!canvas) return { fps: 0, sampledFrames: 0, info: 'no-canvas' };
  const hasWebGPU = typeof navigator.gpu !== 'undefined';
  let adapterInfo = null;
  if (hasWebGPU) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) {
        const info = a.info || (await a.requestAdapterInfo?.());
        adapterInfo = info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : 'no-info';
      } else {
        adapterInfo = 'no-adapter';
      }
    } catch (e) { adapterInfo = String(e); }
  }
  const w = 32, h = 32;
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const c2d = off.getContext('2d');
  function snapshot() {
    c2d.drawImage(canvas, 0, 0, w, h);
    const d = c2d.getImageData(0, 0, w, h).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 16) s = (s * 31 + d[i] + d[i + 1] * 3 + d[i + 2] * 5) & 0xffff;
    let lum = 0;
    for (let i = 0; i < d.length; i += 4) lum += (d[i] + d[i + 1] + d[i + 2]);
    return { hash: s, mean: lum / (w * h * 3) };
  }
  // Wait a beat so the viewer's first orbit frame is in.
  await new Promise((r) => setTimeout(r, 250));
  const samples = [];
  const t0 = performance.now();
  while (performance.now() - t0 < 3000) {
    samples.push(snapshot());
    await new Promise((r) => setTimeout(r, 100));
  }
  let distinct = 1;
  for (let i = 1; i < samples.length; i++) if (samples[i].hash !== samples[i - 1].hash) distinct += 1;
  const meanLum = samples.reduce((a, s) => a + s.mean, 0) / samples.length;
  // distinct-snapshot rate is a lower bound on actual fps when the
  // viewer is auto-rotating (yaw changes 8 deg/s → every 100ms is a
  // ~0.8deg yaw, easily visible per-frame). 30 distinct in 3s ≈ 10 fps,
  // which is the headless-swiftshader floor we'd expect.
  return { hasWebGPU, adapterInfo, distinct, samples: samples.length, meanLum, info: 'ok' };
});
console.log('canvas diag:', diag);

// Give the auto-orbit a moment for the canvas to be visibly mid-orbit.
// Wait long enough that one full rotation has happened (45deg/s on slow,
// 8deg/s on fast — 4s gets us roughly mid-orbit either way).
await page.waitForTimeout(4000);

await page.screenshot({ path: OUT, fullPage: false });
console.log('wrote', OUT);
writeFileSync('tasks/screenshots/hero-fast-after.meta.json', JSON.stringify({ url: URL, status: statusText, diag }, null, 2));

await browser.close();
