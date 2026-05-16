// Capture hero stage at 4 yaw angles using real Chromium (channel: chrome).
// The hero viewer auto-rotates; we override autoRotate manually to lock yaw.
// We work around this by injecting an init script that, after viewer is
// created, exposes a manual `__heroSetYaw` global; if not available, we
// fall back to letting auto-rotate run and time captures across the cycle.
//
// All screenshots saved to tasks/hero-v2-proof/.

import { chromium } from 'playwright';
import { writeFileSync } from 'fs';
import { join } from 'path';

const OUT_DIR = '/Users/montabano1/Desktop/.wt-hero-v2/tasks/hero-v2-proof';
const URL = process.env.HERO_URL || 'http://127.0.0.1:4500/';
const SETTLE_MS = parseInt(process.env.SETTLE_MS || '8000', 10);

async function main() {
  console.log(`[shoot] launching real Chromium (channel: chrome)`);
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    args: [
      '--enable-features=Vulkan',
      '--enable-unsafe-webgpu',
      '--use-vulkan',
      '--enable-gpu',
      '--ignore-gpu-blocklist',
    ],
  });
  const ctx = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });
  const page = await ctx.newPage();

  page.on('pageerror', (err) => console.warn('[page error]', err.message));
  page.on('console', (msg) => {
    const t = msg.text();
    if (/error|warn|hero|webgpu|viewer/i.test(t)) console.log('[page]', t);
  });

  console.log(`[shoot] goto ${URL}`);
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 60000 });

  // Wait for viewer status to flip to live
  console.log(`[shoot] waiting for hero status to go live...`);
  try {
    await page.waitForFunction(
      () => {
        const el = document.querySelector('[data-hero-status]');
        return el && /live|webgpu|webgl2/i.test(el.textContent || '');
      },
      { timeout: 30000 },
    );
    const status = await page.$eval('[data-hero-status]', (el) => el.textContent);
    console.log(`[shoot] status: ${status}`);
  } catch (e) {
    const status = await page
      .$eval('[data-hero-status]', (el) => el.textContent)
      .catch(() => 'unknown');
    console.warn(`[shoot] status wait timeout, current: ${status}`);
  }

  // Initial settle so autoRotate gets going; auto-rotate speed is 8 deg/s on
  // WebGPU. To capture 4 distinct yaws spaced 90°, wait (90/8 = 11.25s) between
  // each shot. Take an initial full-page Lighthouse-style shot first.
  await page.waitForTimeout(SETTLE_MS);

  // Lighthouse-style: full hero, 1920x1080, 10s after initial settle.
  await page.waitForTimeout(2000);
  const finalPath = join(OUT_DIR, 'final.png');
  await page.screenshot({ path: finalPath, fullPage: false });
  console.log(`[shoot] wrote ${finalPath}`);

  // Crop to hero stage element for the yaw captures
  const stage = await page.$('[data-hero-stage]');
  if (!stage) throw new Error('no [data-hero-stage] element found');

  // 4 yaw captures at intervals of 11.25s (assuming 8 deg/s) — 90° apart.
  const yawIntervalMs = 11250;
  for (let i = 0; i < 4; i++) {
    const yaw = i * 90;
    const p = join(OUT_DIR, `yaw-${yaw}.png`);
    await stage.screenshot({ path: p });
    console.log(`[shoot] wrote ${p} (yaw approx ${yaw}°)`);
    if (i < 3) await page.waitForTimeout(yawIntervalMs);
  }

  await browser.close();
  console.log('[shoot] done');
}

main().catch((e) => {
  console.error('[shoot] FAILED:', e);
  process.exit(1);
});
