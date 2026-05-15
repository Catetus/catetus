#!/usr/bin/env node
/**
 * Take Playwright screenshots of the Explore alpha pages.
 * Assumes the Astro preview server is running on http://127.0.0.1:4322.
 *
 * Outputs:
 *   apps/web/screenshots/explore.png        — registry index w/ filter chips
 *   apps/web/screenshots/explore-scene.png  — per-scene page (bonsai)
 */
import { chromium } from '/Users/montabano1/Desktop/SplatForge/.wt-explore/tests/visual/node_modules/playwright-core/index.mjs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const OUT_DIR = resolve(__dirname, '..', 'screenshots');
const BASE = process.env.SF_PREVIEW_URL ?? 'http://127.0.0.1:4322';

const targets = [
  { url: `${BASE}/explore`, out: 'explore.png', width: 1440, height: 1400, fullPage: true },
  { url: `${BASE}/explore/bonsai_mipnerf360_iter7k`, out: 'explore-scene.png', width: 1440, height: 1100, fullPage: true },
  { url: `${BASE}/explore/garden_mipnerf360_7k`, out: 'explore-scene-external.png', width: 1440, height: 1100, fullPage: true },
];

const browser = await chromium.launch({ headless: true });
try {
  for (const t of targets) {
    const ctx = await browser.newContext({
      viewport: { width: t.width, height: t.height },
      deviceScaleFactor: 1,
    });
    const page = await ctx.newPage();
    await page.goto(t.url, { waitUntil: 'networkidle', timeout: 30000 });
    // Give CSS+filters JS a tick to settle.
    await page.waitForTimeout(800);
    await page.screenshot({
      path: resolve(OUT_DIR, t.out),
      fullPage: t.fullPage,
    });
    console.error(`[screenshot] wrote ${t.out} from ${t.url}`);
    await ctx.close();
  }
} finally {
  await browser.close();
}
