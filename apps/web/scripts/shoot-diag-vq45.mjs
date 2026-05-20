// SPDX-License-Identifier: Apache-2.0
//
// One-shot Playwright screenshot of /diag-vq45.html for the compressed-format
// viewer-decoder smoke test. Waits for the page to set
// `<body data-diag-state="done">` so the screenshot is taken after both
// canvases have rendered.
//
// Usage:  node scripts/shoot-diag-vq45.mjs http://localhost:4321/diag-vq45.html out.png

import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const url = process.argv[2];
const outPath = process.argv[3];
if (!url || !outPath) {
  console.error('usage: node shoot-diag-vq45.mjs <url> <out.png>');
  process.exit(2);
}
mkdirSync(resolve(outPath, '..'), { recursive: true });

const browser = await chromium.launch({
  args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1400 } });
const page = await ctx.newPage();
page.on('console', (msg) => {
  console.error(`[browser ${msg.type()}] ${msg.text()}`);
});
page.on('pageerror', (err) => console.error(`[pageerror] ${err.message}`));
const fullUrl = url + (url.includes('?') ? '&' : '?') + 'auto=1';
await page.goto(fullUrl, { waitUntil: 'load', timeout: 60_000 });
// Decoding 252 MiB of zstd in JS takes ~5s on M-series CPU; allow generous time.
await page
  .waitForSelector('body[data-diag-state="done"]', { timeout: 180_000 })
  .catch((err) => {
    console.error('timed out waiting for diag-state=done:', err.message);
  });
// Short settle so the canvas backing store is finalized before we sample it.
await page.waitForTimeout(500);
await page.screenshot({ path: outPath, fullPage: true, timeout: 120_000 });
await browser.close();
console.error('wrote', outPath);
