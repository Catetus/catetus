#!/usr/bin/env node
// Render OG card HTML to PNG at 1200x630 via Playwright headless.
import playwright from 'playwright-core';
import { resolve } from 'node:path';

const OUT = '/Users/montabano1/Desktop/Catetus/apps/web/public/og-image.png';
const HTML = 'file:///tmp/og-card.html';

const browser = await playwright.chromium.launch({ headless: true });
const ctx = await browser.newContext({
  viewport: { width: 1200, height: 630 },
  deviceScaleFactor: 1,
});
const page = await ctx.newPage();
await page.goto(HTML, { waitUntil: 'networkidle' });
// Give web-fonts an extra tick to settle.
await page.waitForTimeout(800);
await page.screenshot({
  path: OUT,
  type: 'png',
  clip: { x: 0, y: 0, width: 1200, height: 630 },
  omitBackground: false,
});
await browser.close();
console.error('wrote', OUT);
