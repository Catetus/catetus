// One-shot Playwright screenshot of /khr-conformance for repo proof.
// Run via:  npx -y playwright@1 screenshot ...  is unreliable for full-page,
// so this script uses the `playwright` Node bindings directly. We install
// the chromium binary on first run via `playwright install chromium`.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const url = process.argv[2];
const outPath = process.argv[3];
if (!url || !outPath) {
  console.error('usage: node screenshot-khr.mjs <url> <out.png>');
  process.exit(2);
}
mkdirSync(resolve(outPath, '..'), { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1800 } });
const page = await ctx.newPage();
await page.goto(url, { waitUntil: 'networkidle', timeout: 30_000 });
await page.waitForSelector('.khrc-table', { timeout: 10_000 });
await page.screenshot({ path: outPath, fullPage: true });
await browser.close();
console.error('wrote', outPath);
