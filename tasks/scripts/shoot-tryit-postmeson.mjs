import { chromium } from 'playwright';
const url = process.argv[2] || 'http://127.0.0.1:4321/';
const out = process.argv[3] || 'apps/web/screenshots/tryit-postmeson.png';
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();
await page.goto(url, { waitUntil: 'networkidle' });
await page.locator('#try').scrollIntoViewIfNeeded();
await page.waitForTimeout(300);
const region = page.locator('.region-idle');
await region.screenshot({ path: out });
// also dump the preset values
const opts = await page.$$eval('#tryit-preset-select option', (els) =>
  els.map((e) => ({ value: e.value, label: e.textContent }))
);
console.log('preset options:', JSON.stringify(opts, null, 2));
console.log('wrote', out);
await browser.close();
