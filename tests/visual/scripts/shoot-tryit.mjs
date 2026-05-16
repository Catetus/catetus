// Visual smoke for the TryIt preset picker. Usage:
//   node shoot-tryit.mjs [url] [out_path] [preset_value]
// `preset_value` is optional — when set, we drive the <select> to that
// option before screenshotting so the rendered blurb matches the
// chosen preset (default screenshot shows the first option's blurb).
import { chromium } from 'playwright-core';
const url = process.argv[2] || 'http://127.0.0.1:4321/';
const out = process.argv[3] || 'apps/web/screenshots/tryit-presets.png';
const preset = process.argv[4] || null;
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();
await page.goto(url, { waitUntil: 'networkidle' });
await page.locator('#try').scrollIntoViewIfNeeded();
if (preset) {
  await page.selectOption('#tryit-preset-select', preset);
  // The blurb is rerendered on `change`; give the listener a tick.
  await page.waitForTimeout(200);
}
await page.waitForTimeout(500);
const region = page.locator('.region-idle');
await region.screenshot({ path: out });
console.log('wrote', out, preset ? `(preset=${preset})` : '');
await browser.close();
