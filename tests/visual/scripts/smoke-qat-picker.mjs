// Smoke test: the QAT-Scaffold preset entry is visible in the TryIt
// picker, selectable, and its blurb renders the headline numbers
// (+0.17 dB, 50%). Runs against a local astro preview server so it
// can ship in CI later without internet access.
import { chromium } from 'playwright-core';
import { strict as assert } from 'node:assert';

const url = process.argv[2] || 'http://127.0.0.1:4399/';
const outDir = process.argv[3] || '/tmp/qat-picker-smoke';
const fs = await import('node:fs/promises');
await fs.mkdir(outDir, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();
await page.goto(url, { waitUntil: 'networkidle' });
await page.locator('#try').scrollIntoViewIfNeeded();
await page.waitForTimeout(400);

// 1. The select contains the new option.
const optionLabel = await page.locator(
  '#tryit-preset-select option[value="catetus-qat-scaffold"]'
).textContent();
assert(optionLabel, 'QAT option missing from picker');
assert(
  optionLabel.includes('QAT-Scaffold'),
  `QAT label wrong: ${optionLabel}`
);

// 2. Select it; verify blurb updates with the headline.
await page.locator('#tryit-preset-select').selectOption('catetus-qat-scaffold');
await page.waitForTimeout(150);
const blurb = (await page.locator('[data-tryit-preset-blurb]').textContent()) ?? '';
assert(blurb.includes('37%'), `blurb missing 37%: ${blurb}`);
assert(blurb.includes('+0.17'), `blurb missing +0.17 dB: ${blurb}`);
assert(/lossless/i.test(blurb), `blurb missing lossless: ${blurb}`);

// 3. Screenshot for visual audit.
await page.locator('.region-idle').screenshot({ path: `${outDir}/tryit-qat-selected.png` });

// 4. /docs/api/qat-scaffold renders.
const docs = await page.goto(`${url.replace(/\/$/, '')}/docs/api/qat-scaffold`, {
  waitUntil: 'networkidle',
});
assert.equal(docs.status(), 200, `docs page status ${docs.status()}`);
const h1 = await page.locator('h1').first().textContent();
assert(/QAT-Scaffold/.test(h1 || ''), `docs h1 wrong: ${h1}`);
await page.screenshot({
  path: `${outDir}/docs-qat-scaffold.png`,
  fullPage: true,
});

await browser.close();
console.log('SMOKE OK — blurb:', blurb.trim());
console.log('screenshots:', outDir);
