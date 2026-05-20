// Visual smoke for the HAC++ bundle-upload UX. Verifies that selecting
// the hacpp-lzma preset:
//   1. Swaps the chip group from single-splat formats to archive formats.
//   2. Surfaces the bundle-help glyph next to the picker.
//   3. Rewrites the file input's accept= attribute.
//   4. Surfaces the bundle-specific error when a stray .ply is dropped.
// Also screenshots the new /docs/hacpp-bundle page so the layout
// reference is captured in the same run.
//
// Usage:
//   node tests/visual/scripts/shoot-tryit-hacpp-bundle.mjs [baseUrl] [outDir]
//
// Defaults to http://127.0.0.1:4321 and apps/web/screenshots/.
import { chromium } from 'playwright-core';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const baseUrl = process.argv[2] || 'http://127.0.0.1:4321';
const outDir = process.argv[3] || 'apps/web/screenshots';
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();

// State A: idle / default preset (web-mobile) — baseline.
await page.goto(baseUrl + '/', { waitUntil: 'networkidle' });
await page.locator('#try').scrollIntoViewIfNeeded();
await page.waitForTimeout(400);
await page.locator('.region-idle').screenshot({
  path: join(outDir, 'hacpp-bundle-flow-1-idle-default.png'),
});

// State B: hacpp-lzma selected — chip group + accept= should swap.
await page.selectOption('#tryit-preset-select', 'hacpp-lzma');
await page.waitForTimeout(250);
const acceptAttr = await page.locator('#splat-file-input').getAttribute('accept');
const chipMode = await page.locator('[data-tryit-format-chips]').getAttribute('data-mode');
const helpHidden = await page.locator('[data-tryit-bundle-help]').getAttribute('hidden');
console.log('accept=', acceptAttr);
console.log('chip mode=', chipMode);
console.log('help hidden=', helpHidden); // null when shown
if (acceptAttr !== '.tar,.tar.gz,.tgz') {
  throw new Error('expected bundle accept list, got ' + acceptAttr);
}
if (chipMode !== 'bundle') {
  throw new Error('expected chip mode="bundle", got ' + chipMode);
}
if (helpHidden !== null) {
  throw new Error('expected bundle-help to be visible (hidden=null), got ' + helpHidden);
}
await page.locator('.region-idle').screenshot({
  path: join(outDir, 'hacpp-bundle-flow-2-hacpp-selected.png'),
});

// State C: drop a wrong-format file (a .ply) while hacpp-lzma is active.
// The validator should switch to the error region with a bundle-specific
// hint. We use setInputFiles with a tiny in-memory buffer so we don't
// hit the network.
const stubBytes = Buffer.from('not really a ply', 'utf-8');
await page.setInputFiles('#splat-file-input', {
  name: 'wrong.ply',
  mimeType: 'application/octet-stream',
  buffer: stubBytes,
});
// Wait on the data-state attribute mutation rather than visibility:
// the `.region-error` block is gated by a CSS sibling rule, so
// Playwright's waitForSelector visibility heuristic occasionally
// resolves before the cascade applies. Polling the attribute is
// deterministic and matches what the state machine actually changes.
await page.waitForFunction(
  () => document.querySelector('.dropzone')?.getAttribute('data-state') === 'error',
  { timeout: 4000 },
);
await page.waitForTimeout(150); // give layout one beat to settle
const errBody = (await page.locator('[data-error-body]').textContent()) || '';
console.log('error body=', errBody);
if (!/hacpp-lzma/.test(errBody) || !/Scaffold-GS/.test(errBody)) {
  throw new Error('expected bundle-specific error hint, got: ' + errBody);
}
await page.locator('.region-error').screenshot({
  path: join(outDir, 'hacpp-bundle-flow-3-wrong-file-error.png'),
});

// State D: docs page — confirm it renders cleanly.
await page.goto(baseUrl + '/docs/hacpp-bundle/', { waitUntil: 'networkidle' });
await page.waitForTimeout(300);
await page.screenshot({
  path: join(outDir, 'hacpp-bundle-flow-4-docs-page.png'),
  fullPage: true,
});

// Dump a small assertion summary alongside the screenshots so the
// artifact trail tells you what was verified without reopening logs.
await writeFile(
  join(outDir, 'hacpp-bundle-flow-summary.json'),
  JSON.stringify(
    {
      baseUrl,
      acceptAfterSelect: acceptAttr,
      chipMode,
      helpHidden,
      errorBody: errBody.trim(),
      stamps: [
        'hacpp-bundle-flow-1-idle-default.png',
        'hacpp-bundle-flow-2-hacpp-selected.png',
        'hacpp-bundle-flow-3-wrong-file-error.png',
        'hacpp-bundle-flow-4-docs-page.png',
      ],
    },
    null,
    2,
  ),
);

await browser.close();
console.log('wrote screenshots to', outDir);
