import { chromium } from 'playwright-core';
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu','--enable-features=WebGPU'],
  headless: true,
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }});
const page = await ctx.newPage();
const errs = [];
page.on('console', m => errs.push(m.type()+': '+m.text()));
page.on('pageerror', e => errs.push('pageerror: '+(e.message||e)));

// Inject a viewer-error listener BEFORE page scripts run.
await page.addInitScript(() => {
  window.__sf_errs = [];
  const orig = Error;
  // No-op for now — the SDK emits 'error' events on the viewer
  // instance; we can't intercept without code change.
});

await page.goto('http://127.0.0.1:4330/preview-hero.html?src=/hero-scene/scene.gltf', { waitUntil: 'load' });
await page.waitForTimeout(8000);
const st = await page.evaluate(() => ({
  state: document.getElementById('state')?.textContent,
  chunks: document.getElementById('chunks')?.textContent,
  splats: document.getElementById('splats')?.textContent,
}));
console.log('state:', JSON.stringify(st));
console.log('--- console ---');
for (const e of errs) console.log(e);
await browser.close();
