import { chromium } from 'playwright-core';
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu','--enable-features=WebGPU'],
  headless: true,
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }});
const page = await ctx.newPage();
await page.goto('http://127.0.0.1:4330/', { waitUntil: 'load' });
await page.waitForTimeout(15000);
await page.screenshot({ path: 'tasks/renderer-unify-proof/hero-cpu.fullpage.png', timeout: 10000, animations: 'disabled' }).catch(e => console.log('shoot err', e.message));
console.log('done');
await browser.close();
