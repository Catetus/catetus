// Shoot antimatter15 splat viewer rendering bonsai (reference benchmark).
import { chromium } from 'playwright';

const OUT = '/Users/montabano1/Desktop/.wt-hero-v2/tasks/hero-v2-proof/antimatter15-reference.png';
// antimatter15's site hosts a few sample splats. The "bonsai" one is at
// https://antimatter15.com/splat/?url=bonsai/bonsai-7k-mini.splat OR
// the variant URL parameters. Use base URL with bonsai param.
const URL = 'https://antimatter15.com/splat/?url=garden.splat';

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-features=Vulkan', '--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'],
});
const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
const page = await ctx.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.warn('[page error]', e.message));

console.log('[shoot] goto', URL);
await page.goto(URL, { waitUntil: 'load', timeout: 60000 });
// wait for the canvas to render bonsai - settle 15s
await page.waitForTimeout(15000);
await page.screenshot({ path: OUT, fullPage: false });
console.log('[shoot] wrote', OUT);
await browser.close();
