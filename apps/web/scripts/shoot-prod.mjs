import { chromium } from 'playwright';
const OUT = '/Users/montabano1/Desktop/.wt-hero-v2/tasks/hero-v2-proof/PROD-current.png';
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-features=Vulkan,WebGPU', '--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'] });
const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
const page = await ctx.newPage();
page.on('console', (m) => { const t = m.text(); if (/error|warn|hero|webgpu|viewer/i.test(t)) console.log('[page]', t); });
console.log('[shoot] goto https://splatforge.dev/');
await page.goto('https://splatforge.dev/', { waitUntil: 'networkidle', timeout: 60000 });
try { await page.waitForFunction(() => /live|webgpu|webgl2/i.test(document.querySelector('[data-hero-status]')?.textContent || ''), { timeout: 30000 }); } catch {}
const status = await page.$eval('[data-hero-status]', e => e.textContent).catch(() => '?');
console.log('STATUS:', status);
await page.waitForTimeout(10000);
const stage = await page.$('[data-hero-stage]');
await stage.screenshot({ path: OUT });
console.log('wrote', OUT);
await browser.close();
