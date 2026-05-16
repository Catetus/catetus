import { chromium } from 'playwright';
const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-features=Vulkan,WebGPU', '--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-vulkan'],
});
const page = await browser.newContext({viewport:{width:1920,height:1080}}).then(c=>c.newPage());
await page.goto('http://127.0.0.1:4500/', { waitUntil: 'networkidle' });
const info = await page.evaluate(async () => {
  const hasGpu = typeof navigator.gpu !== 'undefined';
  let adapter = null;
  let adapterInfo = null;
  if (hasGpu) {
    try {
      adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        adapterInfo = {
          isFallback: adapter.isFallbackAdapter,
          features: Array.from(adapter.features),
          limits: Object.fromEntries(Object.entries(adapter.limits || {})),
        };
      }
    } catch (e) {
      adapterInfo = { error: String(e) };
    }
  }
  const statusEl = document.querySelector('[data-hero-status]');
  return {
    hasGpu,
    adapterPresent: !!adapter,
    adapterInfo,
    statusText: statusEl ? statusEl.textContent : null,
  };
});
console.log(JSON.stringify(info, null, 2));
// wait then snapshot status again
await page.waitForTimeout(10000);
const finalStatus = await page.evaluate(() => document.querySelector('[data-hero-status]')?.textContent);
console.log('FINAL STATUS:', finalStatus);
await browser.close();
