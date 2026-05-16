import { chromium } from 'playwright-core';
const BASE = process.argv[2] || 'http://127.0.0.1:4330';
const URL = process.argv[3] || '/';
const browser = await chromium.launch({
  // macOS: don't force Vulkan — Dawn-Metal is the native (fast) WebGPU backend.
  args: ['--enable-unsafe-webgpu','--enable-features=WebGPU'],
  headless: true,
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }});
const page = await ctx.newPage();
const errs = [];
page.on('console', m => errs.push(m.type()+': '+m.text()));
page.on('pageerror', e => errs.push('pageerror: '+(e.message||e)));
await page.goto(BASE + URL, { waitUntil: 'load', timeout: 20000 });
await page.waitForTimeout(1500);
const samples = [];
for (let i=0;i<20;i++) {
  const fps = await page.evaluate(async ()=>{
    const start = performance.now();
    let n=0;
    return await new Promise(r=>{
      function tick(){ n++; if(performance.now()-start>=1000) r(n/((performance.now()-start)/1000)); else requestAnimationFrame(tick); }
      requestAnimationFrame(tick);
    });
  });
  samples.push(+fps.toFixed(1));
}
console.log('fps samples (20x 1s):', samples.join(','));
console.log('console:', errs.slice(-20));
await browser.close();
