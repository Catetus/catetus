import { chromium } from 'playwright-core';
const BASE = process.argv[2] || 'http://127.0.0.1:4330';
const URL = process.argv[3] || '/';
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-vulkan'],
  headless: true,
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }});
const page = await ctx.newPage();
const errs = [];
page.on('console', m => { if (m.type()==='error'||m.type()==='warning') errs.push(m.type()+': '+m.text()); });
page.on('pageerror', e => errs.push('pageerror: '+(e.message||e)));
await page.goto(BASE + URL, { waitUntil: 'load', timeout: 20000 });
await page.waitForTimeout(2000);
// Sample fps every 2s for 12s.
const samples = [];
for (let i=0;i<6;i++) {
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
const status = await page.evaluate(()=>{
  const el = document.querySelector('[data-hero-status]');
  return el ? el.textContent : null;
});
console.log('fps samples:', samples.join(','));
console.log('hero status:', status);
console.log('console (last 10):', errs.slice(-10));
await browser.close();
