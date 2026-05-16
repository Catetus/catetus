import { chromium } from 'playwright-core';
import { writeFileSync } from 'node:fs';
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu','--enable-features=WebGPU'],
  headless: true,
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }});
const page = await ctx.newPage();
page.on('console', m => { if (m.type()==='error') console.log('ERR:', m.text()); });
await page.goto('http://127.0.0.1:4330/scale', { waitUntil: 'load' });
// Settle for chunk downloads. /scale's L5 is 33 chunks from Vercel Blob;
// the first ~5 chunks land in <5 s and the renderer starts drawing then.
await page.waitForTimeout(20000).catch(()=>{});
// Grab a measurement before the crash, then attempt screenshot.
let fps = 0;
try {
  fps = await page.evaluate(async ()=>{
    const start = performance.now();
    let n=0;
    return await new Promise(r=>{
      function tick(){ n++; if(performance.now()-start>=1000) r(n/((performance.now()-start)/1000)); else requestAnimationFrame(tick); }
      requestAnimationFrame(tick);
    });
  });
} catch(e){ console.log('fps err:', e.message); }
console.log('fps:', fps.toFixed(1));
try {
  const dataUrl = await page.evaluate(() => {
    const c = document.querySelector('canvas');
    return c?.toDataURL ? c.toDataURL('image/png') : null;
  });
  if (dataUrl) {
    writeFileSync('tasks/renderer-unify-proof/scale.png', Buffer.from(dataUrl.slice(22), 'base64'));
    console.log('scale.png written');
  }
} catch(e){ console.log('shot err:', e.message); }
await browser.close();
