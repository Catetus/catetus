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
page.on('requestfailed', r => errs.push('reqfail: '+r.url()+' '+r.failure()?.errorText));
await page.goto('http://127.0.0.1:4330/preview-hero.html?src=/hero-scene/scene.gltf&framing=0.85', { waitUntil: 'load' });

for (let i = 0; i < 8; i++) {
  await page.waitForTimeout(2000);
  const st = await page.evaluate(() => {
    const stateEl = document.getElementById('state');
    const chunksEl = document.getElementById('chunks');
    const splatsEl = document.getElementById('splats');
    return {
      state: stateEl?.textContent,
      chunks: chunksEl?.textContent,
      splats: splatsEl?.textContent,
    };
  });
  console.log(`t=${(i+1)*2}s`, JSON.stringify(st));
}
console.log('---console---');
for (const e of errs.slice(-20)) console.log(e);
await browser.close();
