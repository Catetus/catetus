import { chromium } from 'playwright-core';
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu','--enable-features=WebGPU'],
  headless: true,
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }});
const page = await ctx.newPage();
const url = process.argv[2] || 'http://127.0.0.1:4330/preview-hero.html?src=/hero-scene/scene.gltf&framing=0.85';
await page.goto(url, { waitUntil: 'load' });
await page.waitForTimeout(15000);

// Read the canvas pixels via toDataURL.
const info = await page.evaluate(() => {
  const c = document.querySelector('canvas');
  if (!c) return { error: 'no canvas' };
  const url = c.toDataURL('image/png');
  // Decode to count non-black pixels (avg brightness > 20).
  const off = document.createElement('canvas');
  off.width = c.width; off.height = c.height;
  const oc = off.getContext('2d');
  const img = new Image();
  const ready = new Promise((r) => { img.onload = r; });
  img.src = url;
  return ready.then(() => {
    oc.drawImage(img, 0, 0);
    const data = oc.getImageData(0, 0, c.width, c.height).data;
    let nonBlack = 0;
    let avgR = 0, avgG = 0, avgB = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 20 || data[i+1] > 20 || data[i+2] > 20) nonBlack++;
      avgR += data[i]; avgG += data[i+1]; avgB += data[i+2];
    }
    const n = data.length / 4;
    return {
      width: c.width, height: c.height, pixels: n,
      nonBlackPx: nonBlack, nonBlackPct: (nonBlack / n * 100).toFixed(2),
      avg: { r: (avgR/n).toFixed(1), g: (avgG/n).toFixed(1), b: (avgB/n).toFixed(1) },
    };
  });
});
console.log(JSON.stringify(info, null, 2));
await browser.close();
