// Playwright screenshot of /xr in no-headset and emulator-stubbed states.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

const PORT = 4321;
const HOST = `http://127.0.0.1:${PORT}`;

const server = spawn('npx', ['--yes', 'http-server', 'dist', '-p', String(PORT), '-c', '-1', '-s'], {
  stdio: 'ignore',
});
process.on('exit', () => server.kill());

// Wait for server.
for (let i = 0; i < 30; i++) {
  try {
    const r = await fetch(`${HOST}/xr/`);
    if (r.ok) break;
  } catch {}
  await sleep(250);
}

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });

{
  // State 1: no navigator.xr at all.
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'xr', { value: undefined, configurable: true });
  });
  await page.goto(`${HOST}/xr/`, { waitUntil: 'networkidle' });
  await page.waitForSelector('[data-xr-status]');
  await sleep(400);
  await page.screenshot({ path: 'screenshots/xr-no-headset.png', fullPage: true });
  await page.close();
}

{
  // State 2: stub navigator.xr reporting BOTH modes supported (simulated emulator).
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'xr', {
      configurable: true,
      value: {
        isSessionSupported: async () => true,
        requestSession: async () => { throw new Error('headless emulator stub'); },
      },
    });
  });
  await page.goto(`${HOST}/xr/`, { waitUntil: 'networkidle' });
  await page.waitForSelector('[data-xr-status]');
  // Wait for the probe to resolve.
  await page.waitForFunction(() => {
    const s = document.querySelector('[data-xr-status]')?.textContent || '';
    return s.includes('immersive-vr: yes');
  }, { timeout: 5000 });
  await sleep(200);
  await page.screenshot({ path: 'screenshots/xr-stubbed-emulator.png', fullPage: true });
  await page.close();
}

await browser.close();
server.kill();
console.log('done');
