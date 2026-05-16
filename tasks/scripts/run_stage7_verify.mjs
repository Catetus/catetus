// Stage 7 verification driver: start local HTTP server serving the 4090's
// .lodge dir + the viewer dist + the harness HTML, then launch Chromium
// with the bench's known-working flags, navigate to the harness, wait for
// L1 to finish loading + render 5 poses, save PNGs, dump report.
import http from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { createReadStream, statSync } from 'node:fs';
import { resolve, join, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const REPO_PRE_RESOLVE = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
import { pathToFileURL } from 'node:url';
// Resolve playwright-core relative to this repo's tests/visual/node_modules
// (the bench harness already installs it there).
let chromium;
{
  const candidates = [
    'playwright-core',
    pathToFileURL(resolve(REPO_PRE_RESOLVE, 'tests/visual/node_modules/playwright-core/index.js')).href,
  ];
  for (const c of candidates) {
    try {
      const m = await import(c);
      chromium = m.chromium ?? m.default?.chromium;
      if (chromium) break;
    } catch (_) {}
  }
  if (!chromium) {
    console.error('FATAL: cannot find playwright-core in any of: ' + candidates.join(', '));
    process.exit(2);
  }
}

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..');
const LODGE = process.env.LODGE_DIR || 'C:/Users/monta/SplatForge/.bench-scenes/sweet-corals-full.lodge';
const VIEWER_DIST = resolve(REPO, 'packages', 'viewer', 'dist');
const HARNESS_HTML = resolve(HERE, 'stage7_render_harness.html');
const OUT_DIR = resolve(HERE, 'stage7_verify_out');
await mkdir(OUT_DIR, { recursive: true });

const LEVEL = parseInt(process.env.LEVEL || '1', 10);
const PORT = parseInt(process.env.PORT || '8765', 10);

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.ply': 'application/octet-stream', '.png': 'image/png', '.wasm': 'application/wasm' };
function mime(p) { return MIME[extname(p).toLowerCase()] || 'application/octet-stream'; }

async function tryServe(filePath, res) {
  try {
    const st = await stat(filePath);
    if (!st.isFile()) return false;
    res.writeHead(200, {
      'content-type': mime(filePath),
      'content-length': st.size,
      'access-control-allow-origin': '*',
    });
    createReadStream(filePath).pipe(res);
    return true;
  } catch {
    return false;
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  let p = decodeURIComponent(url.pathname);
  // Routes:
  //   /harness.html -> tasks/scripts/stage7_render_harness.html
  //   /viewer/*     -> packages/viewer/dist/*
  //   /lodge/*      -> $LODGE/*
  if (p === '/' || p === '/harness.html') {
    return void tryServe(HARNESS_HTML, res);
  }
  if (p.startsWith('/viewer/')) {
    return void tryServe(resolve(VIEWER_DIST, p.slice('/viewer/'.length)), res);
  }
  if (p.startsWith('/lodge/')) {
    return void tryServe(resolve(LODGE, p.slice('/lodge/'.length)), res);
  }
  res.writeHead(404, { 'content-type': 'text/plain' });
  res.end('not found: ' + p);
});

await new Promise((resolveS, reject) => server.listen(PORT, '127.0.0.1', resolveS).on('error', reject));
console.log('[server] listening on http://127.0.0.1:' + PORT);

const args = process.platform === 'darwin'
  ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer', '--use-angle=metal']
  : ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer', '--use-vulkan=swiftshader'];

// Prefer the system Chrome on Windows — bundled Chromium with
// --use-vulkan=swiftshader returns no WebGPU adapter on the 4090 host
// (crbug.com/369219127). System Chrome uses native Vulkan / DX.
const useSystemChrome = process.platform === 'win32' || process.env.CHANNEL === 'chrome';
const launchOpts = {
  headless: process.env.HEADLESS === '0' ? false : true,
  args: [...args, '--ignore-gpu-blocklist', '--no-sandbox'],
};
if (useSystemChrome) {
  launchOpts.channel = 'chrome';
  // Drop the swiftshader override so Chrome uses the real GPU driver.
  launchOpts.args = launchOpts.args.filter((a) => a !== '--use-vulkan=swiftshader');
}
console.log('[playwright] launching ' + (useSystemChrome ? 'system Chrome' : 'bundled Chromium') + ' headless=' + launchOpts.headless);
const browser = await chromium.launch(launchOpts);

const result = { level: LEVEL, port: PORT, errors: [] };
try {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await ctx.newPage();
  page.on('console', (msg) => {
    const t = msg.type();
    const text = msg.text();
    if (t === 'error') result.errors.push('[console.error] ' + text);
    process.stderr.write('[page:' + t + '] ' + text + '\n');
  });
  page.on('pageerror', (err) => result.errors.push('[pageerror] ' + err.message));

  const url = `http://127.0.0.1:${PORT}/harness.html?viewer=/viewer&manifest=/lodge/manifest.json&level=${LEVEL}`;
  console.log('[playwright] navigating to ' + url);
  await page.goto(url, { waitUntil: 'load' });

  // Wait for __sf.ready or __sf.error (up to 30 min for L1's 13 GB stream + readback).
  await page.waitForFunction(
    () => globalThis.__sf && (globalThis.__sf.ready === true || globalThis.__sf.error !== null),
    null,
    { timeout: 30 * 60 * 1000 },
  );

  const sf = await page.evaluate(() => ({
    ready: globalThis.__sf.ready, error: globalThis.__sf.error,
    frames: globalThis.__sf.frames, drawCalls: globalThis.__sf.drawCalls,
    numPages: globalThis.__sf.numPages, splatCount: globalThis.__sf.splatCount,
    captures: (globalThis.__sf.captures || []).map((c) => ({ pose: c.pose, yaw: c.yaw, drawCount: c.drawCount, dataUrl: c.dataUrl })),
  }));
  result.harness = { ready: sf.ready, error: sf.error, frames: sf.frames, drawCalls: sf.drawCalls, numPages: sf.numPages, splatCount: sf.splatCount };
  if (sf.error) throw new Error('harness reported error: ' + sf.error);
  console.log('[playwright] harness ready: ' + sf.frames + ' frames, ' + sf.drawCalls + ' draws/frame, ' + sf.numPages + ' pages');
  for (const cap of sf.captures) {
    const b64 = cap.dataUrl.split('base64,')[1];
    const buf = Buffer.from(b64, 'base64');
    const out = resolve(OUT_DIR, `L${LEVEL}_pose${cap.pose}.png`);
    await writeFile(out, buf);
    console.log('[saved] ' + out + ' (' + buf.length + ' B)');
  }

  if (sf.error) throw new Error('harness reported error: ' + sf.error);

  result.success = true;
} catch (err) {
  result.fatal = String(err && err.stack || err);
  console.error('[fatal]', result.fatal);
  result.success = false;
} finally {
  await browser.close();
  server.close();
}

const reportPath = resolve(OUT_DIR, `report_L${LEVEL}.json`);
await writeFile(reportPath, JSON.stringify(result, null, 2));
console.log('\n=== REPORT @ ' + reportPath + ' ===');
console.log(JSON.stringify(result, null, 2));
process.exit(result.success && result.errors.length === 0 ? 0 : 1);
