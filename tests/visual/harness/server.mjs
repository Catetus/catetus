/**
 * Tiny static file server for the Playwright harness.
 *
 * Routes:
 *   /page.html      -> harness/page.html
 *   /viewer/...     -> ../../packages/viewer/dist/...
 *   /fixtures/...   -> fixtures/...
 *   /               -> 404 (no index)
 *
 * No deps beyond node core. CORS-permissive, no caching — we want fresh
 * bytes on every test invocation.
 */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..');                            // tests/visual
const VIEWER_DIST = resolve(ROOT, '../../packages/viewer/dist');  // built viewer
const FIXTURES = resolve(ROOT, 'fixtures');
// Streaming-tile fixture (committed in the optimize crate, used by the
// streaming-tileset visual regression). We mount it transparently under
// /fixtures/geospatial-sample/* so test code stays consistent with the
// other fixture URLs.
const OPT_FIXTURES = resolve(ROOT, '../../crates/splatforge-optimize/tests/fixtures');
const PAGE_HTML = resolve(ROOT, 'harness/page.html');

const portArgIdx = process.argv.indexOf('--port');
const PORT = portArgIdx > 0 ? Number(process.argv[portArgIdx + 1]) : 4317;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.mjs':  'application/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gltf': 'model/gltf+json',
  '.glb':  'model/gltf-binary',
  '.bin':  'application/octet-stream',
  '.ply':  'application/octet-stream',
  '.spz':  'application/octet-stream',
  '.wasm': 'application/wasm',
};

/** Resolve `url` to an absolute file path under `base`, or null if it escapes. */
function safeJoin(base, url) {
  // Strip leading slash, normalize path traversal, then re-anchor under base.
  const trimmed = url.replace(/^\/+/, '');
  const joined = normalize(join(base, trimmed));
  if (!joined.startsWith(base + sep) && joined !== base) return null;
  return joined;
}

async function serveFile(res, filePath) {
  try {
    const s = await stat(filePath);
    if (!s.isFile()) return notFound(res);
    const body = await readFile(filePath);
    res.writeHead(200, {
      'content-type': MIME[extname(filePath).toLowerCase()] ?? 'application/octet-stream',
      'cache-control': 'no-store',
      'access-control-allow-origin': '*',
    });
    res.end(body);
  } catch (err) {
    if (err && err.code === 'ENOENT') return notFound(res);
    res.writeHead(500); res.end(String(err));
  }
}

function notFound(res) { res.writeHead(404, { 'content-type': 'text/plain' }); res.end('not found'); }

const server = createServer(async (req, res) => {
  const url = (req.url || '/').split('?')[0];

  if (url === '/' || url === '/page.html') return serveFile(res, PAGE_HTML);

  if (url.startsWith('/viewer/')) {
    const p = safeJoin(VIEWER_DIST, url.slice('/viewer/'.length));
    if (!p) return notFound(res);
    // Default to index.js when the viewer is requested as a bare module path.
    return serveFile(res, p);
  }

  if (url.startsWith('/fixtures/')) {
    const rel = url.slice('/fixtures/'.length);
    // Streaming-tile fixture is committed in crates/splatforge-optimize.
    // Try that location first, then fall back to the local tests/visual/fixtures.
    const optPath = safeJoin(OPT_FIXTURES, rel);
    if (optPath) {
      try {
        const s = await stat(optPath);
        if (s.isFile()) return serveFile(res, optPath);
      } catch { /* fall through */ }
    }
    const p = safeJoin(FIXTURES, rel);
    if (!p) return notFound(res);
    return serveFile(res, p);
  }

  // Health probe used by Playwright's webServer.url check (it pings the URL
  // itself — page.html — but if a CI script hits /healthz we'll answer).
  if (url === '/healthz') { res.writeHead(200); return res.end('ok'); }

  return notFound(res);
});

server.listen(PORT, '127.0.0.1', () => {
  // Single line so Playwright's webServer can wait on the port. Don't log
  // anything else to stdout — playwright.config.ts has stdout:'ignore'.
  process.stderr.write(`[harness] listening on http://127.0.0.1:${PORT}\n`);
});

const shutdown = () => server.close(() => process.exit(0));
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
