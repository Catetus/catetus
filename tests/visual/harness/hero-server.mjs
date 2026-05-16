/**
 * Hero-regression static server.
 *
 * Serves three roots:
 *   /                  -> apps/web/public/                 (preview-hero.html, /viewer/, /hero-scene/)
 *   /tmp-scenes/...    -> tmp/regression/...               (CLI-produced gltf outputs)
 *   /viewer/...        -> packages/viewer/dist/...         (preferred — built viewer with module imports)
 *
 * The viewer mount overrides apps/web/public/viewer so we pick up the freshest
 * built bundle. If packages/viewer/dist does not exist we transparently fall
 * back to apps/web/public/viewer.
 */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT_TESTS_VISUAL = resolve(__dirname, '..');                       // tests/visual
const REPO_ROOT = resolve(ROOT_TESTS_VISUAL, '../..');                    // SplatForge root
const WEB_PUBLIC = resolve(REPO_ROOT, 'apps/web/public');
const VIEWER_DIST = resolve(REPO_ROOT, 'packages/viewer/dist');
const TMP_SCENES = resolve(REPO_ROOT, 'tmp/regression');

const portArgIdx = process.argv.indexOf('--port');
const PORT = portArgIdx > 0 ? Number(process.argv[portArgIdx + 1]) : 4321;

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
  '.wasm': 'application/wasm',
};

function safeJoin(base, url) {
  const trimmed = url.replace(/^\/+/, '');
  const joined = normalize(join(base, trimmed));
  if (!joined.startsWith(base + sep) && joined !== base) return null;
  return joined;
}

async function serveFile(res, filePath) {
  try {
    const s = await stat(filePath);
    if (!s.isFile()) return false;
    const body = await readFile(filePath);
    res.writeHead(200, {
      'content-type': MIME[extname(filePath).toLowerCase()] ?? 'application/octet-stream',
      'access-control-allow-origin': '*',
      'cache-control': 'no-store',
    });
    res.end(body);
    return true;
  } catch {
    return false;
  }
}

const server = createServer(async (req, res) => {
  const url = (req.url ?? '/').split('?')[0];

  // /tmp-scenes/<scene>/<preset>/scene.gltf
  if (url.startsWith('/tmp-scenes/')) {
    const fp = safeJoin(TMP_SCENES, url.slice('/tmp-scenes/'.length));
    if (fp && await serveFile(res, fp)) return;
    res.writeHead(404); return res.end('not found');
  }

  // /viewer/<...> -> prefer built dist
  if (url.startsWith('/viewer/')) {
    const rel = url.slice('/viewer/'.length);
    if (existsSync(VIEWER_DIST)) {
      const fp = safeJoin(VIEWER_DIST, rel);
      if (fp && await serveFile(res, fp)) return;
    }
    // fall through to apps/web/public/viewer/
    const fp = safeJoin(resolve(WEB_PUBLIC, 'viewer'), rel);
    if (fp && await serveFile(res, fp)) return;
    res.writeHead(404); return res.end('viewer asset missing');
  }

  // everything else from apps/web/public
  const relative = url === '/' ? 'preview-hero.html' : url.slice(1);
  const fp = safeJoin(WEB_PUBLIC, relative);
  if (fp && await serveFile(res, fp)) return;

  res.writeHead(404);
  res.end('not found');
});

server.listen(PORT, '127.0.0.1', () => {
  // eslint-disable-next-line no-console
  console.log(`[hero-server] listening on http://127.0.0.1:${PORT}`);
});
