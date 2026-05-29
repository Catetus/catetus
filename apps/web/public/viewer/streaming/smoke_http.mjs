/**
 * HTTP-transport smoke: proves the tileset + tiles + viewer modules are
 * served over real HTTP exactly as the browser would fetch them, and that the
 * loader can parse the manifest fetched over HTTP and locate/range every tile.
 *
 * Run (with `python3 -m http.server 8137` serving apps/web/public):
 *   node apps/web/public/viewer/streaming/smoke_http.mjs http://localhost:8137
 */
import { parseTileset } from './tileset_loader.js';
import { decodeSftile, isSftile } from './sftile.js';

const origin = process.argv[2] ?? 'http://localhost:8137';
const tilesetUrl = `${origin}/fixtures/tileset-demo/tileset.json`;
function fail(m) { console.error(`FAIL: ${m}`); process.exit(1); }

// 1. The viewer entry + new modules must be served.
for (const path of [
    '/viewer/index.html',
    '/viewer/streaming/runtime.js',
    '/viewer/streaming/sftile.js',
    '/viewer/streaming/tile_streamer.js',
    '/fixtures/tileset-demo/tileset.json',
    '/fixtures/tileset-demo/lod-meta.json',
]) {
    const r = await fetch(`${origin}${path}`);
    if (!r.ok) fail(`HTTP ${r.status} for ${path}`);
    console.log(`ok - served ${path} (${r.headers.get('content-length') ?? '?'} B)`);
}

// 2. index.html must carry the ?tileset= wiring.
const html = await (await fetch(`${origin}/viewer/index.html`)).text();
if (!html.includes("params.get('tileset')")) fail('index.html missing ?tileset= bootstrap');
if (!html.includes('loadTileset(')) fail('index.html missing loadTileset()');
console.log('ok - index.html has ?tileset= bootstrap + loadTileset()');

// 3. Parse the HTTP-fetched manifest and fetch+decode every tile over HTTP,
//    root first (the actual streaming order the viewer uses).
const tileset = parseTileset(await (await fetch(tilesetUrl)).text(), tilesetUrl);
console.log(`ok - parsed HTTP manifest: ${tileset.tiles.length} tiles`);

async function fetchTile(node) {
    const r = await fetch(node.contentUrl);
    if (!r.ok) fail(`HTTP ${r.status} for tile ${node.id} (${node.contentUrl})`);
    const bytes = new Uint8Array(await r.arrayBuffer());
    if (!isSftile(bytes)) fail(`tile ${node.id} served bytes are not .sftile`);
    return decodeSftile(bytes);
}
const root = await fetchTile(tileset.root);
console.log(`ok - root tile fetched+decoded over HTTP first: ${root.count} splats`);

let n = 0, splats = 0;
async function walk(node) { const s = await fetchTile(node); n++; splats += s.count; for (const c of node.children) await walk(c); }
await walk(tileset.root);
console.log(`ok - all ${n} tiles fetched+decoded over HTTP: ${splats} splats`);
if (n < 2) fail('expected multi-tile tileset over HTTP');
console.log(`\nPASS: tileset is served + progressively loadable over real HTTP (${n} tiles).`);
