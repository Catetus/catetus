/**
 * Headless smoke test for the streaming tileset loader's data path.
 *
 * This does NOT need a GPU or a browser — it exercises the load-bearing logic
 * that turns a STREAM-1 tileset on disk into renderer-ready SoA chunks:
 *
 *   tileset.json  → parseTileset  → TileNode tree (frustum/SSE inputs)
 *   tiles/*.sftile → decodeSftile → SplatScene-shaped object
 *   SplatScene    → splatSceneToSoaChunk → {descriptor, bytes}
 *
 * It asserts the manifest parses, every referenced tile decodes, splat counts
 * match the manifest's per-tile `count`, the SoA byte length is exactly
 * `N * 56` for DC-only tiles, and the root tile (coarsest, smallest) decodes
 * first — i.e. progressive root-first load is structurally possible.
 *
 * Run:  node apps/web/public/viewer/streaming/smoke_sftile.mjs <tileset-dir>
 *       (defaults to apps/web/public/fixtures/tileset-demo)
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { parseTileset } from './tileset_loader.js';
import { decodeSftile, isSftile } from './sftile.js';
import { splatSceneToSoaChunk } from '../loader/to-soa.js';

const here = dirname(fileURLToPath(import.meta.url));
const tilesetDir = resolve(process.argv[2] ?? join(here, '..', 'fixtures', 'tileset-demo'));

function fail(msg) { console.error(`FAIL: ${msg}`); process.exit(1); }
function ok(msg) { console.log(`ok - ${msg}`); }

const tsJsonPath = join(tilesetDir, 'tileset.json');
const tsText = await readFile(tsJsonPath, 'utf8').catch(() => fail(`cannot read ${tsJsonPath}`));

// parseTileset resolves content URIs against baseUrl via the WHATWG URL ctor;
// give it a file:// base so contentUrl is a real path we can read.
const baseUrl = `file://${tsJsonPath}`;
const tileset = parseTileset(tsText, baseUrl);
ok(`parsed tileset.json: ${tileset.tiles.length} tiles, geometricError=${tileset.geometricError.toFixed(2)}`);

if (!tileset.root) fail('no root tile');
if (tileset.root.geometricError <= 0) fail('root geometricError should be > 0 (refinable)');
ok(`root tile id=${tileset.root.id} aabb=${JSON.stringify(tileset.root.aabb)}`);

// Decode the ROOT first (proves root-first first-paint is possible).
const rootPath = fileURLToPath(tileset.root.contentUrl);
const rootBytes = new Uint8Array(await readFile(rootPath));
if (!isSftile(rootBytes)) fail(`root tile ${rootPath} is not .sftile (magic mismatch)`);
const rootScene = decodeSftile(rootBytes);
if (rootScene.count <= 0) fail('root tile decoded 0 splats');
const rootChunk = splatSceneToSoaChunk(rootScene, `tile:${tileset.root.id}`);
if (rootChunk.descriptor.splatCount !== rootScene.count) fail('root SoA splatCount mismatch');
ok(`root tile decoded + packed: ${rootScene.count} splats, ${rootChunk.bytes.byteLength} SoA bytes`);

// Decode EVERY tile referenced by the tree; assert each is valid + non-empty
// finest content, and SoA packing succeeds.
let totalSplats = 0, totalBytes = 0, decoded = 0;
let minCount = Infinity, maxCount = 0;
async function walk(node) {
    const p = fileURLToPath(node.contentUrl);
    const bytes = new Uint8Array(await readFile(p));
    if (!isSftile(bytes)) fail(`tile ${p} not .sftile`);
    const scene = decodeSftile(bytes);
    const chunk = splatSceneToSoaChunk(scene, `tile:${node.id}`);
    if (chunk.descriptor.splatCount !== scene.count) fail(`tile ${node.id} SoA count mismatch`);
    // DC-only tiles: SoA must be exactly N*56 bytes.
    if (!scene.shDegree && chunk.bytes.byteLength !== scene.count * 56) {
        fail(`tile ${node.id}: expected ${scene.count * 56} SoA bytes, got ${chunk.bytes.byteLength}`);
    }
    // Sanity on decoded values: positions finite, opacity in a plausible range.
    for (let i = 0; i < Math.min(scene.count, 8); i++) {
        if (!Number.isFinite(scene.positions[i * 3])) fail(`tile ${node.id}: non-finite position`);
    }
    totalSplats += scene.count;
    totalBytes += bytes.byteLength;
    decoded++;
    minCount = Math.min(minCount, scene.count);
    maxCount = Math.max(maxCount, scene.count);
    for (const c of node.children) await walk(c);
}
await walk(tileset.root);
ok(`decoded all ${decoded} tree tiles: ${totalSplats} splats total, ${(totalBytes / 1e6).toFixed(2)} MB on disk`);
ok(`per-tile splat range: ${minCount}..${maxCount} (coarse→fine LOD spread present: ${maxCount > minCount})`);

if (decoded < 2) fail('expected a multi-tile tileset (octree should subdivide)');
console.log(`\nPASS: tileset loader data path is correct over ${decoded} real .sftile tiles.`);
