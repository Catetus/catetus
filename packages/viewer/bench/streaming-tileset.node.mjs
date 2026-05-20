#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
/**
 * Node-side streaming-tile bench. Runs the entire streaming pipeline
 * EXCEPT the GPU rasterization stage against the committed
 * `geospatial-sample` fixture and reports cold-start, sustained-FPS, and
 * peak resident bytes.
 *
 * Why a node bench? The WebGPU bench in `streaming-tileset.bench.ts`
 * requires a real GPU device, which is not always available in CI. The
 * streaming adapter's hot path is bandwidth- + JS-bound (parse, frustum,
 * SSE, LRU touch); the GPU draw is constant per resident splat-count. We
 * measure the JS path here and treat per-tile GPU upload as
 * proportional-to-bytes (see STREAMING.md).
 *
 * Run: `node packages/viewer/bench/streaming-tileset.node.mjs`.
 *
 * Output: a single JSON line to stdout + write `bench/streaming-results.json`.
 */
import { performance } from 'node:perf_hooks';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const fixtureDir = resolve(root, '../../crates/catetus-optimize/tests/fixtures/geospatial-sample');

// Ensure the viewer is built so we can import the compiled streaming code.
const tscBin = resolve(root, 'node_modules', '.bin', 'tsc');
spawnSync(tscBin, ['-p', resolve(root, 'tsconfig.json')], { cwd: root, stdio: 'inherit' });

const distEntry = pathToFileURL(resolve(root, 'dist', 'index.js')).href;
const { parseTileset, TileStreamer, decodeGlb, manifestFromGlb, extractFrustum, selectVisibleTiles, orbitFrames, orbitPose } =
  await import(distEntry);

// Local-file fetch mock — the streamer's default fetch wants HTTP. We swap
// in a synchronous file-system reader so the bench's "network" cost is
// bounded by disk read latency.
const fileFetch = async (url) => {
  const u = new URL(url);
  const bytes = await readFile(u);
  return {
    ok: true,
    status: 200,
    arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  };
};

// Cold-start: parse tileset.json + fetch root tile.
const coldStart0 = performance.now();
const tilesetJson = await readFile(resolve(fixtureDir, 'tileset.json'), 'utf-8');
const tileset = parseTileset(tilesetJson, pathToFileURL(resolve(fixtureDir, 'tileset.json')).href);
const streamer = new TileStreamer({ fetch: fileFetch, maxBytes: 512 * 1024 * 1024 });

// Prime the root tile.
await streamer.fetchTile(tileset.root, Number.MAX_SAFE_INTEGER);
const coldStartMs = performance.now() - coldStart0;

// Sustained FPS over an orbit. We synthesize per-frame camera poses + run
// the selector + streamer. To make the comparison meaningful we *do*
// process the GLB on first-touch of a tile — the real viewer would
// uploadChunk on the GPU at that point, which costs O(bytes_per_tile).
const yaws = orbitFrames(60);
// Bbox from the README.json.
const bbox = { min: [-3.7, -3.5, -0.8], max: [3.7, 3.5, 0.8] };
const aspect = 1;
let peakResidentBytes = 0;
const processedTiles = new Set();

// Build viewProj matrix locally without importing the full renderer (it
// pulls in WebGPU types). The streaming module re-exports `extractFrustum`,
// so we just need lookAt + perspective. Replicating ~30 lines here is
// cheaper than a circular import.
function mat4Identity() { const m = new Float32Array(16); m[0]=m[5]=m[10]=m[15]=1; return m; }
function lookAt(eye, target, up) {
  const fx = target[0]-eye[0], fy = target[1]-eye[1], fz = target[2]-eye[2];
  const fl = Math.hypot(fx,fy,fz)||1; const f0=fx/fl, f1=fy/fl, f2=fz/fl;
  let sx = f1*up[2]-f2*up[1], sy = f2*up[0]-f0*up[2], sz = f0*up[1]-f1*up[0];
  const sl = Math.hypot(sx,sy,sz)||1; sx/=sl; sy/=sl; sz/=sl;
  const ux = sy*f2-sz*f1, uy = sz*f0-sx*f2, uz = sx*f1-sy*f0;
  const m = new Float32Array(16);
  m[0]=sx; m[4]=sy; m[8]=sz; m[12]=-(sx*eye[0]+sy*eye[1]+sz*eye[2]);
  m[1]=ux; m[5]=uy; m[9]=uz; m[13]=-(ux*eye[0]+uy*eye[1]+uz*eye[2]);
  m[2]=-f0; m[6]=-f1; m[10]=-f2; m[14]=(f0*eye[0]+f1*eye[1]+f2*eye[2]);
  m[3]=0; m[7]=0; m[11]=0; m[15]=1;
  return m;
}
function persp(fovY, aspect, n, f) {
  const tf = 1/Math.tan(fovY*0.5), nf = 1/(n-f);
  const m = new Float32Array(16);
  m[0] = tf/aspect; m[5] = tf; m[10] = (f+n)*nf; m[11] = -1; m[14] = 2*f*n*nf;
  return m;
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) {
    let s=0; for (let k=0;k<4;k++) s += a[k*4+r]*b[c*4+k];
    o[c*4+r]=s;
  }
  return o;
}

const t0 = performance.now();
for (const yaw of yaws) {
  const pose = orbitPose(bbox, yaw, aspect);
  const view = lookAt(pose.position, pose.target, pose.up);
  const proj = persp(pose.fovY, aspect, pose.near, pose.far);
  const viewProj = mul(proj, view);
  const frustum = extractFrustum(viewProj);

  // Build resident set.
  const resident = new Set();
  for (const t of tileset.tiles) {
    if (streamer.stateOf(t) === 'loaded') resident.add(t.id);
  }
  const sel = selectVisibleTiles(tileset.root, {
    eye: pose.position,
    fovY: pose.fovY,
    viewportHeight: 512,
    frustum,
    maximumScreenSpaceError: 16,
    resident,
  });
  streamer.touch(sel.render);

  // Kick off fetches.
  for (const tile of sel.fetch) {
    streamer.fetchTile(tile, 1000 - tile.depth);
  }
  // Deterministic mode: wait for fetches to finish before "rendering".
  while (streamer.stats().inFlight > 0) {
    await new Promise((r) => setImmediate(r));
  }

  // Re-run selection with the new resident set.
  resident.clear();
  for (const t of tileset.tiles) {
    if (streamer.stateOf(t) === 'loaded') resident.add(t.id);
  }
  const sel2 = selectVisibleTiles(tileset.root, {
    eye: pose.position,
    fovY: pose.fovY,
    viewportHeight: 512,
    frustum,
    maximumScreenSpaceError: 16,
    resident,
  });

  // Process newly-rendered tiles: decode their GLB and build a manifest.
  // This stands in for `renderer.uploadChunk` (which is GPU-bound).
  for (const tile of sel2.render) {
    if (processedTiles.has(tile.id)) continue;
    const payload = streamer.get(tile);
    if (!payload) continue;
    // The streamer's get() returned the raw bytes; we already decoded GLB
    // on fetch. Build a manifest to simulate the upload path.
    const { manifest } = manifestFromGlb({ json: payload.json, bin: payload.bin });
    // Validate the chunk count + splat count — sanity check.
    if (manifest.chunks.length !== 1) {
      throw new Error(`unexpected chunk count for ${tile.id}`);
    }
    processedTiles.add(tile.id);
  }

  const s = streamer.stats();
  if (s.residentBytes > peakResidentBytes) peakResidentBytes = s.residentBytes;
}
const totalMs = performance.now() - t0;
const fps = (yaws.length * 1000) / totalMs;
const stats = streamer.stats();

const result = {
  coldStartMs: +coldStartMs.toFixed(3),
  frames: yaws.length,
  totalMs: +totalMs.toFixed(3),
  fps: +fps.toFixed(2),
  peakResidentBytes,
  evictions: stats.evictions,
  cacheHits: stats.cacheHits,
  cacheMisses: stats.cacheMisses,
  residentTiles: stats.residentTiles,
  residentBytes: stats.residentBytes,
  budgetBytes: 512 * 1024 * 1024,
  pctOfBudget: +((stats.residentBytes / (512 * 1024 * 1024)) * 100).toFixed(4),
  tilesProcessed: processedTiles.size,
  timestamp: new Date().toISOString(),
};

console.log(JSON.stringify(result, null, 2));
const out = resolve(root, 'bench', 'streaming-results.json');
await mkdir(dirname(out), { recursive: true });
await writeFile(out, JSON.stringify(result, null, 2) + '\n');
