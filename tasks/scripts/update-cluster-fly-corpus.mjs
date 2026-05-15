#!/usr/bin/env node
// Update benches/scenes/real/manifest.json and benches/reports/splatbench-v0.json
// with the cluster_fly_{s,m,l,xl,xxl} entries. Derives ratios from optimize
// output files; pulls splatCount/hash from splatforge analyze JSON written into
// tasks/cluster-fly-out/analyze_<size>.json.
//
// Idempotent: re-running replaces matching ids in-place; aggregates are
// recomputed from the full scenes array.
import { readFileSync, writeFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(new URL('.', import.meta.url).pathname, '..', '..');
const REAL = resolve(ROOT, 'benches', 'scenes', 'real');
const OUTDIR = resolve(ROOT, 'tasks', 'cluster-fly-out');

const SIZES = ['s', 'm', 'l', 'xl', 'xxl'];

const ATTRIBUTION = 'Dany Bittel (CC-BY-4.0) — www.danybittel.ch';
const SOURCE_URL = 'https://www.danybittel.ch';
const LICENSE = 'CC-BY-4.0';
const CAPTURE_PROFILE = 'indoor close-up';

function readJSON(p) {
  return JSON.parse(readFileSync(p, 'utf8'));
}
function writeJSON(p, obj) {
  writeFileSync(p, JSON.stringify(obj, null, 2) + '\n');
}

function bytes(p) {
  return statSync(p).size;
}

function ratio(a, b) {
  return Number((a / b).toFixed(2));
}

// Build per-scene rows from on-disk analyze + optimize outputs.
const rows = SIZES.map((size) => {
  const id = `cluster_fly_${size}`;
  const ply = resolve(REAL, `${id}.ply`);
  const analyze = readJSON(resolve(OUTDIR, `analyze_${size}.json`));
  const wm = resolve(OUTDIR, 'optimize', `${id}.web-mobile.glb`);
  const sm = resolve(OUTDIR, 'optimize', `${id}.size-min.glb`);
  const bytesIn = bytes(ply);
  const wmBytes = bytes(wm);
  const smBytes = bytes(sm);
  return {
    id,
    source: 'real',
    class: 'indoor-close-up',
    origin: SOURCE_URL,
    license: `Dany Bittel (CC-BY-4.0); www.danybittel.ch`,
    attribution: ATTRIBUTION,
    captureProfile: CAPTURE_PROFILE,
    splatCount: analyze.splatCount,
    bytesIn,
    shDegree: analyze.shDegree,
    hash: analyze.hash,
    webMobileSpzBytes: wmBytes,
    webMobileRatio: ratio(bytesIn, wmBytes),
    sizeMinSpzBytes: smBytes,
    sizeMinRatio: ratio(bytesIn, smBytes),
  };
});

// ---- manifest.json ----
const manifestPath = resolve(REAL, 'manifest.json');
const manifest = readJSON(manifestPath);

const manifestNewEntries = rows.map((r) => ({
  id: r.id,
  filename: `${r.id}.ply`,
  splatCount: r.splatCount,
  bytesIn: r.bytesIn,
  shDegree: r.shDegree,
  hash: r.hash,
  sourceUrl: SOURCE_URL,
  license: LICENSE,
  attribution: ATTRIBUTION,
  capture_profile: CAPTURE_PROFILE,
  pulledBy: 'benches/scenes/scripts/pull-cluster-fly.sh',
  rationale:
    'Indoor close-up macro photogrammetry (cluster fly / Pollenia). Adds the missing indoor close-up texture profile to a corpus currently dominated by Mip-NeRF 360 outdoor scenes. 5-LOD ladder (S→XXL: 25k → 3.5M splats) lets SplatBench measure how ratios and fidelity scale with splat count on the same scene.',
}));

// Replace any existing cluster_fly_* entries; preserve everything else.
manifest.scenes = (manifest.scenes ?? []).filter(
  (s) => !s.id?.startsWith('cluster_fly_'),
);
manifest.scenes.push(...manifestNewEntries);

writeJSON(manifestPath, manifest);
console.log(`[manifest] wrote ${manifestNewEntries.length} cluster_fly entries`);

// ---- splatbench-v0.json ----
const bench = readJSON(resolve(ROOT, 'benches', 'reports', 'splatbench-v0.json'));
bench.scenes = (bench.scenes ?? []).filter((s) => !s.id?.startsWith('cluster_fly_'));
bench.scenes.push(...rows);

// Recompute aggregates from the full scenes array.
const scenes = bench.scenes;
const realCount = scenes.filter((s) => s.source === 'real').length;
const synthCount = scenes.filter((s) => s.source === 'synthetic').length;
const splatTotal = scenes.reduce((a, s) => a + (s.splatCount || 0), 0);
const bytesInTotal = scenes.reduce((a, s) => a + (s.bytesIn || 0), 0);
const wmSpzTotal = scenes.reduce((a, s) => a + (s.webMobileSpzBytes || 0), 0);
const smSpzTotal = scenes.reduce((a, s) => a + (s.sizeMinSpzBytes || 0), 0);
const wmRatios = scenes.map((s) => s.webMobileRatio).filter(Number.isFinite).sort((a, b) => a - b);
const smRatios = scenes.map((s) => s.sizeMinRatio).filter(Number.isFinite).sort((a, b) => a - b);
const median = (arr) => {
  if (!arr.length) return 0;
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[m] : Number(((arr[m - 1] + arr[m]) / 2).toFixed(2));
};
const wmZstdScenes = scenes.filter((s) => Number.isFinite(s.webMobileZstdRatio));
const wmZstdRatios = wmZstdScenes.map((s) => s.webMobileZstdRatio).sort((a, b) => a - b);

bench.aggregates = {
  scenesTotal: scenes.length,
  scenesReal: realCount,
  scenesSynthetic: synthCount,
  splatCountTotal: splatTotal,
  bytesInTotal,
  webMobileSpzTotal: wmSpzTotal,
  sizeMinSpzTotal: smSpzTotal,
  webMobileRatioOverall: Number((bytesInTotal / Math.max(1, wmSpzTotal)).toFixed(2)),
  sizeMinRatioOverall: Number((bytesInTotal / Math.max(1, smSpzTotal)).toFixed(2)),
  webMobileRatioMin: wmRatios[0],
  webMobileRatioMedian: median(wmRatios),
  webMobileRatioMax: wmRatios[wmRatios.length - 1],
  sizeMinRatioMin: smRatios[0],
  sizeMinRatioMedian: median(smRatios),
  sizeMinRatioMax: smRatios[smRatios.length - 1],
  // Preserve fidelity counts from prior aggregates (no fidelity run for new scenes yet).
  fidelityWebMobilePass: bench.aggregates?.fidelityWebMobilePass ?? null,
  fidelitySizeMinPass: bench.aggregates?.fidelitySizeMinPass ?? null,
  webMobileZstdRatioMin: wmZstdRatios[0] ?? null,
  webMobileZstdRatioMedian: median(wmZstdRatios),
  webMobileZstdRatioMax: wmZstdRatios[wmZstdRatios.length - 1] ?? null,
  webMobileZstdScenesCovered: wmZstdScenes.length,
};

writeJSON(resolve(ROOT, 'benches', 'reports', 'splatbench-v0.json'), bench);
console.log(`[splatbench-v0] scenesTotal=${scenes.length} real=${realCount} synth=${synthCount}`);

// Console table.
console.table(
  rows.map((r) => ({
    id: r.id,
    splats: r.splatCount,
    bytesIn: r.bytesIn,
    wmBytes: r.webMobileSpzBytes,
    wmRatio: r.webMobileRatio,
    smBytes: r.sizeMinSpzBytes,
    smRatio: r.sizeMinRatio,
  })),
);
