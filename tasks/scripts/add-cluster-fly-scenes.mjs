#!/usr/bin/env node
// Idempotently add the 5 cluster_fly LOD entries to benches/reports/splatbench-v0.json.
// Catetus SPZ bytes computed via `catetus optimize --compress spz` on 2026-05-15.
// Aggregates are recomputed by benches/splatbench-update.mjs the next time fidelity
// runs; here we just write the scene rows with what we measured.
import { readFileSync, writeFileSync } from 'node:fs';

const PATH = '/Users/montabano1/Desktop/Catetus/.wt-cluster-fly/benches/reports/splatbench-v0.json';
const sb = JSON.parse(readFileSync(PATH, 'utf8'));

const lods = [
  { id: 'cluster_fly_s',   splatCount: 25627,   bytesIn: 6049505,   hash: '32662dbed23030183db2dce9b853895bdd0bf5fff5031b449f09e05c7f6407b5', wm: 283616,   sm: 249844 },
  { id: 'cluster_fly_m',   splatCount: 145617,  bytesIn: 34367146,  hash: '6db2c62dd73552f72e18b230d918d57eff628614b9697935ae64835392c535bb', wm: 1436072,  sm: 1128116 },
  { id: 'cluster_fly_l',   splatCount: 301958,  bytesIn: 71263622,  hash: '30c97b5a2c15947b89a58b07ed876cf74da7c1846b5984bf695b3874296bd9e6', wm: 2865976,  sm: 2157472 },
  { id: 'cluster_fly_xl',  splatCount: 624180,  bytesIn: 147308014, hash: 'fec53be4967854d57cede30e986ca7646ed748e2fffeff8252c0e4b805ee07a8', wm: 5697952,  sm: 4126628 },
  { id: 'cluster_fly_xxl', splatCount: 3506799, bytesIn: 827606099, hash: 'c5b81afd176f4d94799ec2dc6feb44963949e1b23d2590bd8549fc52da301fee', wm: 26518932, sm: 13876788 },
];

const round = (n, p = 2) => +n.toFixed(p);

function mkRow(l) {
  return {
    id: l.id,
    source: 'real',
    class: 'indoor-close-up',
    origin: 'https://www.danybittel.ch/ (cluster.fly LOD ladder)',
    license: 'CC-BY-4.0 (Dany Bittel)',
    splatCount: l.splatCount,
    bytesIn: l.bytesIn,
    shDegree: 3,
    hash: `blake3:${l.hash}`,
    analyzeMs: 0,
    webMobileSpzBytes: l.wm,
    webMobileRatio: round(l.bytesIn / l.wm),
    sizeMinSpzBytes: l.sm,
    sizeMinRatio: round(l.bytesIn / l.sm),
  };
}

// Replace-or-append by id, preserving order of existing scenes.
const byId = new Map(sb.scenes.map((s, i) => [s.id, i]));
for (const l of lods) {
  const row = mkRow(l);
  if (byId.has(l.id)) {
    sb.scenes[byId.get(l.id)] = { ...sb.scenes[byId.get(l.id)], ...row };
  } else {
    sb.scenes.push(row);
  }
}

// Recompute aggregates here directly so the v0.json stays internally consistent
// even before the next fidelity-runner pass (which would otherwise overwrite
// it via splatbench-update.mjs).
const wmRatios = sb.scenes.map(s => s.webMobileRatio).filter(Boolean).sort((a, b) => a - b);
const smRatios = sb.scenes.map(s => s.sizeMinRatio).filter(Boolean).sort((a, b) => a - b);
const median = (arr) => {
  if (!arr.length) return 0;
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[m] : (arr[m - 1] + arr[m]) / 2;
};
const scenesReal = sb.scenes.filter(s => s.source === 'real').length;
const scenesSynthetic = sb.scenes.filter(s => s.source === 'synthetic').length;
const splatTotal = sb.scenes.reduce((a, s) => a + (s.splatCount || 0), 0);
const bytesInTotal = sb.scenes.reduce((a, s) => a + (s.bytesIn || 0), 0);
const webMobileSpzTotal = sb.scenes.reduce((a, s) => a + (s.webMobileSpzBytes || 0), 0);
const sizeMinSpzTotal = sb.scenes.reduce((a, s) => a + (s.sizeMinSpzBytes || 0), 0);
const wmPass = sb.scenes.filter(s => s.fidelity?.webMobile?.status && s.fidelity.webMobile.status !== 'fail').length;
const smPass = sb.scenes.filter(s => s.fidelity?.sizeMin?.status && s.fidelity.sizeMin.status !== 'fail').length;

sb.aggregates = {
  ...sb.aggregates,
  scenesTotal: sb.scenes.length,
  scenesReal,
  scenesSynthetic,
  splatCountTotal: splatTotal,
  bytesInTotal,
  webMobileSpzTotal,
  sizeMinSpzTotal,
  webMobileRatioOverall: round(bytesInTotal / Math.max(1, webMobileSpzTotal)),
  sizeMinRatioOverall: round(bytesInTotal / Math.max(1, sizeMinSpzTotal)),
  webMobileRatioMin: round(wmRatios[0] || 0),
  webMobileRatioMedian: round(median(wmRatios)),
  webMobileRatioMax: round(wmRatios[wmRatios.length - 1] || 0),
  sizeMinRatioMin: round(smRatios[0] || 0),
  sizeMinRatioMedian: round(median(smRatios)),
  sizeMinRatioMax: round(smRatios[smRatios.length - 1] || 0),
  fidelityWebMobilePass: wmPass,
  fidelitySizeMinPass: smPass,
};

writeFileSync(PATH, JSON.stringify(sb, null, 2) + '\n');
console.log(`Wrote ${sb.scenes.length} scenes (real=${scenesReal}, synthetic=${scenesSynthetic})`);
console.log(`Aggregates: ${JSON.stringify(sb.aggregates, null, 2)}`);
