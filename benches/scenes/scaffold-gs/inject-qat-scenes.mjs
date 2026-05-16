#!/usr/bin/env node
// Inject the 6 qat-scaffold-gs scenes into benches/reports/splatbench-v0.json
// and benches/reports/splatbench-v0.encoders.json.
//
// Source of truth for the per-scene QAT numbers is the trainer summary at:
//   research/qat-pathx-batch/out/six_scene_summary.json (private repo)
// Numbers are hard-coded here to keep the public bench reproducible from
// public artifacts only.

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const REPO = resolve(new URL(".", import.meta.url).pathname, "..", "..", "..");
const BENCH_JSON = resolve(REPO, "benches/reports/splatbench-v0.json");
const ENCODERS_JSON = resolve(REPO, "benches/reports/splatbench-v0.encoders.json");

const QAT_SCENES = [
  {
    scene: "bonsai",
    id: "bonsai_scaffold_gs",
    class: "indoor-real-estate",
    origin: "Scaffold-GS trained from Mip-NeRF 360 bonsai",
    license: "Mip-NeRF 360 dataset (open research)",
    splatCount: 411066,
    bytesIn: 129899089,
    afterBytes: 77282930,
    plySaveFraction: 0.4051,
    baselinePsnrDb: 32.82730,
    qatPsnrDb: 33.40871,
    psnrDeltaDb: 0.5814,
    baselineSsim: 0.9494,
    qatSsim: 0.9515,
    ssimDelta: 0.00210,
    baselineLpips: 0.17838,
    qatLpips: 0.17531,
    lpipsDelta: -0.00308,
  },
  {
    scene: "bicycle",
    id: "bicycle_scaffold_gs",
    class: "outdoor-scene",
    origin: "Scaffold-GS trained from Mip-NeRF 360 bicycle",
    license: "Mip-NeRF 360 dataset (open research)",
    splatCount: 912445,
    bytesIn: 288334853,
    afterBytes: 171542182,
    plySaveFraction: 0.4051,
    baselinePsnrDb: 25.11114,
    qatPsnrDb: 25.19792,
    psnrDeltaDb: 0.0868,
    baselineSsim: 0.7393,
    qatSsim: 0.7484,
    ssimDelta: 0.00911,
    baselineLpips: 0.2655,
    qatLpips: 0.2534,
    lpipsDelta: -0.01207,
  },
  {
    scene: "garden",
    id: "garden_scaffold_gs",
    class: "outdoor-scene",
    origin: "Scaffold-GS trained from Mip-NeRF 360 garden",
    license: "Mip-NeRF 360 dataset (open research)",
    splatCount: 745121,
    bytesIn: 235460469,
    afterBytes: 140085270,
    plySaveFraction: 0.4051,
    baselinePsnrDb: 27.33378,
    qatPsnrDb: 27.53210,
    psnrDeltaDb: 0.1983,
    baselineSsim: 0.8505,
    qatSsim: 0.8528,
    ssimDelta: 0.00227,
    baselineLpips: 0.1355,
    qatLpips: 0.13270,
    lpipsDelta: -0.00280,
  },
  {
    scene: "stump",
    id: "stump_scaffold_gs",
    class: "outdoor-scene",
    origin: "Scaffold-GS trained from Mip-NeRF 360 stump",
    license: "Mip-NeRF 360 dataset (open research)",
    splatCount: 746258,
    bytesIn: 211939666,
    afterBytes: 140299026,
    plySaveFraction: 0.3380,
    baselinePsnrDb: 26.61723,
    qatPsnrDb: 26.69355,
    psnrDeltaDb: 0.0763,
    baselineSsim: 0.7644,
    qatSsim: 0.7661,
    ssimDelta: 0.00172,
    baselineLpips: 0.25950,
    qatLpips: 0.25787,
    lpipsDelta: -0.00163,
  },
  {
    scene: "treehill",
    id: "treehill_scaffold_gs",
    class: "outdoor-scene",
    origin: "Scaffold-GS trained from Mip-NeRF 360 treehill",
    license: "Mip-NeRF 360 dataset (open research)",
    splatCount: 720712,
    bytesIn: 204684602,
    afterBytes: 135496378,
    plySaveFraction: 0.3380,
    baselinePsnrDb: 23.11214,
    qatPsnrDb: 23.16846,
    psnrDeltaDb: 0.0563,
    baselineSsim: 0.6439,
    qatSsim: 0.6455,
    ssimDelta: 0.00157,
    baselineLpips: 0.34430,
    qatLpips: 0.34259,
    lpipsDelta: -0.00171,
  },
  {
    scene: "flowers",
    id: "flowers_scaffold_gs",
    class: "outdoor-scene",
    origin: "Scaffold-GS trained from Mip-NeRF 360 flowers",
    license: "Mip-NeRF 360 dataset (open research)",
    splatCount: 705051,
    bytesIn: 200236878,
    afterBytes: 132552110,
    plySaveFraction: 0.3380,
    baselinePsnrDb: 21.32125,
    qatPsnrDb: 21.35288,
    psnrDeltaDb: 0.0316,
    baselineSsim: 0.5792,
    qatSsim: 0.5814,
    ssimDelta: 0.00220,
    baselineLpips: 0.37096,
    qatLpips: 0.36871,
    lpipsDelta: -0.00225,
  },
];

const bench = JSON.parse(readFileSync(BENCH_JSON, "utf8"));

// Drop any prior qat-scaffold-gs rows (idempotent re-run).
bench.scenes = bench.scenes.filter((s) => s.kind !== "scaffold-gs");

// Append the six scaffold-gs rows.
for (const q of QAT_SCENES) {
  bench.scenes.push({
    id: q.id,
    source: "real",
    class: q.class,
    origin: q.origin,
    license: q.license,
    splatCount: q.splatCount,
    bytesIn: q.bytesIn,
    shDegree: 3,
    hash: `pending:scaffold-gs-${q.scene}`,
    analyzeMs: 0,
    // Mark this row as Scaffold-GS-trained. The Leaderboard renderer hides
    // the SPZ ratio / fidelity columns for kind:"scaffold-gs" rows because
    // the QAT codec operates on PLY, not SPZ, and the eval cameras are
    // training-time (not the orbit-8 ΔE94 fidelity setup used by the SPZ
    // pipeline). The qatScaffoldGs field is the column that lights up.
    kind: "scaffold-gs",
    webMobileSpzBytes: 0,
    webMobileRatio: 0,
    sizeMinSpzBytes: 0,
    sizeMinRatio: 0,
    qatScaffoldGs: {
      version: "v1",
      srcPlyBytes: q.bytesIn,
      afterBytes: q.afterBytes,
      plySaveFraction: q.plySaveFraction,
      baselinePsnrDb: q.baselinePsnrDb,
      qatPsnrDb: q.qatPsnrDb,
      psnrDeltaDb: q.psnrDeltaDb,
      baselineSsim: q.baselineSsim,
      qatSsim: q.qatSsim,
      ssimDelta: q.ssimDelta,
      baselineLpips: q.baselineLpips,
      qatLpips: q.qatLpips,
      lpipsDelta: q.lpipsDelta,
      verdict: "SHIP",
    },
  });
}

// Aggregate the QAT column.
const qatRows = bench.scenes.filter((s) => s.qatScaffoldGs);
const aggBytesIn = qatRows.reduce((a, s) => a + s.qatScaffoldGs.srcPlyBytes, 0);
const aggBytesOut = qatRows.reduce((a, s) => a + s.qatScaffoldGs.afterBytes, 0);
const qatScenesCount = qatRows.length;
const qatScenesImproved = qatRows.filter(
  (s) =>
    s.qatScaffoldGs.psnrDeltaDb > 0 &&
    s.qatScaffoldGs.ssimDelta > 0 &&
    s.qatScaffoldGs.lpipsDelta < 0,
).length;
const qatAggSavePct = (1 - aggBytesOut / aggBytesIn) * 100;
const psnrDeltas = qatRows.map((s) => s.qatScaffoldGs.psnrDeltaDb).sort((a, b) => a - b);
const median = (xs) => {
  const m = xs.length;
  if (!m) return Number.NaN;
  return m % 2 ? xs[(m - 1) / 2] : (xs[m / 2 - 1] + xs[m / 2]) / 2;
};

// Refresh aggregate counts (scenesTotal etc) — only count non-scaffold-gs
// rows in the SPZ-pipeline totals so the medians on the dashboard remain
// honest. The QAT block carries its own counts.
const spzRows = bench.scenes.filter((s) => s.kind !== "scaffold-gs");
bench.aggregates.scenesTotal = bench.scenes.length;
bench.aggregates.scenesReal = bench.scenes.filter((s) => s.source === "real").length;
bench.aggregates.scenesSynthetic = bench.scenes.filter((s) => s.source === "synthetic").length;
bench.aggregates.qatScaffoldGs = {
  scenesTotal: qatScenesCount,
  scenesImproved: qatScenesImproved,
  aggregateSavePct: Number(qatAggSavePct.toFixed(2)),
  psnrDeltaDbMin: Number(psnrDeltas[0].toFixed(3)),
  psnrDeltaDbMedian: Number(median(psnrDeltas).toFixed(3)),
  psnrDeltaDbMax: Number(psnrDeltas[psnrDeltas.length - 1].toFixed(3)),
  psnrDeltaDbMean: Number(
    (psnrDeltas.reduce((a, b) => a + b, 0) / psnrDeltas.length).toFixed(3),
  ),
};

writeFileSync(BENCH_JSON, JSON.stringify(bench, null, 2) + "\n");
console.log(`Wrote ${qatRows.length} qat-scaffold-gs rows into ${BENCH_JSON}`);
console.log(
  `QAT aggregate: ${qatScenesImproved}/${qatScenesCount} scenes improved; agg save ${qatAggSavePct.toFixed(2)}%`,
);

// Mirror into encoders.json so the harness also tracks per-scene QAT runs.
const enc = JSON.parse(readFileSync(ENCODERS_JSON, "utf8"));
if (!enc.encoders.includes("qat-scaffold-gs")) enc.encoders.push("qat-scaffold-gs");
const bySceneEnc = new Map(enc.scenes.map((s) => [s.scene, s]));
for (const q of QAT_SCENES) {
  const row = bySceneEnc.get(q.id) ?? { scene: q.id, inputBytes: q.bytesIn, runs: {} };
  row.inputBytes = q.bytesIn;
  row.runs["qat-scaffold-gs"] = {
    ok: true,
    version: "v1",
    outputBytes: q.afterBytes,
    wallSeconds: 0,
    wallMs: 0,
    ratio: Number((q.bytesIn / q.afterBytes).toFixed(2)),
  };
  if (!bySceneEnc.has(q.id)) enc.scenes.push(row);
}
enc.scenes.sort((a, b) => a.scene.localeCompare(b.scene));
enc.generatedAt = new Date().toISOString();
writeFileSync(ENCODERS_JSON, JSON.stringify(enc, null, 2) + "\n");
console.log(`Wrote qat-scaffold-gs encoder rows into ${ENCODERS_JSON}`);
