#!/usr/bin/env node
/**
 * Merge `benches/reports/splatbench-v2.json` (output of the v2 ingest
 * harness, `apps/bench-ingest`) into the published leaderboard artifact at
 * `benches/reports/splatbench-v0.json`.
 *
 * Why two files: the v0 JSON carries hand-annotated columns the harness
 * never computes (splatforge-pro ML scores, DifferentiableRepack rows,
 * `fidelity` ΔE94 blocks emitted by `splatforge fidelity`). The merge here
 * is strictly additive on the ratio side: v0 wins on every annotation
 * field, v2 wins on every measurement field.
 *
 * Idempotent. Safe to run on a v2 file that has zero new rows (no-op).
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(here, "..");
const V0_PATH = resolve(REPO_ROOT, "benches", "reports", "splatbench-v0.json");
const V2_PATH = resolve(REPO_ROOT, "benches", "reports", "splatbench-v2.json");

if (!existsSync(V0_PATH)) {
  console.error(`[sync-splatbench] missing ${V0_PATH}`);
  process.exit(1);
}
if (!existsSync(V2_PATH)) {
  console.error(
    `[sync-splatbench] no ${V2_PATH} yet — run \`python -m bench_ingest.cli batch\` first. Nothing to do.`,
  );
  process.exit(0);
}

const v0 = JSON.parse(readFileSync(V0_PATH, "utf8"));
const v2 = JSON.parse(readFileSync(V2_PATH, "utf8"));

const v0Index = new Map(v0.scenes.map((s) => [s.id, s]));
let merged = 0;
let added = 0;

for (const row of v2.rows ?? []) {
  const prior = v0Index.get(row.id);
  if (prior) {
    // Update measurement fields in place; keep everything else
    // (license, origin, fidelity, repack, ML scores, …) untouched.
    if (row.bytesIn) prior.bytesIn = row.bytesIn;
    if (row.splatCount) prior.splatCount = row.splatCount;
    if (row.shDegree) prior.shDegree = row.shDegree;
    if (row.hash) prior.hash = row.hash;
    if (row.analyzeMs) prior.analyzeMs = row.analyzeMs;
    if (row.webMobileRatio) {
      prior.webMobileSpzBytes = row.webMobileSpzBytes;
      prior.webMobileRatio = row.webMobileRatio;
    }
    if (row.sizeMinRatio) {
      prior.sizeMinSpzBytes = row.sizeMinSpzBytes;
      prior.sizeMinRatio = row.sizeMinRatio;
    }
    if (row.presetRuns) {
      prior.presetRuns = { ...(prior.presetRuns ?? {}), ...row.presetRuns };
    }
    merged++;
  } else {
    // New scene — append. Drop the v2-only `presetRuns` mirrors that the
    // v0 schema doesn't know about, but keep them on the new row for
    // forward-compat with the v2-aware Leaderboard UI.
    v0.scenes.push({
      id: row.id,
      source: row.source ?? "real",
      class: row.class ?? "real-scene",
      origin: row.origin ?? "",
      license: row.license ?? "",
      splatCount: row.splatCount ?? 0,
      bytesIn: row.bytesIn ?? 0,
      shDegree: row.shDegree ?? 3,
      hash: row.hash ?? "",
      analyzeMs: row.analyzeMs ?? 0,
      webMobileSpzBytes: row.webMobileSpzBytes ?? 0,
      webMobileRatio: row.webMobileRatio ?? 0,
      sizeMinSpzBytes: row.sizeMinSpzBytes ?? 0,
      sizeMinRatio: row.sizeMinRatio ?? 0,
      presetRuns: row.presetRuns ?? {},
    });
    added++;
  }
}

// Recompute aggregates from the merged scene list. We only touch the
// numeric ratio aggregates; the structural counts (scenesTotal, …) too.
const median = (xs) => {
  const ys = [...xs].sort((a, b) => a - b);
  if (ys.length === 0) return 0;
  const mid = Math.floor(ys.length / 2);
  return ys.length % 2 ? ys[mid] : (ys[mid - 1] + ys[mid]) / 2;
};

const wm = v0.scenes.map((s) => s.webMobileRatio).filter((r) => r > 0);
const sm = v0.scenes.map((s) => s.sizeMinRatio).filter((r) => r > 0);
const real = v0.scenes.filter((s) => s.source === "real").length;
const synth = v0.scenes.filter((s) => s.source === "synthetic").length;
const bytesInTotal = v0.scenes.reduce((a, s) => a + (s.bytesIn || 0), 0);
const wmSpz = v0.scenes.reduce((a, s) => a + (s.webMobileSpzBytes || 0), 0);
const smSpz = v0.scenes.reduce((a, s) => a + (s.sizeMinSpzBytes || 0), 0);

v0.aggregates = {
  ...v0.aggregates,
  scenesTotal: v0.scenes.length,
  scenesReal: real,
  scenesSynthetic: synth,
  splatCountTotal: v0.scenes.reduce((a, s) => a + (s.splatCount || 0), 0),
  bytesInTotal,
  webMobileSpzTotal: wmSpz,
  sizeMinSpzTotal: smSpz,
  webMobileRatioOverall: wmSpz ? Number((bytesInTotal / wmSpz).toFixed(2)) : undefined,
  sizeMinRatioOverall: smSpz ? Number((bytesInTotal / smSpz).toFixed(2)) : undefined,
  webMobileRatioMin: wm.length ? Math.min(...wm) : 0,
  webMobileRatioMedian: Number(median(wm).toFixed(2)),
  webMobileRatioMax: wm.length ? Math.max(...wm) : 0,
  sizeMinRatioMin: sm.length ? Math.min(...sm) : 0,
  sizeMinRatioMedian: Number(median(sm).toFixed(2)),
  sizeMinRatioMax: sm.length ? Math.max(...sm) : 0,
};

writeFileSync(V0_PATH, JSON.stringify(v0, null, 2) + "\n");
console.error(
  `[sync-splatbench] merged ${merged} updated + ${added} new rows into ${V0_PATH}`,
);
