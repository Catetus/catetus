#!/usr/bin/env node
// SplatBench v3 aggregator.
//
// Reads every benches/timeseries/<commit>/*.json the workflow wrote in this
// run, plus the latest per-cell results from previous commits' timeseries
// dirs, and emits:
//
//   benches/timeseries/<commit>.json   -- flat array of cells for this commit
//                                          (one row per preset×scene pair),
//                                          ready to ingest into the public
//                                          dashboard's time-series store.
//   benches/reports/splatbench-v0.json -- the existing public leaderboard
//                                          file, with each scene's preset
//                                          columns refreshed if the new
//                                          numbers are better-or-equal in
//                                          PSNR (≤0.3 dB drop tolerance).
//
// Idempotent: re-running on the same commit reproduces identical output
// modulo timestamp.
//
// Usage:
//   node benches/splatbench-v3-aggregate.mjs <commit-sha> [--dry-run]
//
// Notes:
// * The "≤0.3 dB drop rule" matches CLAUDE.md project policy for auto-ship.
//   If any cell drops >0.3 dB versus the previous splatbench-v0.json entry,
//   the cell is recorded in the time-series but the leaderboard's old value
//   is kept. The regression alert workflow (separate file) will surface
//   the drop on the PR.
// * Schema preserves the existing splatbench-v0.json shape so the
//   apps/web/src/lib/splatbench.ts loader keeps working unchanged.

import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = dirname(__dirname);
const TS_DIR = join(REPO_ROOT, "benches", "timeseries");
const REPORTS_DIR = join(REPO_ROOT, "benches", "reports");
const PUBLIC_REPORT = join(REPORTS_DIR, "splatbench-v0.json");
const MANIFEST = join(REPO_ROOT, "benches", "scenes", "manifest.json");

const PSNR_DROP_TOLERANCE_DB = 0.3;

function readJson(p) {
  return JSON.parse(readFileSync(p, "utf8"));
}

function writeJson(p, obj) {
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(obj, null, 2) + "\n");
}

function* walkCellsFor(commit) {
  const dir = join(TS_DIR, commit);
  if (!existsSync(dir)) return;
  for (const name of readdirSync(dir)) {
    if (!name.endsWith(".json")) continue;
    yield join(dir, name);
  }
}

function main() {
  const args = process.argv.slice(2);
  const commit = args.find((a) => !a.startsWith("--"));
  const dryRun = args.includes("--dry-run");

  if (!commit) {
    console.error("usage: splatbench-v3-aggregate.mjs <commit-sha> [--dry-run]");
    process.exit(2);
  }

  const manifest = readJson(MANIFEST);
  const presetSet = new Set(manifest.presets);
  const sceneSet = new Set(manifest.scenes.map((s) => s.id));

  const cells = [];
  for (const cellPath of walkCellsFor(commit)) {
    try {
      const cell = readJson(cellPath);
      if (!cell.preset || !cell.scene_id) {
        console.warn(`SKIP malformed cell ${basename(cellPath)}`);
        continue;
      }
      cells.push(cell);
    } catch (e) {
      console.warn(`SKIP unreadable ${basename(cellPath)}: ${e.message}`);
    }
  }

  if (cells.length === 0) {
    console.error(`no cells found under ${TS_DIR}/${commit} — nothing to aggregate`);
    process.exit(0);
  }

  // ---- time-series file (one row per cell) ----
  const tsRows = cells.map((c) => ({
    commit,
    timestamp: c.timestamp,
    preset: c.preset,
    scene: c.scene_id,
    ok: c.result?.ok === true,
    ply_save_pct: c.result?.ply_save_pct ?? null,
    delta_psnr_db: c.result?.delta_psnr_db ?? null,
    delta_ssim: c.result?.delta_ssim ?? null,
    delta_lpips: c.result?.delta_lpips ?? null,
    wall_secs: c.result?.wall_secs ?? null,
    modal_cost_cents: c.result?.modal_cost_cents ?? null,
    output_url: c.result?.output_url ?? null,
    error: c.result?.ok === false ? (c.result?.error ?? "unknown") : null,
  }));
  const tsFile = join(TS_DIR, `${commit}.json`);
  if (!dryRun) {
    writeJson(tsFile, {
      schema: "splatforge.splatbench-v3.timeseries/0.1",
      commit,
      generatedAt: new Date().toISOString(),
      cellCount: tsRows.length,
      presets: [...new Set(tsRows.map((r) => r.preset))],
      scenes: [...new Set(tsRows.map((r) => r.scene))],
      rows: tsRows,
    });
    console.log(`wrote ${tsRows.length} rows -> benches/timeseries/${commit}.json`);
  } else {
    console.log(`[dry-run] would write ${tsRows.length} rows to ${tsFile}`);
  }

  // ---- splatbench-v0.json refresh ----
  // We are *additive* in v3 first session: the existing manually-curated
  // splatbench-v0.json keeps its structure; we add a top-level
  // `continuousBench` block alongside `scenes` + `aggregates`, populated
  // from the latest cells per (preset, scene). This way the existing
  // /bench leaderboard page renders unchanged while /bench/history
  // (new dashboard) reads from `continuousBench` + the time-series
  // store.
  if (!existsSync(PUBLIC_REPORT)) {
    console.warn(`no existing ${PUBLIC_REPORT} — skipping leaderboard merge`);
    return;
  }
  const report = readJson(PUBLIC_REPORT);
  const prevCont = report.continuousBench ?? { commits: [] };

  // Per-cell latest with drop-tolerance gate.
  const cellMap = {}; // `${preset}__${scene}` -> best-or-latest cell row
  for (const row of tsRows) {
    if (!presetSet.has(row.preset) || !sceneSet.has(row.scene)) continue;
    const key = `${row.preset}__${row.scene}`;
    cellMap[key] = row;
  }

  // Apply ≤0.3 dB drop guard versus the previous commit's recorded value.
  const prevCells = prevCont.latestCells ?? {};
  const acceptedCells = { ...prevCells };
  const rejectedCells = [];
  for (const [key, row] of Object.entries(cellMap)) {
    const prev = prevCells[key];
    if (!row.ok) {
      // Failed cell — keep prior value if any; record the failure.
      rejectedCells.push({ key, reason: row.error ?? "failed", commit });
      continue;
    }
    if (
      prev &&
      typeof prev.delta_psnr_db === "number" &&
      typeof row.delta_psnr_db === "number" &&
      row.delta_psnr_db < prev.delta_psnr_db - PSNR_DROP_TOLERANCE_DB
    ) {
      rejectedCells.push({
        key,
        reason: `psnr-drop:${(prev.delta_psnr_db - row.delta_psnr_db).toFixed(3)}dB`,
        prevCommit: prev.commit,
        commit,
      });
      // Keep `prev` in acceptedCells.
      continue;
    }
    acceptedCells[key] = row;
  }

  report.continuousBench = {
    schema: "splatforge.splatbench-v3.cont/0.1",
    description:
      "Continuous-bench rollup of every (preset, scene) cell. " +
      "Updated by .github/workflows/splatbench-v3.yml on every commit to main. " +
      "`latestCells` is the best-passing value per cell under the ≤0.3 dB PSNR drop rule; " +
      "`commits` is the rolling list of commits that produced at least one cell. " +
      "Full per-commit timeseries lives in benches/timeseries/<commit>.json.",
    latestCells: acceptedCells,
    lastCommit: commit,
    lastCommitAt: new Date().toISOString(),
    commits: [
      ...(prevCont.commits ?? []).filter((c) => c.commit !== commit),
      {
        commit,
        timestamp: new Date().toISOString(),
        cellCount: tsRows.length,
        passCount: tsRows.filter((r) => r.ok).length,
        rejectedCount: rejectedCells.length,
      },
    ].slice(-90), // keep 90 most recent commits in the public file
  };

  if (rejectedCells.length) {
    report.continuousBench.lastRejections = rejectedCells;
    console.warn(
      `rejected ${rejectedCells.length} cells (>0.3 dB drop or failure): kept previous values`,
    );
  }

  if (!dryRun) {
    writeJson(PUBLIC_REPORT, report);
    console.log(
      `refreshed continuousBench in ${PUBLIC_REPORT.replace(REPO_ROOT + "/", "")} ` +
        `(accepted ${Object.keys(acceptedCells).length} cells, rejected ${rejectedCells.length})`,
    );
  } else {
    console.log(`[dry-run] would refresh continuousBench in splatbench-v0.json`);
  }
}

main();
