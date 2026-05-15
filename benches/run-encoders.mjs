#!/usr/bin/env node
// Bench harness for third-party encoders registered under
// `benches/encoders/<name>/`. Per scene × per encoder, runs the encoder's
// `run.sh`, collects `meta.json`, and writes a merged comparison report
// into `benches/reports/splatbench-v0.encoders.json`.
//
// Today the harness is **compression-only** — fidelity scoring against a
// non-SplatForge output format requires a viewer that can decode it. SOG
// support lands when splatforge-spz's SOG reader path is wired in; for now
// we publish ratio + encode-time so the /bench page has something concrete
// from each competitor.
import { execSync, spawnSync } from "node:child_process";
import { readdirSync, readFileSync, writeFileSync, mkdirSync, statSync, existsSync } from "node:fs";
import { join, basename, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ENCODERS_DIR = join(here, "encoders");
const SCENES_DIR = join(here, "scenes");
const REPORT_DIR = join(here, "reports");
const OUT_FILE = join(REPORT_DIR, "splatbench-v0.encoders.json");

mkdirSync(REPORT_DIR, { recursive: true });

const encoders = readdirSync(ENCODERS_DIR, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => ({
    name: d.name,
    dir: join(ENCODERS_DIR, d.name),
    runScript: join(ENCODERS_DIR, d.name, "run.sh"),
  }))
  .filter((e) => existsSync(e.runScript));

const scenes = readdirSync(SCENES_DIR)
  .filter((f) => f.endsWith(".ply"))
  .map((f) => ({
    name: basename(f, ".ply"),
    path: join(SCENES_DIR, f),
  }));

const SCENE_FILTER = process.env.SCENES ? new Set(process.env.SCENES.split(",")) : null;
const ENCODER_FILTER = process.env.ENCODERS ? new Set(process.env.ENCODERS.split(",")) : null;

console.log(`Encoders: ${encoders.map((e) => e.name).join(", ") || "(none)"}`);
console.log(`Scenes:   ${scenes.length}`);

// Merge with any prior report so a partial sweep (`SCENES=foo,bar`) does
// not drop scenes captured by an earlier invocation. The harness is
// otherwise idempotent — a re-run for a given (scene, encoder) replaces
// just that row in-place.
let prior = { schema: "", encoders: [], scenes: [] };
if (existsSync(OUT_FILE)) {
  try {
    prior = JSON.parse(readFileSync(OUT_FILE, "utf8"));
  } catch (e) {
    console.warn(`[bench] could not parse ${OUT_FILE}; starting fresh: ${e.message}`);
  }
}
const priorByScene = new Map((prior.scenes ?? []).map((s) => [s.scene, s]));

const report = {
  schema: "splatforge.splatbench.encoders/0.1",
  generatedAt: new Date().toISOString(),
  encoders: Array.from(
    new Set([...(prior.encoders ?? []), ...encoders.map((e) => e.name)]),
  ),
  scenes: [],
};

const tmpRoot = join("/Users/montabano1/Desktop/SplatForge/.tmp", "bench-encoders");
mkdirSync(tmpRoot, { recursive: true });

for (const scene of scenes) {
  if (SCENE_FILTER && !SCENE_FILTER.has(scene.name)) continue;
  const inputBytes = statSync(scene.path).size;
  // Seed the row from prior data so unfiltered encoders' results survive
  // a targeted re-run (e.g. `ENCODERS=splat-transform` shouldn't drop
  // SOG or CodecGS-Lite numbers captured in a different sweep).
  const seed = priorByScene.get(scene.name);
  const sceneRow = {
    scene: scene.name,
    inputBytes,
    runs: { ...(seed?.runs ?? {}) },
  };

  for (const enc of encoders) {
    if (ENCODER_FILTER && !ENCODER_FILTER.has(enc.name)) continue;
    const outDir = join(tmpRoot, `${enc.name}__${scene.name}`);
    mkdirSync(outDir, { recursive: true });
    const t0 = Date.now();
    const res = spawnSync("bash", [enc.runScript, scene.path, outDir], {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env },
    });
    const wallMs = Date.now() - t0;
    if (res.status !== 0) {
      sceneRow.runs[enc.name] = {
        ok: false,
        wallMs,
        stderr: res.stderr?.toString().slice(0, 1024) ?? "",
      };
      console.error(`FAIL ${enc.name} on ${scene.name} (${wallMs}ms): ${res.stderr?.toString().slice(0, 200)}`);
      continue;
    }
    const metaPath = join(outDir, "meta.json");
    let meta = null;
    try {
      meta = JSON.parse(readFileSync(metaPath, "utf8"));
    } catch (e) {
      sceneRow.runs[enc.name] = { ok: false, wallMs, stderr: `meta.json unreadable: ${e.message}` };
      continue;
    }
    const ratio = inputBytes / meta.output_bytes;
    sceneRow.runs[enc.name] = {
      ok: true,
      version: meta.version,
      outputBytes: meta.output_bytes,
      wallSeconds: meta.wall_seconds,
      wallMs,
      ratio: Number(ratio.toFixed(2)),
    };
    console.log(`OK   ${enc.name} on ${scene.name}: ${ratio.toFixed(2)}x (${meta.output_bytes} bytes)`);
  }

  report.scenes.push(sceneRow);
  priorByScene.delete(scene.name);
}

// Preserve any scenes that weren't in this sweep (filter or absent).
for (const leftover of priorByScene.values()) {
  report.scenes.push(leftover);
}
// Stable sort by scene name so commits diff sensibly.
report.scenes.sort((a, b) => a.scene.localeCompare(b.scene));

writeFileSync(OUT_FILE, JSON.stringify(report, null, 2));
console.log(`\nWrote ${OUT_FILE}`);
