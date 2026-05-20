// Typed loader for benches/reports/splatbench-v0.json.
// Imported at build time only — Astro inlines the JSON into the static bundle,
// so there's no runtime fetch and no server.
//
// The `fidelity` field is optional today; once v0.1.1 emits it per scene
// (ΔE94 / SSIM / PSNR), the leaderboard column lights up automatically.

// Synced from `benches/reports/splatbench-v0.json` by `scripts/sync-data.mjs`
// at build time. Keeps the Astro project self-contained for monorepo deploys.
import raw from "../data/splatbench-v0.json" with { type: "json" };

export type SceneSource = "real" | "synthetic";

/** One preset's fidelity row, as emitted by `benches/splatbench-update.mjs`. */
export interface PresetFidelity {
  meanDeltaE94: number;
  maxDeltaE94: number;
  p95DeltaE94: number;
  meanPixelMatch: number;
  meanSsimLoss: number;
  status: "baseline" | "pass" | "borderline" | "fail";
  passed: boolean;
  /** Splat-aware perceptual metric mean — only present when scored by
   *  `catetus-pro` (proprietary). */
  mlScore?: number;
  mlScoreMax?: number;
  mlScoreVersion?: string;
}

export interface SceneFidelity {
  baseline?: string;
  renderer?: string;
  cameraPath?: string;
  frameSize?: number;
  losslessRepack?: PresetFidelity;
  webMobile?: PresetFidelity;
  sizeMin?: PresetFidelity;
}

export interface SplatBenchScene {
  id: string;
  source: SceneSource;
  class: string;
  origin: string;
  license: string;
  splatCount: number;
  bytesIn: number;
  shDegree: number;
  hash: string;
  analyzeMs: number;
  webMobileSpzBytes: number;
  webMobileRatio: number;
  sizeMinSpzBytes: number;
  sizeMinRatio: number;
  /** Optional — populated once v0.1.1 fidelity benchmark lands. */
  fidelity?: SceneFidelity;
  /** Optional — populated only for scenes scored by catetus-pro
   *  `DifferentiableRepack` (proprietary premium-tier pass). */
  repack?: {
    targetRatio: number;
    splatsIn: number;
    splatsOut: number;
    bytesIn: number;
    bytesOut: number;
    psnrRepackDb: number;
    psnrOpacityPruneDb: number;
    psnrDeltaDb: number;
  };
  /** Optional — when set to "scaffold-gs", the scene is a Scaffold-GS
   *  trained PLY scored only by the qat-scaffold-gs column. The
   *  SPZ-pipeline ratio/fidelity columns are hidden on these rows because
   *  the QAT codec operates on PLY (not SPZ) against training-time eval
   *  cameras (not the orbit-8 ΔE94 fidelity setup). */
  kind?: "scaffold-gs";
  /** Optional — populated for Scaffold-GS PLYs that went through the
   *  qat-scaffold-gs codec. Same scene's FP32 baseline vs QAT'd output,
   *  measured on the source training-time eval cameras. */
  qatScaffoldGs?: {
    version: string;
    srcPlyBytes: number;
    afterBytes: number;
    /** 1 - (afterBytes / srcPlyBytes); save fraction of the PLY. */
    plySaveFraction: number;
    baselinePsnrDb: number;
    qatPsnrDb: number;
    psnrDeltaDb: number;
    baselineSsim: number;
    qatSsim: number;
    ssimDelta: number;
    baselineLpips: number;
    qatLpips: number;
    lpipsDelta: number;
    verdict: "SHIP" | "HOLD" | "FAIL";
  };
}

export interface SplatBenchAggregates {
  scenesTotal: number;
  scenesReal: number;
  scenesSynthetic: number;
  splatCountTotal: number;
  bytesInTotal: number;
  webMobileSpzTotal?: number;
  sizeMinSpzTotal?: number;
  webMobileRatioOverall?: number;
  sizeMinRatioOverall?: number;
  webMobileRatioMin: number;
  webMobileRatioMedian: number;
  webMobileRatioMax: number;
  sizeMinRatioMin: number;
  sizeMinRatioMedian: number;
  sizeMinRatioMax: number;
  fidelityWebMobilePass?: number;
  fidelitySizeMinPass?: number;
  /** Optional — aggregate stats for the qat-scaffold-gs column. */
  qatScaffoldGs?: {
    scenesTotal: number;
    scenesImproved: number;
    aggregateSavePct: number;
    psnrDeltaDbMin: number;
    psnrDeltaDbMedian: number;
    psnrDeltaDbMax: number;
    psnrDeltaDbMean: number;
  };
}

export interface SplatBenchReport {
  schema: string;
  name: string;
  description: string;
  catetusVersion: string;
  runDate: string;
  platform: string;
  preset: string;
  scenes: SplatBenchScene[];
  aggregates: SplatBenchAggregates;
}

// `raw` is the JSON literal — cast through unknown to apply our schema without
// silently dropping unknown fields the upstream report may add later.
export const splatbench: SplatBenchReport = raw as unknown as SplatBenchReport;

export type Preset = "webMobile" | "sizeMin";

export interface PresetView {
  spz: number;
  ratio: number;
}

export function presetView(scene: SplatBenchScene, preset: Preset): PresetView {
  return preset === "webMobile"
    ? { spz: scene.webMobileSpzBytes, ratio: scene.webMobileRatio }
    : { spz: scene.sizeMinSpzBytes, ratio: scene.sizeMinRatio };
}

export function sortedScenes(scenes: SplatBenchScene[], preset: Preset): SplatBenchScene[] {
  return [...scenes].sort((a, b) => presetView(b, preset).ratio - presetView(a, preset).ratio);
}

export function ratioClass(r: number): "ratio-high" | "ratio-mid" | "ratio-low" {
  if (r >= 25) return "ratio-high";
  if (r >= 21) return "ratio-mid";
  return "ratio-low";
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1_048_576) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1_073_741_824) return `${(n / 1_048_576).toFixed(1)} MB`;
  return `${(n / 1_073_741_824).toFixed(2)} GB`;
}

export function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtInt(n: number): string {
  return n.toLocaleString("en-US");
}

/** True if at least one scene in the report has fidelity numbers wired up. */
export function hasAnyFidelity(scenes: SplatBenchScene[]): boolean {
  return scenes.some(
    (s) =>
      s.fidelity?.webMobile?.meanDeltaE94 !== undefined ||
      s.fidelity?.sizeMin?.meanDeltaE94 !== undefined,
  );
}

/** True if any scene carries a catetus-pro ML Score (proprietary column). */
export function hasAnyMlScore(scenes: SplatBenchScene[]): boolean {
  return scenes.some(
    (s) =>
      s.fidelity?.webMobile?.mlScore !== undefined ||
      s.fidelity?.sizeMin?.mlScore !== undefined,
  );
}

/** Return the `PresetFidelity` for the given preset, if present. */
export function fidelityFor(
  scene: SplatBenchScene,
  preset: Preset,
): PresetFidelity | undefined {
  return preset === "webMobile" ? scene.fidelity?.webMobile : scene.fidelity?.sizeMin;
}

/** True if any scene carries a catetus-pro DifferentiableRepack result. */
export function hasAnyRepack(scenes: SplatBenchScene[]): boolean {
  return scenes.some((s) => typeof s.repack?.psnrDeltaDb === "number");
}

/* ---------- third-party encoder comparison ---------- */

/**
 * Shape emitted by `benches/run-encoders.mjs` into
 * `benches/reports/splatbench-v0.encoders.json` (synced into
 * `apps/web/src/data/`).
 */
export interface EncoderRunOk {
  ok: true;
  version: string;
  outputBytes: number;
  wallSeconds: number;
  wallMs: number;
  ratio: number;
}
export interface EncoderRunFail {
  ok: false;
  wallMs: number;
  stderr?: string;
}
export type EncoderRun = EncoderRunOk | EncoderRunFail;

export interface EncoderReport {
  schema: string;
  generatedAt: string;
  encoders: string[];
  scenes: Array<{
    scene: string;
    inputBytes: number;
    runs: Record<string, EncoderRun>;
  }>;
}

import encoderReport from "../data/splatbench-v0.encoders.json" with { type: "json" };
export const encoders: EncoderReport = encoderReport as unknown as EncoderReport;

/** Lookup a third-party encoder's result for a scene, by canonical id. */
export function encoderRunFor(
  sceneId: string,
  encoder: string,
): EncoderRun | undefined {
  const row = encoders.scenes.find((s) => s.scene === sceneId);
  return row?.runs[encoder];
}

/**
 * Median ratio for a given third-party encoder across the scenes it
 * successfully encoded. `Number.NaN` if none ran.
 */
export function encoderMedianRatio(encoder: string): number {
  const ratios = encoders.scenes
    .map((s) => s.runs[encoder])
    .filter((r): r is EncoderRunOk => !!r && r.ok)
    .map((r) => r.ratio)
    .sort((a, b) => a - b);
  if (ratios.length === 0) return Number.NaN;
  const m = ratios.length;
  return m % 2 ? ratios[(m - 1) / 2] : (ratios[m / 2 - 1] + ratios[m / 2]) / 2;
}

/** True if any scene carries qat-scaffold-gs codec results. */
export function hasAnyQat(scenes: SplatBenchScene[]): boolean {
  return scenes.some((s) => !!s.qatScaffoldGs);
}

/** True for rows that are Scaffold-GS-only (SPZ columns should render "-"). */
export function isScaffoldGsRow(scene: SplatBenchScene): boolean {
  return scene.kind === "scaffold-gs";
}
