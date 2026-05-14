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

export interface SceneFidelity {
  /** CIE ΔE94 mean (lower is better). */
  deltaE94?: number;
  /** Structural similarity index, 0..1 (higher is better). */
  ssim?: number;
  /** Peak signal-to-noise ratio, dB (higher is better). */
  psnr?: number;
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
}

export interface SplatBenchAggregates {
  scenesTotal: number;
  scenesReal: number;
  scenesSynthetic: number;
  splatCountTotal: number;
  bytesInTotal: number;
  webMobileRatioMin: number;
  webMobileRatioMedian: number;
  webMobileRatioMax: number;
  sizeMinRatioMin: number;
  sizeMinRatioMedian: number;
  sizeMinRatioMax: number;
}

export interface SplatBenchReport {
  schema: string;
  name: string;
  description: string;
  splatforgeVersion: string;
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
      s.fidelity !== undefined &&
      (s.fidelity.deltaE94 !== undefined ||
        s.fidelity.ssim !== undefined ||
        s.fidelity.psnr !== undefined),
  );
}
