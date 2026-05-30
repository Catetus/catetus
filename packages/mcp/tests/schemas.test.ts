// Smoke test: every tool's inputSchema accepts a well-formed example AND rejects bad input.
// Also asserts every tool has all 4 annotations + outputSchema (per ARCHITECTURE.md).

import { describe, it, expect } from "vitest";
import { z } from "zod";

import { analyzeMeta, analyzeInput } from "../src/tools/analyze.js";
import { listPresetsMeta, listPresetsInput } from "../src/tools/list-presets.js";
import { listScenesMeta, listScenesInput } from "../src/tools/list-scenes.js";
import { getSceneMeta, getSceneInput } from "../src/tools/get-scene.js";
import { optimizeMeta, optimizeInput } from "../src/tools/optimize.js";
import { compareMeta, compareInput } from "../src/tools/compare.js";
import { listCompetitorCodecsMeta, listCompetitorCodecsInput } from "../src/tools/list-competitor-codecs.js";
import { validatePipelineMeta, validatePipelineInput } from "../src/tools/validate-pipeline.js";
import { encodeMeta, encodeInput } from "../src/tools/encode.js";
import { scoreFidelityMeta, scoreFidelityInput } from "../src/tools/score-fidelity.js";
import { repackMeta, repackInput } from "../src/tools/repack.js";
import { predictQualityMeta, predictQualityInput } from "../src/tools/predict-quality.js";
import { recommendPresetMeta, recommendPresetInput } from "../src/tools/recommend-preset.js";
import { batchJobsMeta, batchJobsInput } from "../src/tools/batch-jobs.js";
import { listJobsMeta, listJobsInput } from "../src/tools/list-jobs.js";

import { resolveTier, isToolVisible, TOOL_TIER } from "../src/auth.js";
import { encodeCursor, decodeCursor, paginate } from "../src/pagination.js";
import { errorResult, okResult, ERROR_CODES } from "../src/errors.js";

const sampleSplatRef = { kind: "url" as const, url: "https://example.com/scene.ply" };

const TOOL_TABLE: Array<[string, { description: string; inputSchema: Record<string, z.ZodTypeAny>; outputSchema?: unknown; annotations?: Record<string, unknown> }, Record<string, unknown>, Record<string, unknown>]> = [
  ["analyze", analyzeMeta, { input: sampleSplatRef }, { input: 42 }],
  ["list_presets", listPresetsMeta, {}, { tier_filter: "garbage" }],
  ["list_scenes", listScenesMeta, {}, { corpus: "lol" }],
  ["get_scene", getSceneMeta, { scene_id: "bonsai" }, {}],
  ["optimize", optimizeMeta, { input: sampleSplatRef, preset: "web-mobile" }, { input: sampleSplatRef, preset: "v52-quality" }],
  ["compare", compareMeta, { before: sampleSplatRef, after: sampleSplatRef }, { before: sampleSplatRef }],
  ["list_competitor_codecs", listCompetitorCodecsMeta, {}, { limit: 999 }],
  ["validate_pipeline", validatePipelineMeta, { input: sampleSplatRef }, { input: 42 }],
  ["encode", encodeMeta, { input: sampleSplatRef }, { input: sampleSplatRef, target: "bogus" }],
  ["score_fidelity", scoreFidelityMeta, { before: sampleSplatRef, after: sampleSplatRef }, { before: sampleSplatRef }],
  ["repack", repackMeta, { input: sampleSplatRef }, { input: sampleSplatRef, targetRatio: 99 }],
  ["predict_quality", predictQualityMeta, { input: sampleSplatRef, preset: "web-mobile" }, { input: sampleSplatRef, preset: "made-up" }],
  ["recommend_preset", recommendPresetMeta, { input: sampleSplatRef, constraints: { targetDevice: "desktop-web", allowPaid: true } }, { constraints: {} }],
  ["batch_jobs", batchJobsMeta, { jobs: [{ input: sampleSplatRef, target: "sog" }] }, { jobs: [] }],
  ["list_jobs", listJobsMeta, {}, { limit: 1000 }],
];

// Bypass: a few tools use loose inputs ({}) that pass with defaults; the negative test
// for those uses a clearly-bad value chosen above.

describe("annotations + metadata", () => {
  for (const [name, meta] of TOOL_TABLE) {
    it(`${name}: has title, description, all 4 annotations`, () => {
      expect(meta.description).toBeTruthy();
      const a = (meta as { annotations?: Record<string, unknown> }).annotations ?? {};
      expect(typeof a.readOnlyHint).toBe("boolean");
      expect(typeof a.destructiveHint).toBe("boolean");
      expect(typeof a.idempotentHint).toBe("boolean");
      expect(typeof a.openWorldHint).toBe("boolean");
      expect(a.title || (meta as { title?: string }).title).toBeTruthy();
    });
    it(`${name}: has outputSchema`, () => {
      expect((meta as { outputSchema?: unknown }).outputSchema).toBeTruthy();
    });
  }
});

describe("input schemas — acceptance", () => {
  for (const [name, _meta, good] of TOOL_TABLE) {
    it(`${name}: accepts well-formed input`, () => {
      const schemaShape = (_meta as { inputSchema: Record<string, z.ZodTypeAny> }).inputSchema;
      const result = z.object(schemaShape).safeParse(good);
      if (!result.success) {
        console.error(`${name} accept failed:`, result.error.errors);
      }
      expect(result.success).toBe(true);
    });
  }
});

describe("input schemas — rejection of malformed inputs", () => {
  // Map of tool → input that MUST fail. Tools whose minimal schema is too permissive use
  // a value chosen to violate at least one Zod constraint.
  const REJECT_CASES: Record<string, unknown> = {
    analyze: { input: 42 },
    list_presets: { tier_filter: "garbage" },
    list_scenes: { corpus: "made-up-corpus" },
    get_scene: {},
    optimize: { input: sampleSplatRef, preset: "v52-quality" }, // paid preset on free tool
    compare: { before: sampleSplatRef },
    list_competitor_codecs: { limit: 999 },
    validate_pipeline: { input: 42 },
    encode: { input: sampleSplatRef, target: "bogus" },
    score_fidelity: { before: sampleSplatRef },
    repack: { input: sampleSplatRef, targetRatio: 99 },
    predict_quality: { input: sampleSplatRef, preset: "made-up" },
    recommend_preset: { constraints: {} },
    batch_jobs: { jobs: [] },
    list_jobs: { limit: 1000 },
  };
  for (const [name, meta] of TOOL_TABLE) {
    it(`${name}: rejects malformed input`, () => {
      const shape = (meta as { inputSchema: Record<string, z.ZodTypeAny> }).inputSchema;
      const result = z.object(shape).safeParse(REJECT_CASES[name]);
      expect(result.success).toBe(false);
    });
  }
});

describe("auth/tier", () => {
  it("public tier when no key", () => {
    const t = resolveTier({}, { ...process.env, CATETUS_API_KEY: undefined });
    expect(t.tier).toBe("public");
  });
  it("paid tier via env", () => {
    const t = resolveTier({}, { CATETUS_API_KEY: "cat_live_abc" } as NodeJS.ProcessEnv);
    expect(t.tier).toBe("paid");
    expect(t.apiKey).toBe("cat_live_abc");
  });
  it("paid tier via Authorization header", () => {
    const t = resolveTier({ authorization: "Bearer cat_live_xyz" }, {} as NodeJS.ProcessEnv);
    expect(t.tier).toBe("paid");
    expect(t.apiKey).toBe("cat_live_xyz");
  });
  it("hides paid tools from public tier", () => {
    const pub = { tier: "public" as const };
    expect(isToolVisible("analyze", pub)).toBe(true);
    expect(isToolVisible("encode", pub)).toBe(false);
    expect(isToolVisible("score_fidelity", pub)).toBe(false);
  });
  it("shows all tools to paid tier", () => {
    const paid = { tier: "paid" as const, scopes: ["encode", "score_fidelity", "repack", "predict", "batch"] };
    for (const name of Object.keys(TOOL_TIER)) {
      expect(isToolVisible(name, paid)).toBe(true);
    }
  });
});

describe("pagination", () => {
  it("roundtrips a cursor", () => {
    const tok = encodeCursor({ offset: 5, filter_hash: "abc123" });
    const back = decodeCursor(tok);
    expect(back).toEqual({ offset: 5, filter_hash: "abc123" });
  });
  it("rejects garbage cursors", () => {
    expect(decodeCursor("not-a-cursor")).toBeNull();
  });
  it("paginates an array", () => {
    const items = Array.from({ length: 25 }, (_, i) => i);
    const r1 = paginate(items, undefined, 10, { f: "x" });
    expect(r1.ok).toBe(true);
    if (r1.ok) {
      expect(r1.page).toHaveLength(10);
      expect(r1.next_cursor).toBeDefined();
      const r2 = paginate(items, r1.next_cursor!, 10, { f: "x" });
      expect(r2.ok).toBe(true);
      if (r2.ok) {
        expect(r2.page).toHaveLength(10);
        const r3 = paginate(items, r2.next_cursor!, 10, { f: "x" });
        if (r3.ok) {
          expect(r3.page).toHaveLength(5);
          expect(r3.next_cursor).toBeUndefined();
        }
      }
    }
  });
  it("rejects cursor with wrong filter_hash", () => {
    const items = Array.from({ length: 25 }, (_, i) => i);
    const r1 = paginate(items, undefined, 10, { f: "x" });
    if (r1.ok && r1.next_cursor) {
      const r2 = paginate(items, r1.next_cursor, 10, { f: "DIFFERENT" });
      expect(r2.ok).toBe(false);
    }
  });
});

describe("errors", () => {
  it("envelope shape", () => {
    const e = errorResult("scene_not_found", "no such scene", { hint: "use list_scenes" });
    expect(e.isError).toBe(true);
    expect(e.content[0].type).toBe("text");
    const env = JSON.parse(e.content[0].text);
    expect(env.code).toBe("scene_not_found");
    expect(env.message).toBe("no such scene");
    expect(env.hint).toBe("use list_scenes");
    expect(env.docs_url).toContain("scene_not_found");
    expect(env.server_version).toBeTruthy();
    expect(e.structuredContent).toEqual(env);
  });
  it("ok result mirrors payload", () => {
    const r = okResult({ a: 1, b: "two" });
    expect(r.structuredContent).toEqual({ a: 1, b: "two" });
    expect(JSON.parse(r.content[0].text)).toEqual({ a: 1, b: "two" });
  });
  it("error code registry has 29+ codes", () => {
    expect(ERROR_CODES.length).toBeGreaterThanOrEqual(29);
  });
});
