// Shared server-core: builds an McpServer with all tools/prompts registered and tier-filtered.
// Transports (stdio/http) wrap this with their own connect logic.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CATETUS_MCP_NAME, CATETUS_MCP_VERSION } from "./version.js";
import { type TierContext, isToolVisible } from "./auth.js";

// Tools
import { analyzeMeta, makeAnalyzeHandler } from "./tools/analyze.js";
import { listPresetsMeta, makeListPresetsHandler } from "./tools/list-presets.js";
import { listScenesMeta, makeListScenesHandler } from "./tools/list-scenes.js";
import { getSceneMeta, makeGetSceneHandler } from "./tools/get-scene.js";
import { optimizeMeta, makeOptimizeHandler } from "./tools/optimize.js";
import { compareMeta, makeCompareHandler } from "./tools/compare.js";
import { listCompetitorCodecsMeta, makeListCompetitorCodecsHandler } from "./tools/list-competitor-codecs.js";
import { validatePipelineMeta, makeValidatePipelineHandler } from "./tools/validate-pipeline.js";
import { encodeMeta, makeEncodeHandler } from "./tools/encode.js";
import { scoreFidelityMeta, makeScoreFidelityHandler } from "./tools/score-fidelity.js";
import { repackMeta, makeRepackHandler } from "./tools/repack.js";
import { predictQualityMeta, makePredictQualityHandler } from "./tools/predict-quality.js";
import { recommendPresetMeta, makeRecommendPresetHandler } from "./tools/recommend-preset.js";
import { batchJobsMeta, makeBatchJobsHandler } from "./tools/batch-jobs.js";
import { listJobsMeta, makeListJobsHandler } from "./tools/list-jobs.js";

// Prompts
import { compressForProductPageMeta, compressForProductPagePrompt } from "./prompts/compress-for-product-page.js";
import { compareAgainstSogMeta, compareAgainstSogPrompt } from "./prompts/compare-against-sog.js";
import { pickBestPresetMeta, pickBestPresetPrompt } from "./prompts/pick-best-preset.js";
import { auditExistingOutputsMeta, auditExistingOutputsPrompt } from "./prompts/audit-existing-outputs.js";

export const INSTRUCTIONS = `You have access to the Catetus 3D Gaussian Splatting toolkit.

Recommended workflow for "compress this scene":
  1. analyze(input)              — get scene metadata
  2. recommend_preset(input, …)  — pick a preset
  3. encode(input, target)       — execute (or optimize() for free presets, stdio only)
  4. validate_pipeline(output)   — confirm output is shippable
  5. score_fidelity(before, after) — optional perceptual check

When asked "how does Catetus compare to X?", read the resource
catetus://bench/canonical-11 for headline numbers (Catetus V5.2 vs SOG: +15.56 dB avg PSNR
on canonical-11 at 1.02x bytes, 11/11 strict wins).

Tier hints:
- analyze, list_*, get_*, validate_pipeline, optimize, compare are FREE.
- encode, repack, score_fidelity, predict_quality, recommend_preset, batch_jobs require an
  API key (CATETUS_API_KEY env var on stdio, Authorization: Bearer on HTTP).
`.trim();

export interface BuildServerOpts {
  tier: TierContext;
  /** Whether this server is running on HTTP transport (affects optimize/compare path). */
  isHttp?: boolean;
}

export function buildServer(opts: BuildServerOpts): McpServer {
  const { tier, isHttp = false } = opts;
  const server = new McpServer(
    { name: CATETUS_MCP_NAME, version: CATETUS_MCP_VERSION },
    {
      capabilities: {
        tools: { listChanged: false },
        prompts: { listChanged: false },
        resources: { subscribe: true, listChanged: true },
        logging: {},
      },
      instructions: INSTRUCTIONS,
    },
  );

  // ----- Tools -----
  const reg = (name: string, meta: { title?: string; description: string; inputSchema: unknown; outputSchema?: unknown; annotations?: unknown }, handler: (args: unknown, extra?: unknown) => unknown) => {
    if (!isToolVisible(name, tier)) return;
    server.registerTool(
      name,
      meta as Parameters<McpServer["registerTool"]>[1],
      handler as Parameters<McpServer["registerTool"]>[2],
    );
  };

  reg("analyze", analyzeMeta, makeAnalyzeHandler(tier) as never);
  reg("list_presets", listPresetsMeta, makeListPresetsHandler() as never);
  reg("list_scenes", listScenesMeta, makeListScenesHandler() as never);
  reg("get_scene", getSceneMeta, makeGetSceneHandler() as never);
  reg("optimize", optimizeMeta, makeOptimizeHandler(tier, { isHttp }) as never);
  reg("compare", compareMeta, makeCompareHandler({ isHttp }) as never);
  reg("list_competitor_codecs", listCompetitorCodecsMeta, makeListCompetitorCodecsHandler() as never);
  reg("validate_pipeline", validatePipelineMeta, makeValidatePipelineHandler() as never);

  // Paid-tier (only registered when tier.tier === "paid")
  reg("encode", encodeMeta, makeEncodeHandler(tier) as never);
  reg("score_fidelity", scoreFidelityMeta, makeScoreFidelityHandler(tier) as never);
  reg("repack", repackMeta, makeRepackHandler(tier) as never);
  reg("predict_quality", predictQualityMeta, makePredictQualityHandler(tier) as never);
  reg("recommend_preset", recommendPresetMeta, makeRecommendPresetHandler(tier) as never);
  reg("batch_jobs", batchJobsMeta, makeBatchJobsHandler(tier) as never);
  reg("list_jobs", listJobsMeta, makeListJobsHandler(tier) as never);

  // ----- Prompts -----
  type PromptArgs = Record<string, string | undefined>;
  server.registerPrompt(
    "compress_for_product_page",
    compressForProductPageMeta as Parameters<McpServer["registerPrompt"]>[1],
    ((args: PromptArgs) =>
      compressForProductPagePrompt({
        scenes: args.scenes ?? "",
        target_bytes_per_scene: args.target_bytes_per_scene,
        device: args.device,
      })) as Parameters<McpServer["registerPrompt"]>[2],
  );
  server.registerPrompt(
    "compare_against_sog",
    compareAgainstSogMeta as Parameters<McpServer["registerPrompt"]>[1],
    ((args: PromptArgs) =>
      compareAgainstSogPrompt({ scene: args.scene ?? "", output_dir: args.output_dir })) as Parameters<McpServer["registerPrompt"]>[2],
  );
  server.registerPrompt(
    "pick_best_preset",
    pickBestPresetMeta as Parameters<McpServer["registerPrompt"]>[1],
    ((args: PromptArgs) =>
      pickBestPresetPrompt({ scene: args.scene ?? "", constraints_text: args.constraints_text ?? "" })) as Parameters<McpServer["registerPrompt"]>[2],
  );
  server.registerPrompt(
    "audit_existing_outputs",
    auditExistingOutputsMeta as Parameters<McpServer["registerPrompt"]>[1],
    ((args: PromptArgs) =>
      auditExistingOutputsPrompt({ output_dir: args.output_dir ?? "" })) as Parameters<McpServer["registerPrompt"]>[2],
  );

  // Resources are registered by the MCP-RESOURCES agent in src/resources/.
  // We intentionally do not touch that subdirectory here.

  return server;
}
