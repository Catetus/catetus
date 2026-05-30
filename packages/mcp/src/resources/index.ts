// Central resource registry. Implementer A's `src/server.ts` calls
// `registerAllResources(server)` after constructing the McpServer.
//
// Per ARCHITECTURE.md §6 — 7 static resources + 1 template:
//   STATIC (7):
//     catetus://bench/canonical-11        (application/json)
//     catetus://bench/splatbench-v0       (application/json)
//     catetus://corpus/3-tier-comparison  (text/markdown)
//     catetus://corpus/competitor-codecs  (application/json)
//     catetus://presets/catalog           (application/json)
//     catetus://license/sdk-terms         (text/markdown)
//     catetus://docs/preset-cheatsheet    (text/markdown)
//   TEMPLATE (1):
//     catetus://scene/{scene_id}          (application/json, per-scene)
//
// Subscription capability — per §6.3, declare `subscribe: true, listChanged: true`
// on the server capability set but do NOT wire the notification loop in v1.
// That wiring lives in src/server.ts (owned by MCP-IMPL); we just provide a
// const here that they can read.

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

import { registerBenchResources, BENCH_URIS } from "./bench.js";
import { registerCorpusResources, CORPUS_URIS } from "./corpus.js";
import { registerPresetResources, PRESET_URIS } from "./presets.js";
import { registerLicenseResources, LICENSE_URIS } from "./license.js";
import { registerSceneResources, SCENE_URI_TEMPLATE } from "./scenes.js";

/**
 * Capability flags the McpServer should declare for resources support.
 * Read by src/server.ts (owned by MCP-IMPL) when constructing the McpServer.
 */
export const RESOURCES_CAPABILITY = {
  // TODO(v2): wire the notification loop. v1 clients refetch on session start.
  subscribe: true,
  listChanged: true,
} as const;

/**
 * Register all 7 static resources + 1 template on the server.
 * Idempotent at the call-site level (call once per server instance).
 */
export function registerAllResources(server: McpServer): void {
  registerBenchResources(server);
  registerCorpusResources(server);
  registerPresetResources(server);
  registerLicenseResources(server);
  registerSceneResources(server);
}

/**
 * Canonical URI registry — exported for tests, tools that emit resource_link,
 * and for the prompts subsystem (which references resource URIs in messages).
 */
export const RESOURCE_URIS = {
  ...BENCH_URIS,
  ...CORPUS_URIS,
  ...PRESET_URIS,
  ...LICENSE_URIS,
  sceneTemplate: SCENE_URI_TEMPLATE,
} as const;

/**
 * Flat list of every static resource URI (no templates). Used by tests.
 */
export const STATIC_RESOURCE_URIS: readonly string[] = [
  BENCH_URIS.canonical11,
  BENCH_URIS.splatbenchV0,
  CORPUS_URIS.threeTierComparison,
  CORPUS_URIS.competitorCodecs,
  PRESET_URIS.catalog,
  LICENSE_URIS.sdkTerms,
  LICENSE_URIS.presetCheatsheet,
];

/** Resource-template URIs (no static counterparts). */
export const RESOURCE_TEMPLATE_URIS: readonly string[] = [SCENE_URI_TEMPLATE];
