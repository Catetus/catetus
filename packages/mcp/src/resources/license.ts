// License + docs resources.
//
// URIs:
//   catetus://license/sdk-terms        — Apache-2.0 summary + commercial-use FAQ (text/markdown)
//   catetus://docs/preset-cheatsheet   — LLM-cheat-sheet for picking presets (text/markdown)

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { loadTextAsset, assetByteLength } from "./asset-loader.js";

const LICENSE_URI = "catetus://license/sdk-terms";
const PRESET_CHEATSHEET_URI = "catetus://docs/preset-cheatsheet";

export function registerLicenseResources(server: McpServer): void {
  server.registerResource(
    "license-sdk-terms",
    LICENSE_URI,
    {
      title: "Catetus SDK License (Apache-2.0)",
      description:
        "Catetus SDK license summary — Apache-2.0 + commercial-use FAQ. Covers " +
        "what you can do with the free SDK (CLI, viewer, codec, MCP server), what " +
        "requires an API key (paid tier: encode, score_fidelity, repack, " +
        "predict_quality, recommend_preset, batch_jobs), trademark guidance, the " +
        "patent grant, attribution snippets, and corpus-license details. Mirrors " +
        "what api.catetus.com's /v1/sdk-license endpoint issues.",
      mimeType: "text/markdown",
      size: assetByteLength("license/sdk-terms.md"),
      annotations: {
        audience: ["assistant", "user"],
        priority: 0.6,
        lastModified: "2026-05-27T00:00:00Z",
      },
    },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "text/markdown",
          text: loadTextAsset("license/sdk-terms.md"),
        },
      ],
    }),
  );

  server.registerResource(
    "docs-preset-cheatsheet",
    PRESET_CHEATSHEET_URI,
    {
      title: "Preset Cheatsheet (LLM Quick-Pick)",
      description:
        "Compact (~150 lines) 'if X then Y' decision tree for picking a Catetus " +
        "preset by output target (web/XR/thumbnail/archive), byte budget, quality " +
        "floor, and device. Includes anti-patterns and a programmatic picker " +
        "recipe (analyze -> branch on splatCount + likelyClass -> preset). " +
        "Designed for in-context inclusion when an LLM is choosing a preset.",
      mimeType: "text/markdown",
      size: assetByteLength("docs/preset-cheatsheet.md"),
      annotations: {
        audience: ["assistant"],
        priority: 0.9,
        lastModified: "2026-05-27T00:00:00Z",
      },
    },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "text/markdown",
          text: loadTextAsset("docs/preset-cheatsheet.md"),
        },
      ],
    }),
  );
}

export const LICENSE_URIS = {
  sdkTerms: LICENSE_URI,
  presetCheatsheet: PRESET_CHEATSHEET_URI,
} as const;
