// Preset catalog resource.
//
// URI: catetus://presets/catalog  — application/json

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { loadTextAsset, assetByteLength } from "./asset-loader.js";

const PRESETS_CATALOG_URI = "catetus://presets/catalog";

export function registerPresetResources(server: McpServer): void {
  server.registerResource(
    "presets-catalog",
    PRESETS_CATALOG_URI,
    {
      title: "Preset Catalog",
      description:
        "Full catalog of built-in Catetus encoder presets across both free and " +
        "paid tiers. Each entry includes the canonical name, tier (free/paid), " +
        "one-paragraph description, typical compression ratio (output/input), " +
        "the ordered pass list the encoder runs, and best-fit use cases. " +
        "Backing data for the list_presets tool. The 12 presets cover lossless " +
        "repack, web-mobile, web-desktop, quest-browser, visionos-preview, " +
        "thumbnail-preview, quality-max, size-min (free) and " +
        "differentiable-repack, v52-quality, v52-balanced, t21r-fast (paid).",
      mimeType: "application/json",
      size: assetByteLength("presets/catalog.json"),
      annotations: {
        audience: ["assistant", "user"],
        priority: 0.85,
        lastModified: "2026-05-27T00:00:00Z",
      },
    },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "application/json",
          text: loadTextAsset("presets/catalog.json"),
        },
      ],
    }),
  );
}

export const PRESET_URIS = {
  catalog: PRESETS_CATALOG_URI,
} as const;
