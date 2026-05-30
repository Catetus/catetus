// Corpus resources: 3-tier comparison (markdown) + competitor codecs (json).
//
// URIs:
//   catetus://corpus/3-tier-comparison    — SF / T2.1.R / V5.2 vs SOG digest (text/markdown)
//   catetus://corpus/competitor-codecs    — SuperSplat / SPZ / HAC++ catalog (application/json)

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { loadTextAsset, assetByteLength } from "./asset-loader.js";

const THREE_TIER_URI = "catetus://corpus/3-tier-comparison";
const COMPETITOR_CODECS_URI = "catetus://corpus/competitor-codecs";

export function registerCorpusResources(server: McpServer): void {
  server.registerResource(
    "corpus-3-tier-comparison",
    THREE_TIER_URI,
    {
      title: "3-Tier Comparison: SF / T2.1.R / V5.2 vs SOG",
      description:
        "Markdown digest of the canonical-11 leaderboard comparing the three " +
        "Catetus encode tiers (SF baseline, T2.1.R fast, V5.2 quality) against " +
        "PlayCanvas SOG. Includes per-scene PSNR deltas, byte ratios, win counts, " +
        "and the methodology caveats (orbit-vs-SF-GT, palette-on/off SF variants). " +
        "This is the stable URL the landing page links to for the V5.2 +15.56 dB " +
        "headline claim.",
      mimeType: "text/markdown",
      size: assetByteLength("corpus/3-tier-comparison.md"),
      annotations: {
        audience: ["assistant", "user"],
        priority: 0.95,
        lastModified: "2026-05-27T00:00:00Z",
      },
    },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "text/markdown",
          text: loadTextAsset("corpus/3-tier-comparison.md"),
        },
      ],
    }),
  );

  server.registerResource(
    "corpus-competitor-codecs",
    COMPETITOR_CODECS_URI,
    {
      title: "Competitor Codec Catalog",
      description:
        "Audited catalog of competing 3DGS compression codecs (PlayCanvas SOG / " +
        "SOGS, Niantic SPZ, HAC++, Compact3DGS, gsplat-Niedermayr, KSPLAT/SPLAT, " +
        "GSICO, FlexGaussian, Aras-P). Each entry includes the published claim, " +
        "our measured cross-corpus ratio (where applicable), PSNR if published, " +
        "license, repo/paper link, and an honest verdict. Backing data for " +
        "list_competitor_codecs. Source audit at " +
        "splatforge-private/research/competitive/supersplat_claims_2026-05-27.md.",
      mimeType: "application/json",
      size: assetByteLength("corpus/competitor-codecs.json"),
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
          text: loadTextAsset("corpus/competitor-codecs.json"),
        },
      ],
    }),
  );
}

export const CORPUS_URIS = {
  threeTierComparison: THREE_TIER_URI,
  competitorCodecs: COMPETITOR_CODECS_URI,
} as const;
