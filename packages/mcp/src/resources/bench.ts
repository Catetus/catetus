// Bench resources: canonical-11 + splatbench-v0.
// Both exposed as application/json passthroughs of the published benches.
//
// URIs:
//   catetus://bench/canonical-11   — 11-scene Inria 3DGS leaderboard
//   catetus://bench/splatbench-v0  — 16-scene synthetic+real corpus

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { loadTextAsset, assetByteLength } from "./asset-loader.js";

const CANONICAL_11_URI = "catetus://bench/canonical-11";
const SPLATBENCH_V0_URI = "catetus://bench/splatbench-v0";

export function registerBenchResources(server: McpServer): void {
  server.registerResource(
    "bench-canonical-11",
    CANONICAL_11_URI,
    {
      title: "Canonical-11 Leaderboard",
      description:
        "Per-scene results for the 11-scene Inria 3DGS canonical corpus " +
        "(bicycle, bonsai, counter, drjohnson, garden, kitchen, playroom, " +
        "room, stump, train, truck) under the wmv-vq45-no-prune-tight preset. " +
        "Includes splat counts, input PLY hashes, encoded byte sizes, compression " +
        "ratios, and per-scene PSNR/SSIM over an orbit-8 camera path. " +
        "This is the canonical 'SF baseline' leaderboard the landing page cites. " +
        "Read this resource when you need exact per-scene numbers for headline claims.",
      mimeType: "application/json",
      size: assetByteLength("bench/canonical-11.json"),
      annotations: {
        audience: ["assistant", "user"],
        priority: 0.9,
        lastModified: "2026-05-20T17:54:14Z",
      },
    },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "application/json",
          text: loadTextAsset("bench/canonical-11.json"),
        },
      ],
    }),
  );

  server.registerResource(
    "bench-splatbench-v0",
    SPLATBENCH_V0_URI,
    {
      title: "SplatBench v0 Corpus",
      description:
        "Full 16-scene SplatBench v0 corpus index: 3 real Mip-NeRF360 scenes " +
        "(bonsai, bicycle, stump), 12 synthetic class proxies covering PRD corpus " +
        "classes (product, indoor, floater, outdoor, dense, specular, foliage, " +
        "lowlight, portrait, texture, transparency, motion, depth, banding), " +
        "and 3 cluster-fly synthetic scenes. Per-scene records include splat count, " +
        "blake3 hash, input bytes, per-preset compression metrics, fidelity scores " +
        "(meanDeltaE94, ssim, ml-score), and PRD-class labels. Backing for " +
        "find_similar_scenes and list_scenes when corpus='splatbench-v0'.",
      mimeType: "application/json",
      size: assetByteLength("bench/splatbench-v0.json"),
      annotations: {
        audience: ["assistant", "user"],
        priority: 0.8,
        lastModified: "2026-05-15T00:00:00Z",
      },
    },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "application/json",
          text: loadTextAsset("bench/splatbench-v0.json"),
        },
      ],
    }),
  );
}

// Exported for tests + the scenes template (which reads canonical-11 to enumerate scene_ids).
export const BENCH_URIS = {
  canonical11: CANONICAL_11_URI,
  splatbenchV0: SPLATBENCH_V0_URI,
} as const;
