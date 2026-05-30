// Per-scene resource template.
//
// URI template: catetus://scene/{scene_id}
//
// For canonical-11 scenes, returns the JSON record for the requested scene
// (lifted from bench/canonical-11.json). For splatbench-v0 scenes, returns
// the corresponding splatbench-v0 record. The `scene_id` is matched against
// the `scene` field in canonical-11.json, then the `id` field in splatbench-v0.json.
//
// Per ARCHITECTURE.md §6.1, fixtures >2 MB should be returned as resource_link
// rather than embedded bytes. We don't ship .ply fixtures in this package, so
// we always return the metadata-only JSON record (small).

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { loadJsonAsset } from "./asset-loader.js";

const SCENE_TEMPLATE = "catetus://scene/{scene_id}";

type CanonicalSceneRow = {
  scene: string;
  splats: number;
  input_ply_bytes: number;
  input_ply_md5: string;
  encoded_glb_bytes: number;
  encoded_shpal_bytes?: number;
  encoded_total_bytes: number;
  compression_ratio: number;
  psnr_mean_db: number;
  psnr_min_db?: number;
  psnr_max_db?: number;
  ssim_mean: number;
  frames?: number;
  frame_size?: string;
};

type CanonicalBench = {
  scenes: CanonicalSceneRow[];
};

type SplatbenchScene = {
  id: string;
  source?: string;
  class?: string;
  origin?: string;
  license?: string;
  splatCount?: number;
  bytesIn?: number;
  hash?: string;
  shDegree?: number;
  [k: string]: unknown;
};

type SplatbenchCorpus = {
  scenes: SplatbenchScene[];
};

// Cached at module load — these JSON files are static.
let canonicalCache: CanonicalBench | null = null;
let splatbenchCache: SplatbenchCorpus | null = null;

function getCanonical(): CanonicalBench {
  if (!canonicalCache) {
    canonicalCache = loadJsonAsset<CanonicalBench>("bench/canonical-11.json");
  }
  return canonicalCache;
}

function getSplatbench(): SplatbenchCorpus {
  if (!splatbenchCache) {
    splatbenchCache = loadJsonAsset<SplatbenchCorpus>("bench/splatbench-v0.json");
  }
  return splatbenchCache;
}

type SceneLookupResult =
  | { found: true; corpus: "canonical-11"; record: CanonicalSceneRow }
  | { found: true; corpus: "splatbench-v0"; record: SplatbenchScene }
  | { found: false; available: string[] };

function lookupScene(sceneId: string): SceneLookupResult {
  const canonical = getCanonical();
  const c = canonical.scenes.find((s) => s.scene === sceneId);
  if (c) return { found: true, corpus: "canonical-11", record: c };

  const splatbench = getSplatbench();
  const sb = splatbench.scenes.find((s) => s.id === sceneId);
  if (sb) return { found: true, corpus: "splatbench-v0", record: sb };

  return {
    found: false,
    available: [
      ...canonical.scenes.map((s) => s.scene),
      ...splatbench.scenes.map((s) => s.id),
    ],
  };
}

export function registerSceneResources(server: McpServer): void {
  server.registerResource(
    "scene-detail",
    new ResourceTemplate(SCENE_TEMPLATE, {
      // list: enumerate every known scene as a concrete resource entry.
      list: async () => {
        const canonical = getCanonical();
        const splatbench = getSplatbench();
        const canonicalEntries = canonical.scenes.map((s) => ({
          uri: `catetus://scene/${s.scene}`,
          name: `scene-${s.scene}`,
          title: `Canonical-11 scene: ${s.scene}`,
          mimeType: "application/json",
          description: `Per-scene record for ${s.scene} from the canonical-11 leaderboard ` +
            `(${s.splats.toLocaleString()} splats, ${(s.encoded_total_bytes / 1_048_576).toFixed(1)} MB encoded, ` +
            `${s.psnr_mean_db.toFixed(2)} dB PSNR).`,
        }));
        const splatbenchEntries = splatbench.scenes.map((s) => ({
          uri: `catetus://scene/${s.id}`,
          name: `scene-${s.id}`,
          title: `SplatBench-v0 scene: ${s.id}`,
          mimeType: "application/json",
          description: `Per-scene record for ${s.id} from SplatBench v0` +
            (s.class ? ` (class=${s.class})` : "") +
            (s.splatCount ? `, ${s.splatCount.toLocaleString()} splats` : "") +
            ".",
        }));
        return { resources: [...canonicalEntries, ...splatbenchEntries] };
      },
      // complete: autocomplete scene_id from union of both corpora.
      complete: {
        scene_id: async (value: string) => {
          const ids = [
            ...getCanonical().scenes.map((s) => s.scene),
            ...getSplatbench().scenes.map((s) => s.id),
          ];
          const v = value.toLowerCase();
          return ids.filter((id) => id.toLowerCase().includes(v));
        },
      },
    }),
    {
      title: "Per-scene corpus record",
      description:
        "Dynamic per-scene resource. URI: catetus://scene/{scene_id}. Returns the " +
        "JSON record for the requested scene — canonical-11 scenes first " +
        "(bicycle, bonsai, counter, drjohnson, garden, kitchen, playroom, room, " +
        "stump, train, truck), then splatbench-v0 ids. The record includes " +
        "splat count, input PLY hash, encoded bytes, compression ratio, and PSNR/SSIM. " +
        "Use after list_scenes when you have a scene_id and want the full per-scene " +
        "leaderboard row.",
      mimeType: "application/json",
      annotations: {
        audience: ["assistant", "user"],
        priority: 0.7,
      },
    },
    async (uri, variables) => {
      const sceneId = String(variables.scene_id ?? "");
      const result = lookupScene(sceneId);
      if (!result.found) {
        // Return an MCP-spec-conformant error payload as a text content block.
        // (The spec allows resources/read to surface application errors via content.)
        const errBody = {
          error: {
            code: "scene_not_found",
            message: `Scene '${sceneId}' not found in either canonical-11 or splatbench-v0.`,
            hint: `Use one of: ${result.available.slice(0, 8).join(", ")}${result.available.length > 8 ? ", ..." : ""}`,
            availableCount: result.available.length,
          },
        };
        return {
          contents: [
            {
              uri: uri.href,
              mimeType: "application/json",
              text: JSON.stringify(errBody, null, 2),
            },
          ],
        };
      }

      const payload = {
        scene_id: sceneId,
        corpus: result.corpus,
        record: result.record,
        leaderboardRef:
          result.corpus === "canonical-11"
            ? `catetus://bench/canonical-11#${sceneId}`
            : `catetus://bench/splatbench-v0#${sceneId}`,
      };
      return {
        contents: [
          {
            uri: uri.href,
            mimeType: "application/json",
            text: JSON.stringify(payload, null, 2),
          },
        ],
      };
    },
  );
}

export const SCENE_URI_TEMPLATE = SCENE_TEMPLATE;
