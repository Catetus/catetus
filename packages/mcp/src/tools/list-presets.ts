// list_presets (public, paginated) — ARCHITECTURE.md §5.4.2
import { z } from "zod";
import { okResult, errorResult } from "../errors.js";
import { Cursor, PresetName } from "../schemas.js";
import { paginate } from "../pagination.js";

export const listPresetsInput = {
  tier_filter: z.enum(["all", "free", "paid"]).default("all"),
  cursor: Cursor.optional(),
  limit: z.number().int().min(1).max(50).default(50),
};

export const listPresetsOutput = {
  presets: z.array(
    z.object({
      name: PresetName,
      tier: z.enum(["free", "paid"]),
      description: z.string(),
      typicalRatio: z.number(),
      passes: z.array(z.string()),
      bestFor: z.array(z.string()),
      targetDevice: z
        .array(z.enum(["mobile-web", "desktop-web", "quest", "visionos"]))
        .optional(),
    }),
  ),
  next_cursor: Cursor.optional(),
};

export const listPresetsMeta = {
  title: "List encoder presets",
  description:
    "List built-in Catetus encoder presets (free + paid) with target use case, typical compression ratio, and tier. Use before picking a preset.",
  inputSchema: listPresetsInput,
  outputSchema: listPresetsOutput,
  annotations: {
    title: "List encoder presets",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

// Built-in catalog. Real catalog should live in src/resources/assets — owned by MCP-RESOURCES.
const CATALOG = [
  {
    name: "lossless-repack",
    tier: "free" as const,
    description: "Re-pack source PLY with no perceptual loss; 5-10% size reduction.",
    typicalRatio: 0.92,
    passes: ["reorder", "quantize-conservative", "pack"],
    bestFor: ["audit", "archival", "minimal change"],
  },
  {
    name: "web-mobile",
    tier: "free" as const,
    description: "Aggressive mobile-web target; 1-3 MB output for most scenes.",
    typicalRatio: 0.04,
    passes: ["prune", "spherical-harmonics-trim", "quantize-q8", "pack-spz"],
    bestFor: ["mobile-first sites", "thumbnails"],
    targetDevice: ["mobile-web" as const],
  },
  {
    name: "web-desktop",
    tier: "free" as const,
    description: "Desktop-web target; 10-30 MB output, ΔE94 < 0.03.",
    typicalRatio: 0.12,
    passes: ["prune-soft", "sh-keep", "quantize-q10", "pack-gltf"],
    bestFor: ["product pages", "portfolio"],
    targetDevice: ["desktop-web" as const],
  },
  {
    name: "quest-browser",
    tier: "free" as const,
    description: "Meta Quest browser VR; 5-10 MB target with mobile-class limits.",
    typicalRatio: 0.06,
    passes: ["prune", "sh-trim", "quantize-q9", "pack-spz"],
    bestFor: ["WebXR", "Quest demos"],
    targetDevice: ["quest" as const],
  },
  {
    name: "visionos-preview",
    tier: "free" as const,
    description: "Apple Vision Pro preview; quality-leaning ≤ 50 MB.",
    typicalRatio: 0.18,
    passes: ["prune-soft", "quantize-q10", "pack-gltf"],
    bestFor: ["AVP demos"],
    targetDevice: ["visionos" as const],
  },
  {
    name: "thumbnail-preview",
    tier: "free" as const,
    description: "<1 MB thumbnail-grade preview.",
    typicalRatio: 0.02,
    passes: ["prune-aggressive", "sh-zero", "quantize-q8"],
    bestFor: ["catalog thumbnails", "loading previews"],
  },
  {
    name: "quality-max",
    tier: "free" as const,
    description: "Quality-leaning free preset; ΔE94 < 0.01 typical.",
    typicalRatio: 0.35,
    passes: ["prune-soft", "sh-keep", "quantize-q11"],
    bestFor: ["pre-final reviews"],
  },
  {
    name: "size-min",
    tier: "free" as const,
    description: "Smallest possible output that still validates.",
    typicalRatio: 0.025,
    passes: ["prune-aggressive", "sh-zero", "quantize-q7"],
    bestFor: ["bandwidth-constrained delivery"],
  },
  {
    name: "differentiable-repack",
    tier: "paid" as const,
    description:
      "gsplat self-distillation on A100; squeeze size while holding PSNR. 3-6 min, $0.05-$0.12.",
    typicalRatio: 0.5,
    passes: ["differentiable-prune", "rd-quantize"],
    bestFor: ["custom byte budgets"],
  },
  {
    name: "v52-quality",
    tier: "paid" as const,
    description:
      "Catetus V5.2 SOTA quality tier — +15.56 dB PSNR vs SOG on canonical-11 at 1.02x bytes.",
    typicalRatio: 1.02,
    passes: ["v52-hyperprior", "v52-residual"],
    bestFor: ["highest fidelity at SOG byte parity"],
  },
  {
    name: "v52-balanced",
    tier: "paid" as const,
    description: "Catetus V5.2 balanced tier — strong fidelity at ~0.6x SOG bytes.",
    typicalRatio: 0.62,
    passes: ["v52-hyperprior", "v52-residual-balanced"],
    bestFor: ["balanced quality+size"],
  },
  {
    name: "t21r-fast",
    tier: "paid" as const,
    description:
      "Catetus T2.1.R fast preset — +6.24 dB vs SOG at 0.98x bytes, faster than V5.2.",
    typicalRatio: 0.98,
    passes: ["t21r-encode"],
    bestFor: ["interactive iterations"],
  },
];

type Args = { tier_filter: "all" | "free" | "paid"; cursor?: string; limit: number };

export function makeListPresetsHandler() {
  return async function listPresets(args: Args) {
    const filtered = CATALOG.filter(
      (p) => args.tier_filter === "all" || p.tier === args.tier_filter,
    );
    const res = paginate(filtered, args.cursor, args.limit, { tier_filter: args.tier_filter });
    if (!res.ok) {
      return errorResult("invalid_cursor", "Pagination cursor is invalid or stale.");
    }
    return okResult({
      presets: res.page,
      next_cursor: res.next_cursor,
    });
  };
}
