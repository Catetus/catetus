// list_competitor_codecs (public, paginated) — ARCHITECTURE.md §5.4.7
import { z } from "zod";
import { okResult, errorResult } from "../errors.js";
import { Cursor } from "../schemas.js";
import { paginate } from "../pagination.js";

export const listCompetitorCodecsInput = {
  cursor: Cursor.optional(),
  limit: z.number().int().min(1).max(50).default(20),
  filter: z
    .object({
      open_source_only: z.boolean().optional(),
    })
    .optional(),
};

export const listCompetitorCodecsOutput = {
  codecs: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      url: z.string().url(),
      license: z.string(),
      last_updated: z.string(),
      summary: z.string(),
      catetus_comparison: z
        .object({
          tier: z.enum(["sf-baseline", "t21r", "v52"]).optional(),
          avg_psnr_delta_db: z.number().optional(),
          avg_bytes_ratio: z.number().optional(),
          benchmark: z.string(),
        })
        .optional(),
    }),
  ),
  next_cursor: Cursor.optional(),
};

export const listCompetitorCodecsMeta = {
  title: "List comparable codecs",
  description:
    "List public competitor codecs (PlayCanvas SOG, Inria 3DGS, SPZ, glTF KHR_gaussian_splatting, …) with license, summary and head-to-head numbers against Catetus. Use when asked 'how do you compare to X?'",
  inputSchema: listCompetitorCodecsInput,
  outputSchema: listCompetitorCodecsOutput,
  annotations: {
    title: "List comparable codecs",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

const CATALOG = [
  {
    id: "playcanvas-sog",
    name: "PlayCanvas SOG",
    url: "https://github.com/playcanvas/supersplat",
    license: "MIT",
    last_updated: "2026-04-01",
    summary:
      "SuperSplat-compatible Self-Organizing Gaussians; current industry-standard quantized format.",
    open_source: true,
    catetus_comparison: {
      tier: "v52" as const,
      avg_psnr_delta_db: 15.56,
      avg_bytes_ratio: 1.02,
      benchmark: "canonical-11",
    },
  },
  {
    id: "inria-3dgs",
    name: "Inria 3DGS",
    url: "https://github.com/graphdeco-inria/gaussian-splatting",
    license: "custom-non-commercial",
    last_updated: "2025-09-15",
    summary: "Reference 3DGS trainer/exporter; uncompressed baseline.",
    open_source: true,
    catetus_comparison: {
      tier: "sf-baseline" as const,
      avg_psnr_delta_db: 0,
      avg_bytes_ratio: 0.0,
      benchmark: "canonical-11",
    },
  },
  {
    id: "spz",
    name: "SPZ",
    url: "https://github.com/nianticlabs/spz",
    license: "MIT",
    last_updated: "2025-10-20",
    summary: "Niantic SPZ binary format; mobile-web optimized.",
    open_source: true,
  },
  {
    id: "khr-gaussian-splatting",
    name: "glTF KHR_gaussian_splatting",
    url: "https://github.com/KhronosGroup/glTF/pull/2580",
    license: "Khronos",
    last_updated: "2026-05-10",
    summary: "Khronos draft extension for glTF gaussian splat carriage; container, not codec.",
    open_source: true,
  },
];

type Args = {
  cursor?: string;
  limit: number;
  filter?: { open_source_only?: boolean };
};

export function makeListCompetitorCodecsHandler() {
  return async function listCompetitorCodecs(args: Args) {
    const filtered = CATALOG.filter((c) => {
      if (args.filter?.open_source_only && !c.open_source) return false;
      return true;
    });
    const res = paginate(filtered, args.cursor, args.limit, { filter: args.filter });
    if (!res.ok) {
      return errorResult("invalid_cursor", "Pagination cursor is invalid or stale.");
    }
    return okResult({
      codecs: res.page.map((c) => ({
        id: c.id,
        name: c.name,
        url: c.url,
        license: c.license,
        last_updated: c.last_updated,
        summary: c.summary,
        catetus_comparison: c.catetus_comparison,
      })),
      next_cursor: res.next_cursor,
    });
  };
}
