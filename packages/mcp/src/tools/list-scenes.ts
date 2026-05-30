// list_scenes (public, paginated) — ARCHITECTURE.md §5.4.3
import { z } from "zod";
import { okResult, errorResult } from "../errors.js";
import { Cursor } from "../schemas.js";
import { paginate } from "../pagination.js";
import { CANONICAL_11_SCENES, SPLATBENCH_V0_SCENES } from "./_scene-data.js";

export const listScenesInput = {
  corpus: z.enum(["splatbench-v0", "canonical-11"]).default("canonical-11"),
  filter: z
    .object({
      class: z.enum(["indoor", "outdoor", "object", "transparent", "portrait"]).optional(),
      license: z.enum(["mit", "cc-by", "cc-by-nc", "custom"]).optional(),
      min_splats: z.number().int().optional(),
      max_splats: z.number().int().optional(),
    })
    .optional(),
  cursor: Cursor.optional(),
  limit: z.number().int().min(1).max(50).default(20),
};

export const listScenesOutput = {
  scenes: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      class: z.string(),
      license: z.string(),
      splatCount: z.number().int(),
      bytesBaseline: z.number().int(),
      shDegree: z.number().int(),
      hash: z.string(),
      summary: z.string(),
      leaderboardRef: z.string().optional(),
    }),
  ),
  next_cursor: Cursor.optional(),
  total: z.number().int(),
};

export const listScenesMeta = {
  title: "List SplatBench scenes",
  description:
    "List SplatBench corpus scenes with class, license, splat count, and headline metrics. Use to ground answers in real numbers or pick a reference scene.",
  inputSchema: listScenesInput,
  outputSchema: listScenesOutput,
  annotations: {
    title: "List SplatBench scenes",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

type Args = z.infer<z.ZodObject<typeof listScenesInput>>;

export function makeListScenesHandler() {
  return async function listScenes(args: Args) {
    const all = args.corpus === "canonical-11" ? CANONICAL_11_SCENES : SPLATBENCH_V0_SCENES;
    const filtered = all.filter((s) => {
      const f = args.filter;
      if (!f) return true;
      if (f.class && s.class !== f.class) return false;
      if (f.license && s.license !== f.license) return false;
      if (f.min_splats != null && s.splatCount < f.min_splats) return false;
      if (f.max_splats != null && s.splatCount > f.max_splats) return false;
      return true;
    });
    const res = paginate(filtered, args.cursor, args.limit, {
      corpus: args.corpus,
      filter: args.filter,
    });
    if (!res.ok) {
      return errorResult("invalid_cursor", "Pagination cursor is invalid or stale.");
    }
    return okResult({
      scenes: res.page,
      next_cursor: res.next_cursor,
      total: filtered.length,
    });
  };
}
