// get_scene (public) — ARCHITECTURE.md §5.4.4
import { z } from "zod";
import { okResult, errorResult } from "../errors.js";
import { PresetName } from "../schemas.js";
import { CANONICAL_11_SCENES, SPLATBENCH_V0_SCENES } from "./_scene-data.js";

export const getSceneInput = {
  scene_id: z.string().describe("From list_scenes."),
  corpus: z.enum(["splatbench-v0", "canonical-11"]).default("canonical-11"),
};

export const getSceneOutput = {
  id: z.string(),
  name: z.string(),
  class: z.string(),
  license: z.string(),
  splatCount: z.number().int(),
  bytesBaseline: z.number().int(),
  shDegree: z.number().int(),
  hash: z.string(),
  bbox: z
    .object({
      min: z.tuple([z.number(), z.number(), z.number()]),
      max: z.tuple([z.number(), z.number(), z.number()]),
    })
    .optional(),
  perPreset: z
    .record(
      z.string(),
      z.object({
        bytesOut: z.number().int(),
        ratio: z.number(),
        meanDeltaE94: z.number().optional(),
        psnr: z.number().optional(),
        mlScore: z.number().nullable(),
      }),
    )
    .optional(),
  leaderboardRow: z
    .object({
      sog_mb: z.number(),
      sf_mb: z.number(),
      t21r_mb: z.number(),
      v52_mb: z.number(),
      sog_psnr: z.number(),
      sf_psnr: z.number(),
      t21r_psnr: z.number(),
      v52_psnr: z.number(),
      delta_v52_minus_sog: z.number(),
      delta_v52_minus_sf: z.number(),
    })
    .optional(),
  fixtureUri: z.string().optional(),
};

void PresetName;

export const getSceneMeta = {
  title: "Get SplatBench scene detail",
  description:
    "Get the full per-scene record including per-preset compression metrics and the 3-tier (SF / T2.1.R / V5.2) vs SOG leaderboard row. Use after list_scenes once you've picked a reference.",
  inputSchema: getSceneInput,
  outputSchema: getSceneOutput,
  annotations: {
    title: "Get SplatBench scene detail",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

type Args = { scene_id: string; corpus: "splatbench-v0" | "canonical-11" };

export function makeGetSceneHandler() {
  return async function getScene(args: Args) {
    const all = args.corpus === "canonical-11" ? CANONICAL_11_SCENES : SPLATBENCH_V0_SCENES;
    const found = all.find((s) => s.id === args.scene_id);
    if (!found) {
      return errorResult("scene_not_found", `Scene '${args.scene_id}' not found in ${args.corpus}.`, {
        hint: "Use `list_scenes` to enumerate valid IDs.",
      });
    }
    return okResult({
      id: found.id,
      name: found.name,
      class: found.class,
      license: found.license,
      splatCount: found.splatCount,
      bytesBaseline: found.bytesBaseline,
      shDegree: found.shDegree,
      hash: found.hash,
      bbox: found.bbox,
      perPreset: found.perPreset,
      leaderboardRow: found.leaderboardRow,
      fixtureUri: found.fixtureUri,
    });
  };
}
