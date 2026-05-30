// recommend_preset (paid) — ARCHITECTURE.md §5.4.13
import { z } from "zod";
import { okResult, errorResult } from "../errors.js";
import { SplatRef, PresetName, FreePresetName, PaidPresetName, PredictedFidelity } from "../schemas.js";
import type { TierContext } from "../auth.js";
import { makePredictQualityHandler } from "./predict-quality.js";

export const recommendPresetInput = {
  input: SplatRef,
  constraints: z.object({
    maxBytes: z.number().int().optional(),
    maxMeanDeltaE94: z.number().optional(),
    minMlScore: z.number().min(0).max(1).optional(),
    targetDevice: z
      .enum(["mobile-web", "desktop-web", "quest", "visionos", "unspecified"])
      .default("unspecified"),
    allowPaid: z.boolean().default(true),
  }),
};

export const recommendPresetOutput = {
  recommended: z.object({
    preset: PresetName,
    tier: z.enum(["free", "paid"]),
    predictedBytes: z.number().int(),
    predictedFidelity: PredictedFidelity,
  }),
  alternative: z
    .object({
      preset: PresetName,
      tier: z.enum(["free", "paid"]),
      predictedBytes: z.number().int(),
      predictedFidelity: PredictedFidelity,
    })
    .optional(),
  rationale: z.string(),
  status: z.enum(["ok", "no_free_preset_fits", "no_preset_fits", "corpus_too_sparse"]),
};

export const recommendPresetMeta = {
  title: "Recommend best preset for constraints",
  description:
    "Given a splat scene and constraints (target bytes, max color drift, target device, allow_paid), recommend the best preset and a next-best alternative. The right tool when the user says 'compress this'.",
  inputSchema: recommendPresetInput,
  outputSchema: recommendPresetOutput,
  annotations: {
    title: "Recommend best preset for constraints",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: true,
  },
};

type Args = z.infer<z.ZodObject<typeof recommendPresetInput>>;

export function makeRecommendPresetHandler(ctx: TierContext) {
  const predict = makePredictQualityHandler(ctx);
  return async function recommendPreset(args: Args) {
    const candidates: string[] = [
      ...FreePresetName.options,
      ...(args.constraints.allowPaid ? PaidPresetName.options : []),
    ];

    type Scored = {
      preset: string;
      tier: "free" | "paid";
      predictedBytes: number;
      predictedFidelity: z.infer<typeof PredictedFidelity>;
      fits: boolean;
    };

    const scored: Scored[] = [];
    for (const preset of candidates) {
      const res = (await predict({ input: args.input, preset: preset as z.infer<typeof PresetName> })) as {
        structuredContent?: { predictedBytes?: number; predictedFidelity?: z.infer<typeof PredictedFidelity> };
        isError?: boolean;
      };
      if (res.isError || !res.structuredContent?.predictedFidelity) continue;
      const pf = res.structuredContent.predictedFidelity;
      const predictedBytes = res.structuredContent.predictedBytes ?? 0;
      const tier: "free" | "paid" = FreePresetName.options.includes(preset as never)
        ? "free"
        : "paid";
      let fits = true;
      if (args.constraints.maxBytes != null && predictedBytes > args.constraints.maxBytes) fits = false;
      if (args.constraints.maxMeanDeltaE94 != null && pf.meanDeltaE94 > args.constraints.maxMeanDeltaE94)
        fits = false;
      if (
        args.constraints.minMlScore != null &&
        pf.mlScore != null &&
        pf.mlScore < args.constraints.minMlScore
      )
        fits = false;
      scored.push({ preset, tier, predictedBytes, predictedFidelity: pf, fits });
    }

    if (scored.length === 0) {
      return errorResult("corpus_too_sparse", "No predictions could be generated.", {
        hint: "Make sure input is well-formed; otherwise corpus_too_sparse is permanent.",
      });
    }

    const fitting = scored
      .filter((s) => s.fits)
      .sort((a, b) => {
        // primary: mlScore desc, secondary: predictedBytes asc
        const aScore = a.predictedFidelity.mlScore ?? 0;
        const bScore = b.predictedFidelity.mlScore ?? 0;
        if (bScore !== aScore) return bScore - aScore;
        return a.predictedBytes - b.predictedBytes;
      });

    if (fitting.length === 0) {
      // No preset fits at all
      const haveFree = scored.some((s) => s.tier === "free");
      return okResult({
        recommended: {
          preset: scored[0].preset as z.infer<typeof PresetName>,
          tier: scored[0].tier,
          predictedBytes: scored[0].predictedBytes,
          predictedFidelity: scored[0].predictedFidelity,
        },
        rationale: "No preset satisfied your constraints; returning closest match. Try relaxing constraints.",
        status: haveFree && !args.constraints.allowPaid ? "no_free_preset_fits" : "no_preset_fits",
      });
    }

    const recommended = fitting[0];
    const alternative = fitting[1];
    return okResult({
      recommended: {
        preset: recommended.preset as z.infer<typeof PresetName>,
        tier: recommended.tier,
        predictedBytes: recommended.predictedBytes,
        predictedFidelity: recommended.predictedFidelity,
      },
      alternative: alternative
        ? {
            preset: alternative.preset as z.infer<typeof PresetName>,
            tier: alternative.tier,
            predictedBytes: alternative.predictedBytes,
            predictedFidelity: alternative.predictedFidelity,
          }
        : undefined,
      rationale: `Picked '${recommended.preset}' (${recommended.tier}) — predicted ${(recommended.predictedBytes / 1_048_576).toFixed(2)} MB at meanΔE94=${recommended.predictedFidelity.meanDeltaE94.toFixed(3)}, mlScore=${recommended.predictedFidelity.mlScore ?? "n/a"}. Method: corpus-interp over canonical-11.`,
      status: "ok",
    });
  };
}
