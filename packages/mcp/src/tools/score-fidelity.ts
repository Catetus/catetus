// score_fidelity (paid) — ARCHITECTURE.md §5.4.10
import { z } from "zod";
import { okResult } from "../errors.js";
import { SplatRef } from "../schemas.js";
import { sharedApiClient, apiErrorToResult } from "../api-client.js";
import type { TierContext } from "../auth.js";

export const scoreFidelityInput = {
  before: SplatRef,
  after: SplatRef,
  before_frames: z.array(z.string().url()).optional(),
  after_frames: z.array(z.string().url()).optional(),
  cameraPath: z.enum(["orbit-8", "orbit-16", "forward-pan"]).default("orbit-8"),
  frameSize: z.enum(["256x256", "512x512", "1024x1024"]).default("512x512"),
};

export const scoreFidelityOutput = {
  mlScore: z.number().min(0).max(1),
  mlScoreVersion: z.string(),
  features: z.record(z.string(), z.number()),
  status: z.enum(["pass", "borderline", "fail"]),
  perFrame: z
    .array(z.object({ i: z.number().int(), score: z.number() }))
    .optional(),
  costUsd: z.number(),
  creditsRemainingUsd: z.number(),
};

export const scoreFidelityMeta = {
  title: "ML-score a render-pair",
  description:
    "Wraps /v1/fidelity. ML-score a render pair (original vs compressed) and return the V0.4 perceptual ML score plus the 22 input features. Use for honest perceptual checks beyond ΔE94/PSNR.",
  inputSchema: scoreFidelityInput,
  outputSchema: scoreFidelityOutput,
  annotations: {
    title: "ML-score a render-pair",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: true,
  },
};

type Args = z.infer<z.ZodObject<typeof scoreFidelityInput>>;

export function makeScoreFidelityHandler(ctx: TierContext) {
  return async function scoreFidelity(args: Args, extra?: { signal?: AbortSignal }) {
    try {
      const res = await sharedApiClient.call<Record<string, unknown>>("/v1/fidelity", {
        method: "POST",
        body: args,
        apiKey: ctx.apiKey,
        signal: extra?.signal,
        timeoutMs: 120_000,
      });
      return okResult(res);
    } catch (e) {
      return apiErrorToResult(e);
    }
  };
}
