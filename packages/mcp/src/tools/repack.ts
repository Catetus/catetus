// repack (paid) — ARCHITECTURE.md §5.4.11
import { z } from "zod";
import { okResult } from "../errors.js";
import { SplatRef, JobId } from "../schemas.js";
import { sharedApiClient, apiErrorToResult } from "../api-client.js";
import type { TierContext } from "../auth.js";

export const repackInput = {
  input: SplatRef,
  targetRatio: z
    .number()
    .min(0.05)
    .max(0.95)
    .default(0.5)
    .describe("Splat-count retention ratio. 0.5 = output ≈ half the splats."),
  targetBytes: z.number().int().optional(),
  iters: z.number().int().min(200).max(5000).default(1500),
  seed: z.number().int().default(0),
};

export const repackOutput = {
  job_id: JobId,
  status: z.enum(["queued", "running"]),
  etaSeconds: z.number().int(),
  estimatedCostUsd: z.number(),
  creditsRemainingUsd: z.number(),
  poll_with: z.literal("list_jobs"),
};

export const repackMeta = {
  title: "Differentiable repack (A100)",
  description:
    "Run differentiable repack on a Catetus encoded asset to push it to a target byte budget while preserving PSNR. Uses gsplat self-distillation on A100. Async; 3-6 min, $0.05-$0.12. Returns job_id.",
  inputSchema: repackInput,
  outputSchema: repackOutput,
  annotations: {
    title: "Differentiable repack (A100)",
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: true,
  },
};

type Args = z.infer<z.ZodObject<typeof repackInput>>;

export function makeRepackHandler(ctx: TierContext) {
  return async function repack(args: Args, extra?: { signal?: AbortSignal }) {
    try {
      const res = await sharedApiClient.call<Record<string, unknown>>("/v1/jobs", {
        method: "POST",
        body: { preset: "differentiable-repack", ...args },
        apiKey: ctx.apiKey,
        signal: extra?.signal,
      });
      // Normalize for outputSchema
      const out: Record<string, unknown> = {
        job_id: (res as { job_id?: string }).job_id ?? "",
        status: (res as { status?: string }).status ?? "queued",
        etaSeconds: (res as { etaSeconds?: number }).etaSeconds ?? 240,
        estimatedCostUsd: (res as { estimatedCostUsd?: number }).estimatedCostUsd ?? 0.08,
        creditsRemainingUsd: (res as { creditsRemainingUsd?: number }).creditsRemainingUsd ?? 0,
        poll_with: "list_jobs",
      };
      return okResult(out);
    } catch (e) {
      return apiErrorToResult(e);
    }
  };
}
