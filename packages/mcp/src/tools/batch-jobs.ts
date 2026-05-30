// batch_jobs (paid) — ARCHITECTURE.md §5.4.14
import { z } from "zod";
import { okResult } from "../errors.js";
import { SplatRef, PresetName, JobId } from "../schemas.js";
import { ErrorEnvelope } from "../errors.js";
import { sharedApiClient, apiErrorToResult } from "../api-client.js";
import type { TierContext } from "../auth.js";

export const batchJobsInput = {
  jobs: z
    .array(
      z.object({
        input: SplatRef,
        target: z.enum(["sog", "sog+v5tail", "v52", "t21r", "sf-baseline"]),
        preset: PresetName.optional(),
      }),
    )
    .min(1)
    .max(100),
  max_wait_seconds: z.number().int().min(0).max(300).default(0),
  webhook_url: z.string().url().optional(),
};

const outputRef = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("url"), url: z.string().url(), expires_at: z.string().datetime() }),
  z.object({ kind: z.literal("path"), path: z.string() }),
  z.object({ kind: z.literal("blob_id"), blob_id: z.string() }),
]);

export const batchJobsOutput = {
  batch_id: z.string().uuid(),
  jobs: z.array(
    z.object({
      input_index: z.number().int(),
      job_id: JobId,
      status: z.enum(["queued", "running", "done", "error"]),
      output: outputRef.optional(),
      error: ErrorEnvelope.optional(),
    }),
  ),
  estimatedCostUsd: z.number(),
  creditsRemainingUsd: z.number(),
  next_action: z.enum(["wait", "poll_list_jobs", "done"]),
};

export const batchJobsMeta = {
  title: "Submit and poll a batch of encodes",
  description:
    "Submit a batch of encode/repack jobs. Returns a batch_id you poll with `list_jobs?batch_id=...`, or with max_wait_seconds > 0 waits inline up to N seconds.",
  inputSchema: batchJobsInput,
  outputSchema: batchJobsOutput,
  annotations: {
    title: "Submit and poll a batch of encodes",
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: true,
  },
};

type Args = z.infer<z.ZodObject<typeof batchJobsInput>>;

export function makeBatchJobsHandler(ctx: TierContext) {
  return async function batchJobs(args: Args, extra?: { signal?: AbortSignal }) {
    try {
      const res = await sharedApiClient.call<Record<string, unknown>>("/v1/jobs/batch", {
        method: "POST",
        body: args,
        apiKey: ctx.apiKey,
        signal: extra?.signal,
        timeoutMs: Math.max(30_000, args.max_wait_seconds * 1000 + 5_000),
      });
      // Normalize defaults so the outputSchema validates even if the API returns sparser shape.
      const normalized: Record<string, unknown> = {
        batch_id: (res as { batch_id?: string }).batch_id ?? "00000000-0000-0000-0000-000000000000",
        jobs: (res as { jobs?: unknown }).jobs ?? [],
        estimatedCostUsd: (res as { estimatedCostUsd?: number }).estimatedCostUsd ?? 0,
        creditsRemainingUsd: (res as { creditsRemainingUsd?: number }).creditsRemainingUsd ?? 0,
        next_action: (res as { next_action?: string }).next_action ?? "poll_list_jobs",
      };
      return okResult(normalized);
    } catch (e) {
      return apiErrorToResult(e);
    }
  };
}
