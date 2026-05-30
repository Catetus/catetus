// list_jobs (paid, paginated) — ARCHITECTURE.md §5.5
import { z } from "zod";
import { okResult } from "../errors.js";
import { JobId, Cursor, PresetName } from "../schemas.js";
import { sharedApiClient, apiErrorToResult } from "../api-client.js";
import type { TierContext } from "../auth.js";

export const listJobsInput = {
  status: z.enum(["all", "queued", "running", "done", "error"]).default("all"),
  batch_id: z.string().uuid().optional(),
  since: z.string().datetime().optional(),
  cursor: Cursor.optional(),
  limit: z.number().int().min(1).max(50).default(20),
};

export const listJobsOutput = {
  jobs: z.array(
    z.object({
      job_id: JobId,
      status: z.enum(["queued", "running", "done", "error"]),
      target: z.string().optional(),
      preset: PresetName.optional(),
      created_at: z.string(),
      finished_at: z.string().optional(),
      output: z
        .discriminatedUnion("kind", [
          z.object({ kind: z.literal("url"), url: z.string().url(), expires_at: z.string().datetime() }),
          z.object({ kind: z.literal("path"), path: z.string() }),
          z.object({ kind: z.literal("blob_id"), blob_id: z.string() }),
        ])
        .optional(),
      error: z.unknown().optional(),
      costUsd: z.number().optional(),
    }),
  ),
  next_cursor: Cursor.optional(),
};

export const listJobsMeta = {
  title: "List recent jobs for the current API key",
  description:
    "List recent encode/repack jobs. Filter by status, batch_id, or date range. Use to poll for results when an original tool returned mode:'async'.",
  inputSchema: listJobsInput,
  outputSchema: listJobsOutput,
  annotations: {
    title: "List recent jobs",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: true,
  },
};

type Args = z.infer<z.ZodObject<typeof listJobsInput>>;

export function makeListJobsHandler(ctx: TierContext) {
  return async function listJobs(args: Args, extra?: { signal?: AbortSignal }) {
    try {
      const params = new URLSearchParams();
      if (args.status !== "all") params.set("status", args.status);
      if (args.batch_id) params.set("batch_id", args.batch_id);
      if (args.since) params.set("since", args.since);
      if (args.cursor) params.set("cursor", args.cursor);
      params.set("limit", String(args.limit));
      const res = await sharedApiClient.call<Record<string, unknown>>(`/v1/jobs?${params}`, {
        method: "GET",
        apiKey: ctx.apiKey,
        signal: extra?.signal,
      });
      const normalized: Record<string, unknown> = {
        jobs: (res as { jobs?: unknown }).jobs ?? [],
        next_cursor: (res as { next_cursor?: string }).next_cursor,
      };
      return okResult(normalized);
    } catch (e) {
      return apiErrorToResult(e);
    }
  };
}
