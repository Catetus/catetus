// encode (paid) — ARCHITECTURE.md §5.4.9
import { z } from "zod";
import { okResult } from "../errors.js";
import { SplatRef, PresetName, JobId } from "../schemas.js";
import { sharedApiClient, apiErrorToResult } from "../api-client.js";
import type { TierContext } from "../auth.js";

export const encodeInput = {
  input: SplatRef,
  target: z
    .enum(["sog", "sog+v5tail", "v52", "t21r", "sf-baseline"])
    .default("sog")
    .describe("Encoded format. v52 = SOTA quality. sog+v5tail = SuperSplat-compatible w/ V5 sidecar."),
  preset: PresetName.optional(),
  output_format: z.enum(["gltf", "glb", "spz", "sog"]).optional(),
  webhook_url: z.string().url().optional(),
};

const outputRef = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("url"), url: z.string().url(), expires_at: z.string().datetime() }),
  z.object({ kind: z.literal("path"), path: z.string() }),
  z.object({ kind: z.literal("blob_id"), blob_id: z.string() }),
]);

export const encodeOutput = {
  mode: z.enum(["inline", "async"]),
  output: outputRef.optional(),
  bytesIn: z.number().int().optional(),
  bytesOut: z.number().int().optional(),
  ratio: z.number().optional(),
  report: z
    .object({
      passes: z.array(
        z.object({
          name: z.string(),
          ms: z.number().int(),
        }),
      ),
      totalMs: z.number().int(),
    })
    .optional(),
  fidelity: z
    .object({
      meanDeltaE94: z.number().optional(),
      psnr: z.number().optional(),
      mlScore: z.number().nullable(),
    })
    .optional(),
  costUsd: z.number().optional(),
  creditsRemainingUsd: z.number().optional(),
  // async-only:
  job_id: JobId.optional(),
  status: z.enum(["queued", "running"]).optional(),
  etaSeconds: z.number().int().optional(),
  estimatedCostUsd: z.number().optional(),
  poll_with: z.literal("list_jobs").optional(),
};

export const encodeMeta = {
  title: "Hosted encode (SOG/V5.2)",
  description:
    "Hosted encode on Modal-backed workers (api.catetus.com). Returns inline output for fast scenes, or a job_id you poll with `list_jobs`. target='v52' = +15.56 dB avg PSNR vs SOG at byte parity (canonical-11).",
  inputSchema: encodeInput,
  outputSchema: encodeOutput,
  annotations: {
    title: "Hosted encode (SOG/V5.2)",
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: true,
  },
};

type Args = {
  input: z.infer<typeof SplatRef>;
  target: "sog" | "sog+v5tail" | "v52" | "t21r" | "sf-baseline";
  preset?: z.infer<typeof PresetName>;
  output_format?: "gltf" | "glb" | "spz" | "sog";
  webhook_url?: string;
};

export function makeEncodeHandler(ctx: TierContext) {
  return async function encode(args: Args, extra?: { signal?: AbortSignal }) {
    try {
      const res = await sharedApiClient.call<Record<string, unknown>>(
        `/v1/encode?target=${encodeURIComponent(args.target)}`,
        {
          method: "POST",
          body: args,
          apiKey: ctx.apiKey,
          signal: extra?.signal,
          timeoutMs: 5 * 60_000, // up to 5 min for inline mode
        },
      );
      // Normalize: ensure `mode` discriminator is present so outputSchema validates.
      if (!("mode" in res)) {
        if ("job_id" in res) (res as Record<string, unknown>).mode = "async";
        else (res as Record<string, unknown>).mode = "inline";
      }
      return okResult(res);
    } catch (e) {
      return apiErrorToResult(e);
    }
  };
}
