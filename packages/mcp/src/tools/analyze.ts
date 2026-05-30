// analyze (public) — ARCHITECTURE.md §5.4.1
import { z } from "zod";
import { errorResult, okResult } from "../errors.js";
import { SplatRef, SceneFormat } from "../schemas.js";
import { apiErrorToResult, sharedApiClient } from "../api-client.js";
import type { TierContext } from "../auth.js";

export const analyzeInput = {
  input: SplatRef,
  include_recommendations: z
    .boolean()
    .default(true)
    .describe("Include recommendations block. Off for size-sensitive callers."),
};

export const analyzeOutput = {
  format: SceneFormat,
  splatCount: z.number().int().nonnegative(),
  bytesIn: z.number().int().nonnegative(),
  hash: z.string().regex(/^blake3:[0-9a-f]{64}$/),
  shDegree: z.number().int().min(0).max(3),
  bbox: z.object({
    min: z.tuple([z.number(), z.number(), z.number()]),
    max: z.tuple([z.number(), z.number(), z.number()]),
  }),
  opacityHistogram: z.array(z.number().int()).length(16),
  analyzeMs: z.number().int(),
  recommendations: z
    .object({
      likelyClass: z.enum([
        "product-scan",
        "indoor-real-estate",
        "outdoor-scene",
        "object-isolated",
        "transparent-volume",
        "portrait",
        "other",
      ]),
      needsFloaterPrune: z.boolean(),
      needsOpacityPrune: z.boolean(),
    })
    .optional(),
};

export const analyzeMeta = {
  title: "Analyze splat scene",
  description:
    "Analyze a Gaussian Splat file (.ply / .spz / .gltf / .glb) and return splat count, bounding box, SH degree, opacity distribution, content hash, and rough recommendations. Read-only. Use this first when the user provides a scene you don't already know — every other tool benefits from the metadata.",
  inputSchema: analyzeInput,
  outputSchema: analyzeOutput,
  annotations: {
    title: "Analyze splat scene",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

type AnalyzeArgs = { input: z.infer<typeof SplatRef>; include_recommendations: boolean };

export function makeAnalyzeHandler(_ctx: TierContext) {
  return async function analyze(args: AnalyzeArgs) {
    try {
      // Try hosted endpoint first (architecture says it doesn't exist yet — will 404).
      const res = await sharedApiClient.call<Record<string, unknown>>("/v1/analyze", {
        method: "POST",
        body: { input: args.input, include_recommendations: args.include_recommendations },
        timeoutMs: 10_000,
      });
      return okResult(res);
    } catch (e) {
      // Fallback: in-process stub (real impl: ply-stream parse or shell to `catetus analyze`).
      // ARCHITECTURE §5.4.1 explicitly permits an in-process fallback in v1.
      return fallbackAnalyze(args);
    }
    void apiErrorToResult; // imported for future paid-path use
  };
}

function fallbackAnalyze(args: AnalyzeArgs) {
  // v1 fallback returns a structured "not_yet_hosted" sentinel with the expected envelope shape.
  // Implementer roadmap: wire ply-stream or `catetus analyze` shellout in 1.0.0.
  if (args.input.kind === "path" || args.input.kind === "content_b64") {
    return errorResult(
      "not_yet_hosted",
      "POST /v1/analyze is not yet live and the in-process PLY parser is not yet bundled in this MCP server.",
      {
        hint:
          "Use the local `catetus` CLI directly for now, or set CATETUS_API_BASE to a staging API that exposes /v1/analyze.",
      },
    );
  }
  return errorResult(
    "not_yet_hosted",
    "Analyze fallback for remote inputs is pending Phase 2.",
    { hint: "Pass kind:'path' with a local `catetus` binary on PATH once the CLI shellout lands." },
  );
}
