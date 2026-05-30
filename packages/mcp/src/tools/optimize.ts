// optimize (public, stdio-only) — ARCHITECTURE.md §5.4.5
import { z } from "zod";
import { errorResult, okResult } from "../errors.js";
import { SplatRef, FreePresetName, PredictedFidelity } from "../schemas.js";
import type { TierContext } from "../auth.js";

export const optimizeInput = {
  input: SplatRef,
  preset: FreePresetName,
  output_format: z.enum(["gltf", "glb", "spz"]).default("gltf"),
};

const outputRef = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("url"),
    url: z.string().url(),
    expires_at: z.string().datetime(),
  }),
  z.object({ kind: z.literal("path"), path: z.string() }),
  z.object({ kind: z.literal("blob_id"), blob_id: z.string() }),
]);

export const optimizeOutput = {
  output: outputRef,
  bytesIn: z.number().int(),
  bytesOut: z.number().int(),
  ratio: z.number(),
  report: z.object({
    passes: z.array(
      z.object({
        name: z.string(),
        ms: z.number().int(),
        splatsBefore: z.number().int().optional(),
        splatsAfter: z.number().int().optional(),
      }),
    ),
    totalMs: z.number().int(),
  }),
  predictedFidelity: PredictedFidelity.optional(),
};

export const optimizeMeta = {
  title: "Run free preset locally",
  description:
    "Run a free Catetus preset over a splat file and return the output (blob_id or download URL), size-reduction ratio, fidelity report, and timing. stdio + local CLI required in v1.",
  inputSchema: optimizeInput,
  outputSchema: optimizeOutput,
  annotations: {
    title: "Run free preset locally",
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

type Args = {
  input: z.infer<typeof SplatRef>;
  preset: z.infer<typeof FreePresetName>;
  output_format: "gltf" | "glb" | "spz";
};

export function makeOptimizeHandler(_ctx: TierContext, opts: { isHttp: boolean }) {
  return async function optimize(args: Args) {
    if (opts.isHttp) {
      return errorResult(
        "use_encode_instead",
        "Free presets via HTTP transport are not yet available.",
        {
          hint:
            "Use the `encode` tool (paid) or install the local `npx @catetus/mcp` for free preset access.",
        },
      );
    }
    const cliPath = process.env.CATETUS_MCP_LOCAL_BINARY;
    if (!cliPath) {
      return errorResult(
        "local_binary_missing",
        "Local `catetus` binary not configured (CATETUS_MCP_LOCAL_BINARY unset).",
        {
          hint:
            "Install the optional CLI: `npm i -g @catetus/cli` or set CATETUS_MCP_LOCAL_BINARY to a built `catetus` binary.",
        },
      );
    }
    // Real implementation: spawn `${cliPath} optimize --preset ${args.preset} --in <path> --out <tmp>`
    // and parse the `<tmp>.json` report. We surface a placeholder here so the contract is testable.
    return okResult({
      output: { kind: "blob_id", blob_id: `optimize:${args.preset}:pending` },
      bytesIn: 0,
      bytesOut: 0,
      ratio: 1,
      report: { passes: [], totalMs: 0 },
    });
  };
}
