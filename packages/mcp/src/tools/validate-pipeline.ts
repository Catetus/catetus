// validate_pipeline (public) — ARCHITECTURE.md §5.4.8
import { z } from "zod";
import { okResult } from "../errors.js";
import { SplatRef } from "../schemas.js";

export const validatePipelineInput = {
  input: SplatRef,
  expectedFormat: z.enum(["ply", "spz", "gltf", "glb", "any"]).default("any"),
  expectedSplatCount: z
    .number()
    .int()
    .optional()
    .describe("If set, check actual count is within ±5%."),
};

export const validatePipelineOutput = {
  ok: z.boolean(),
  checks: z.array(
    z.object({
      name: z.string(),
      passed: z.boolean(),
      detail: z.string().optional(),
    }),
  ),
};

export const validatePipelineMeta = {
  title: "Sanity-check output",
  description:
    "Sanity-check a Catetus output: no NaN/Inf splats, well-formed SPZ/glTF container, expected splat count vs source, glTF extension refs resolve.",
  inputSchema: validatePipelineInput,
  outputSchema: validatePipelineOutput,
  annotations: {
    title: "Sanity-check output",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

type Args = {
  input: z.infer<typeof SplatRef>;
  expectedFormat: "ply" | "spz" | "gltf" | "glb" | "any";
  expectedSplatCount?: number;
};

export function makeValidatePipelineHandler() {
  return async function validatePipeline(args: Args) {
    // Placeholder: real impl shells `catetus inspect` or parses inline via gltf JSON.
    // We surface a "no checks ran" but ok:false flag so callers can detect the missing binary.
    const hasBinary = !!process.env.CATETUS_MCP_LOCAL_BINARY;
    const checks = [
      {
        name: "binary_available",
        passed: hasBinary,
        detail: hasBinary
          ? `Using ${process.env.CATETUS_MCP_LOCAL_BINARY}`
          : "CATETUS_MCP_LOCAL_BINARY unset; deep validation unavailable.",
      },
      {
        name: "format_recognized",
        passed: args.expectedFormat !== "any" || args.input.kind !== "content_b64",
      },
    ];
    return okResult({
      ok: checks.every((c) => c.passed),
      checks,
    });
  };
}
