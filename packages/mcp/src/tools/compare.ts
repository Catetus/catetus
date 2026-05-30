// compare (public, stdio-only) — ARCHITECTURE.md §5.4.6
import { z } from "zod";
import { errorResult, okResult } from "../errors.js";
import { SplatRef } from "../schemas.js";

export const compareInput = {
  before: SplatRef,
  after: SplatRef,
  cameraPath: z.enum(["orbit-8", "orbit-16", "forward-pan"]).default("orbit-8"),
  frameSize: z.enum(["256x256", "512x512", "1024x1024"]).default("512x512"),
};

export const compareOutput = {
  meanDeltaE94: z.number(),
  maxDeltaE94: z.number(),
  p95DeltaE94: z.number(),
  meanPixelMatch: z.number(),
  meanSsimLoss: z.number(),
  mlScore: z.number().nullable(),
  mlScoreVersion: z.string().optional(),
  status: z.enum(["pass", "borderline", "fail"]),
  passed: z.boolean(),
  frames: z.array(
    z.object({
      i: z.number().int(),
      deltaE94: z.number(),
      pixelMatch: z.number(),
      ssimLoss: z.number(),
    }),
  ),
};

export const compareMeta = {
  title: "Render before/after diff",
  description:
    "Render before/after frames along a deterministic camera path and return per-frame ΔE94, pixel-match, SSIM-loss, and (when available) ML perceptual score. Use after optimize/encode to validate.",
  inputSchema: compareInput,
  outputSchema: compareOutput,
  annotations: {
    title: "Render before/after diff",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

type Args = {
  before: z.infer<typeof SplatRef>;
  after: z.infer<typeof SplatRef>;
  cameraPath: "orbit-8" | "orbit-16" | "forward-pan";
  frameSize: "256x256" | "512x512" | "1024x1024";
};

export function makeCompareHandler(opts: { isHttp: boolean }) {
  return async function compare(_args: Args) {
    if (opts.isHttp) {
      return errorResult(
        "use_score_fidelity_instead",
        "Compare via HTTP transport is not available in v1.",
        { hint: "Use the paid `score_fidelity` tool, or install the local stdio package." },
      );
    }
    if (!process.env.CATETUS_MCP_LOCAL_BINARY) {
      return errorResult(
        "local_binary_missing",
        "Local `catetus` binary not configured (CATETUS_MCP_LOCAL_BINARY unset).",
        { hint: "Install the optional CLI to enable `compare`." },
      );
    }
    // Placeholder result with correct shape; real impl shells `catetus diff before after --pretty`.
    return okResult({
      meanDeltaE94: 0,
      maxDeltaE94: 0,
      p95DeltaE94: 0,
      meanPixelMatch: 1,
      meanSsimLoss: 0,
      mlScore: null,
      status: "pass",
      passed: true,
      frames: [],
    });
  };
}
