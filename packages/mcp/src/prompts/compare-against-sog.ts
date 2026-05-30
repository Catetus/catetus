// compare_against_sog prompt — ARCHITECTURE.md §7
import { z } from "zod";

export const compareAgainstSogMeta = {
  title: "Compare Catetus vs SuperSplat SOG",
  description:
    "Encode a scene with both Catetus V5.2 and PlayCanvas SOG, then `compare` and `score_fidelity` to produce a head-to-head report.",
  argsSchema: {
    scene: z.string().describe("Path or URL to the scene."),
    output_dir: z.string().default("./catetus-vs-sog"),
  },
};

export function compareAgainstSogPrompt(args: { scene: string; output_dir?: string }) {
  return {
    messages: [
      {
        role: "user" as const,
        content: {
          type: "text" as const,
          text: `Produce a head-to-head Catetus V5.2 vs PlayCanvas SOG report for this scene:

  ${args.scene}

Output directory: ${args.output_dir ?? "./catetus-vs-sog"}

Steps:
  1. Read the resource catetus://bench/canonical-11 for headline numbers context.
  2. Call \`analyze\` on the scene.
  3. Call \`encode\` twice: once with target="sog" and once with target="v52".
  4. Call \`score_fidelity\` on the SOG output (before=original, after=sog_output).
  5. Call \`score_fidelity\` on the V5.2 output (before=original, after=v52_output).
  6. Report: bytes, ΔE94 mean/p95, ML score for each, and PSNR delta if available.

Include the canonical-11 headline number ("Catetus V5.2 beats SOG by +15.56 dB on average").`,
        },
      },
    ],
  };
}
