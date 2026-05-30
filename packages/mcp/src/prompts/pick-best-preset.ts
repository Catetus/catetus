// pick_best_preset prompt — ARCHITECTURE.md §7
import { z } from "zod";

export const pickBestPresetMeta = {
  title: "Pick the best preset for this scene",
  description:
    "Run analyze, then recommend_preset with the constraints I describe. Returns rationale + alternative.",
  argsSchema: {
    scene: z.string().describe("Path or URL to the splat scene."),
    constraints_text: z
      .string()
      .describe("Free-form English; the agent extracts maxBytes/maxMeanDeltaE94/etc."),
  },
};

export function pickBestPresetPrompt(args: { scene: string; constraints_text: string }) {
  return {
    messages: [
      {
        role: "user" as const,
        content: {
          type: "text" as const,
          text: `Pick the best Catetus preset for this scene.

Scene: ${args.scene}

My constraints (in English): ${args.constraints_text}

Steps:
  1. Call \`analyze\` on the scene.
  2. Translate my constraints into the recommend_preset constraints object. Use:
     - maxBytes if I mention a byte/MB target
     - maxMeanDeltaE94 if I mention quality threshold (0.025 = pass, 0.05 = borderline)
     - minMlScore if I mention perceptual score
     - targetDevice if I mention mobile/desktop/quest/visionos
     - allowPaid: true unless I explicitly forbid paid presets
  3. Call \`recommend_preset\` and report the recommendation, alternative, and rationale.`,
        },
      },
    ],
  };
}
