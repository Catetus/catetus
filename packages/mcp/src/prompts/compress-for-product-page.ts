// compress_for_product_page prompt — ARCHITECTURE.md §7
import { z } from "zod";

export const compressForProductPageMeta = {
  title: "Compress scenes for product page",
  description:
    "Given a list of .ply file paths/URLs and a per-file byte budget, run analyze → recommend_preset → encode for each. Optimized for product/marketing use.",
  argsSchema: {
    scenes: z.string().describe("Newline-separated paths or URLs to splat scenes."),
    target_bytes_per_scene: z
      .string()
      .default("30000000")
      .describe("Per-file byte budget. Parsed as int inside the handler."),
    device: z.string().default("desktop-web"),
  },
};

export function compressForProductPagePrompt(args: {
  scenes: string;
  target_bytes_per_scene?: string;
  device?: string;
}) {
  const sceneList = args.scenes
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  const budget = parseInt(args.target_bytes_per_scene ?? "30000000", 10);
  return {
    messages: [
      {
        role: "user" as const,
        content: {
          type: "text" as const,
          text: `I need to compress ${sceneList.length} 3DGS scenes for a product page.

Scenes:
${sceneList.map((s, i) => `  ${i + 1}. ${s}`).join("\n")}

Target: each output should be ≤ ${budget.toLocaleString()} bytes (${(budget / 1_048_576).toFixed(1)} MB) for ${args.device ?? "desktop-web"}.

Please use the Catetus MCP tools in this order:
  1. For each scene, call \`analyze\` to get metadata.
  2. For each scene, call \`recommend_preset\` with constraints { maxBytes: ${budget}, targetDevice: "${args.device ?? "desktop-web"}", allowPaid: true }.
  3. For each scene, call \`encode\` with the recommended preset and target.
  4. Summarize results in a table: scene | preset | bytes_in | bytes_out | ratio | predicted ΔE94.

Use parallel tool calls where possible.`,
        },
      },
    ],
  };
}
