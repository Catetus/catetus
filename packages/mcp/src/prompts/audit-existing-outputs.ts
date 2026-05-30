// audit_existing_outputs prompt — ARCHITECTURE.md §7
import { z } from "zod";

export const auditExistingOutputsMeta = {
  title: "Audit my Catetus outputs",
  description:
    "Given a directory of .gltf/.spz files, run validate_pipeline on each and summarize pass/fail.",
  argsSchema: {
    output_dir: z.string().describe("Path to a directory of compressed outputs."),
  },
};

export function auditExistingOutputsPrompt(args: { output_dir: string }) {
  return {
    messages: [
      {
        role: "user" as const,
        content: {
          type: "text" as const,
          text: `Audit the Catetus compression outputs in this directory:

  ${args.output_dir}

Steps:
  1. List all .gltf, .glb, .spz, .ply files in the directory (use your filesystem tools).
  2. For each file, call \`validate_pipeline\` with kind:"path" and expectedFormat matching the extension.
  3. Summarize: count passed/failed, list failed files with their failing checks + suggested fixes.

If validate_pipeline returns local_binary_missing, advise the user to install the Catetus CLI.`,
        },
      },
    ],
  };
}
