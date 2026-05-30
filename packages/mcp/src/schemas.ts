// Shared Zod schemas per ARCHITECTURE.md §5.3
import { z } from "zod";

// --- SplatRef: discriminated union of how a scene is referenced ---
export const SplatRef = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("path"),
    path: z.string().describe("Absolute filesystem path. stdio transport only."),
  }),
  z.object({
    kind: z.literal("url"),
    url: z
      .string()
      .url()
      .describe("HTTPS URL to a .ply / .spz / .gltf / .glb file. Max 1.5 GB."),
  }),
  z.object({
    kind: z.literal("blob_id"),
    blob_id: z.string().describe("Blob key returned from a previous tool call."),
  }),
  z.object({
    kind: z.literal("content_b64"),
    content_b64: z.string().describe("Inline base64 bytes. Max 8 MB; use url/blob_id for larger."),
  }),
]);
export type SplatRefT = z.infer<typeof SplatRef>;

export const SceneFormat = z.enum(["ply", "spz", "gltf", "glb"]);
export type SceneFormatT = z.infer<typeof SceneFormat>;

export const FreePresetName = z.enum([
  "lossless-repack",
  "web-mobile",
  "web-desktop",
  "quest-browser",
  "visionos-preview",
  "thumbnail-preview",
  "quality-max",
  "size-min",
]);
export type FreePresetNameT = z.infer<typeof FreePresetName>;

export const PaidPresetName = z.enum([
  "differentiable-repack",
  "v52-quality",
  "v52-balanced",
  "t21r-fast",
]);
export type PaidPresetNameT = z.infer<typeof PaidPresetName>;

export const PresetName = z.union([FreePresetName, PaidPresetName]);
export type PresetNameT = z.infer<typeof PresetName>;

export const SizeReport = z.object({
  bytes: z.number().int().nonnegative(),
  mb: z.number(),
});

export const PredictedFidelity = z.object({
  meanDeltaE94: z.number(),
  p95DeltaE94: z.number().optional(),
  mlScore: z.number().min(0).max(1).nullable(),
  mlScoreVersion: z.string().optional(),
  confidence: z.enum(["low", "medium", "high"]),
  source: z.enum(["corpus-interp", "learned-predictor", "measured"]),
});
export type PredictedFidelityT = z.infer<typeof PredictedFidelity>;

export const JobId = z.string().uuid();
export const Cursor = z
  .string()
  .describe("Opaque pagination cursor. Pass back unchanged to fetch next page.");
