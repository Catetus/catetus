// Structured error envelope + 29 canonical error codes per ARCHITECTURE.md §10.
// Every tool that returns { isError: true } MUST use errorResult() to wrap.

import { z } from "zod";
import { CATETUS_MCP_VERSION } from "./version.js";

// ---- Canonical error code registry ----
export const ERROR_CODES = [
  "scene_too_large",
  "unsupported_format",
  "malformed_input",
  "path_not_allowed",
  "unknown_preset",
  "optimize_failed",
  "output_too_small",
  "tier_not_available",
  "auth_invalid",
  "insufficient_credits",
  "modal_unavailable",
  "network_error",
  "predictor_unavailable",
  "corpus_too_sparse",
  "no_free_preset_fits",
  "no_preset_fits",
  "scene_not_found",
  "rate_limited",
  "render_failed",
  "format_mismatch",
  "unsupported_target",
  "model_unavailable",
  "batch_too_large",
  "partial_failure",
  "local_binary_missing",
  "use_encode_instead",
  "use_score_fidelity_instead",
  "invalid_cursor",
  "unsupported_input_format",
  "not_yet_hosted",
  "request_id_missing",
] as const;
export type ErrorCode = (typeof ERROR_CODES)[number];

export const ErrorEnvelope = z.object({
  code: z.string(),
  message: z.string(),
  hint: z.string().optional(),
  retry_after: z.number().optional(),
  docs_url: z.string().optional(),
  server_version: z.string(),
  request_id: z.string().optional(),
});
export type ErrorEnvelopeT = z.infer<typeof ErrorEnvelope>;

export interface ErrorOpts {
  hint?: string;
  retry_after?: number;
  docs_url?: string;
  request_id?: string;
}

/**
 * Build a structured MCP tool error result.
 * The envelope is JSON-stringified into content[0].text (LLM-readable) AND mirrored into structuredContent.
 */
export function errorResult(code: ErrorCode | string, message: string, opts: ErrorOpts = {}) {
  const envelope: ErrorEnvelopeT = {
    code,
    message,
    hint: opts.hint,
    retry_after: opts.retry_after,
    docs_url: opts.docs_url ?? `https://docs.catetus.com/mcp/errors#${code}`,
    server_version: CATETUS_MCP_VERSION,
    request_id: opts.request_id,
  };
  return {
    isError: true as const,
    content: [{ type: "text" as const, text: JSON.stringify(envelope, null, 2) }],
    structuredContent: envelope as Record<string, unknown>,
  };
}

/**
 * Convenience: an "ok" tool result.
 * Mirrors the structured payload into both content[0].text and structuredContent
 * per MCP 2025-11-25 spec recommendation.
 */
export function okResult<T extends Record<string, unknown>>(payload: T) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload, null, 2) }],
    structuredContent: payload,
  };
}
