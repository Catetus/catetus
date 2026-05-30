// Tier-1 static API-key tier resolution per ARCHITECTURE.md §4.
// v1 surface: presence-of-key check + scope filter. Upstream validation is deferred to
// the first paid-tool call (catches expired/revoked keys at the boundary).

export type Tier = "public" | "paid";

export interface TierContext {
  tier: Tier;
  apiKey?: string;
  /** When present, restricts paid tools to only those listed.  */
  scopes?: string[];
  remainingUsd?: number;
}

const PAID_SCOPES = [
  "encode",
  "score_fidelity",
  "repack",
  "predict",
  "batch",
] as const;
export type PaidScope = (typeof PAID_SCOPES)[number];

/**
 * Resolve tier from headers (HTTP transport) or env (stdio transport).
 * On stdio: CATETUS_API_KEY presence → paid.
 * On HTTP: Authorization: Bearer <key> presence → paid.
 *
 * v1 does not call upstream `/v1/me` — that's a Phase 2 wiring once we have key-cache.ts.
 * Paid tools individually surface auth_invalid / insufficient_credits on actual call.
 */
export function resolveTier(
  headers: Record<string, string | string[] | undefined> = {},
  env: NodeJS.ProcessEnv = process.env,
): TierContext {
  // 1) HTTP transport — Authorization header
  const authRaw = headers["authorization"] ?? headers["Authorization"];
  const auth = Array.isArray(authRaw) ? authRaw[0] : authRaw;
  if (auth && typeof auth === "string") {
    const m = auth.match(/^Bearer\s+(\S+)$/i);
    if (m) {
      return { tier: "paid", apiKey: m[1], scopes: [...PAID_SCOPES] };
    }
  }

  // 2) stdio transport — env var
  const envKey = env.CATETUS_API_KEY;
  if (envKey && envKey.trim().length > 0) {
    return { tier: "paid", apiKey: envKey.trim(), scopes: [...PAID_SCOPES] };
  }

  return { tier: "public" };
}

/** Map every tool to its required tier — drives tools/list filtering. */
export const TOOL_TIER: Record<string, { tier: Tier; scope?: PaidScope }> = {
  // Public
  analyze: { tier: "public" },
  list_presets: { tier: "public" },
  list_scenes: { tier: "public" },
  get_scene: { tier: "public" },
  optimize: { tier: "public" },
  compare: { tier: "public" },
  list_competitor_codecs: { tier: "public" },
  validate_pipeline: { tier: "public" },
  // Paid
  encode: { tier: "paid", scope: "encode" },
  score_fidelity: { tier: "paid", scope: "score_fidelity" },
  repack: { tier: "paid", scope: "repack" },
  predict_quality: { tier: "paid", scope: "predict" },
  recommend_preset: { tier: "paid", scope: "predict" },
  batch_jobs: { tier: "paid", scope: "batch" },
  list_jobs: { tier: "paid", scope: "batch" },
};

/** True iff a given tool is visible to the given tier context. */
export function isToolVisible(toolName: string, ctx: TierContext): boolean {
  const meta = TOOL_TIER[toolName];
  if (!meta) return false;
  if (meta.tier === "public") return true;
  if (ctx.tier !== "paid") return false;
  if (meta.scope && ctx.scopes && !ctx.scopes.includes(meta.scope)) return false;
  return true;
}
