/**
 * Thin client for the customer-facing dashboard JSON endpoint.
 *
 * Talks to `GET /v1/me/usage` on catetus-api. The bearer token is
 * read from `localStorage.catetus.apiKey` — the same key the
 * /pricing and /import pages persist after Stripe Checkout reveals
 * it. Base URL falls back to the production host so the page works
 * out of the box on catetus.com; a contributor running a local
 * API sets `localStorage.catetus.apiBase` to override.
 *
 * Zero runtime dependencies on purpose: this gets inlined into the
 * static Astro bundle and runs client-side after the page hydrates.
 */

/** Backend shape — kept in lockstep with `customer_dashboard::DashboardResponse`. */
export interface DashboardResponse {
  plan: "free" | "paid";
  key_masked: string;
  email?: string;
  usage: {
    repack_runs: number;
    repack_seconds: number;
    period_start?: string;
  };
  recent_jobs: RecentJob[];
}

export interface RecentJob {
  timestamp: string;
  route: string;
  method: string;
  status: number;
  duration_ms: number;
  error?: string;
}

export const DASHBOARD_API_KEY_STORAGE = "catetus.apiKey";
export const DASHBOARD_API_BASE_STORAGE = "catetus.apiBase";
export const DEFAULT_API_BASE = "https://catetus-api.fly.dev";

/** True if the visitor has an API key in localStorage. */
export function hasApiKey(): boolean {
  if (typeof localStorage === "undefined") return false;
  const v = localStorage.getItem(DASHBOARD_API_KEY_STORAGE);
  return Boolean(v && v.trim().length > 0);
}

/** Stable masked form for display. Mirrors `ratelimit::key_prefix`
 *  (first 8 chars + ellipsis) so the page label matches what the
 *  backend logs. */
export function maskApiKey(key: string): string {
  if (!key) return "";
  const prefix = key.slice(0, 8);
  return `${prefix}…`;
}

export interface FetchDashboardOptions {
  /** How many recent-jobs rows to ask for. Backend caps at 100. */
  limit?: number;
  /** Override the base URL — used by tests. */
  baseUrl?: string;
}

/**
 * Fetch the dashboard JSON for the visitor's stored API key.
 * Throws a typed error so the page can render the right state.
 */
export class DashboardAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DashboardAuthError";
  }
}
export class DashboardServerError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "DashboardServerError";
  }
}

export async function fetchDashboard(
  opts: FetchDashboardOptions = {},
): Promise<DashboardResponse> {
  const apiKey = localStorage.getItem(DASHBOARD_API_KEY_STORAGE) ?? "";
  if (!apiKey.trim()) {
    throw new DashboardAuthError("no API key in localStorage");
  }
  const base =
    opts.baseUrl ??
    localStorage.getItem(DASHBOARD_API_BASE_STORAGE) ??
    DEFAULT_API_BASE;
  const url = new URL("/v1/me/usage", base);
  if (opts.limit) url.searchParams.set("limit", String(opts.limit));

  const resp = await fetch(url.toString(), {
    method: "GET",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: "application/json",
    },
  });
  if (resp.status === 401 || resp.status === 403) {
    throw new DashboardAuthError(`API rejected key: HTTP ${resp.status}`);
  }
  if (!resp.ok) {
    const body = await resp.text();
    throw new DashboardServerError(resp.status, body || `HTTP ${resp.status}`);
  }
  return (await resp.json()) as DashboardResponse;
}

/** Format seconds into a human-readable duration ("1m 23s" / "42s"). */
export function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return s === 0 ? `${m}m` : `${m}m ${s}s`;
}

/** Format an ISO timestamp into a short local-time stamp. */
export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Daily quota for the free tier, surfaced on the dashboard.
 *  Mirrors the pricing page copy — keep in sync. */
export const FREE_TIER_QUOTA = {
  scenesPerDay: 1,
  bundlesPerMonth: 1,
};
