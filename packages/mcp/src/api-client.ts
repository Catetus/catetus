// Typed fetch wrapper for api.catetus.com per ARCHITECTURE.md §13.
// Retries 5xx with backoff+jitter, no retry on 4xx, AbortSignal-aware.

import { errorResult, type ErrorCode } from "./errors.js";

const DEFAULT_BASE = process.env.CATETUS_API_BASE ?? "https://api.catetus.com";
const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_RETRIES = 3;

export interface ApiCallOpts {
  method?: "GET" | "POST" | "DELETE" | "PUT" | "PATCH";
  body?: unknown;
  apiKey?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: ErrorCode,
    message: string,
    public readonly upstream?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class ApiClient {
  constructor(public readonly baseUrl: string = DEFAULT_BASE) {}

  async call<T = unknown>(path: string, opts: ApiCallOpts = {}): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const method = opts.method ?? "GET";
    const headers: Record<string, string> = {
      "user-agent": "catetus-mcp/1.0",
      ...opts.headers,
    };
    if (opts.apiKey) headers["authorization"] = `Bearer ${opts.apiKey}`;
    let bodyText: string | undefined;
    if (opts.body !== undefined) {
      bodyText = JSON.stringify(opts.body);
      headers["content-type"] = "application/json";
    }

    let lastErr: unknown;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (opts.signal?.aborted) throw new ApiError(0, "network_error", "Aborted");

      const ctrl = new AbortController();
      const composite = anySignal([opts.signal, ctrl.signal]);
      const timeout = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);

      try {
        const res = await fetch(url, { method, headers, body: bodyText, signal: composite });
        clearTimeout(timeout);
        if (res.status >= 200 && res.status < 300) {
          const ct = res.headers.get("content-type") ?? "";
          if (ct.includes("application/json")) {
            return (await res.json()) as T;
          }
          return (await res.text()) as unknown as T;
        }

        // Map common status → error codes
        let body: unknown;
        try {
          body = await res.json();
        } catch {
          body = await res.text().catch(() => undefined);
        }

        if (res.status === 401 || res.status === 403) {
          throw new ApiError(res.status, "auth_invalid", `Upstream ${res.status}`, body);
        }
        if (res.status === 402) {
          throw new ApiError(res.status, "insufficient_credits", `Upstream 402`, body);
        }
        if (res.status === 429) {
          throw new ApiError(res.status, "rate_limited", `Upstream 429`, body);
        }
        if (res.status === 404) {
          throw new ApiError(res.status, "not_yet_hosted", `Upstream 404 ${path}`, body);
        }
        if (res.status >= 500 && attempt < MAX_RETRIES) {
          // backoff w/ jitter, then retry
          const wait = backoff(attempt);
          await sleep(wait);
          continue;
        }
        throw new ApiError(
          res.status,
          res.status >= 500 ? "modal_unavailable" : "network_error",
          `Upstream HTTP ${res.status}`,
          body,
        );
      } catch (err) {
        clearTimeout(timeout);
        if (err instanceof ApiError) throw err;
        const aborted = (err as { name?: string })?.name === "AbortError";
        if (aborted) throw new ApiError(0, "network_error", "Request aborted/timed out");
        // Network-level: retry
        lastErr = err;
        if (attempt < MAX_RETRIES) {
          await sleep(backoff(attempt));
          continue;
        }
        throw new ApiError(0, "network_error", `Fetch failed: ${(err as Error).message ?? err}`);
      }
    }
    throw new ApiError(0, "network_error", `Exhausted retries: ${String(lastErr)}`);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function backoff(attempt: number): number {
  const base = 250 * Math.pow(2, attempt);
  const jitter = Math.random() * 250;
  return base + jitter;
}

function anySignal(signals: (AbortSignal | undefined)[]): AbortSignal {
  const ctrl = new AbortController();
  for (const s of signals) {
    if (!s) continue;
    if (s.aborted) {
      ctrl.abort();
      break;
    }
    s.addEventListener("abort", () => ctrl.abort(), { once: true });
  }
  return ctrl.signal;
}

/**
 * Tools that hit api.catetus.com share this helper to convert thrown ApiError
 * into the canonical MCP error envelope.
 */
export function apiErrorToResult(e: unknown) {
  if (e instanceof ApiError) {
    return errorResult(e.code, e.message, {
      hint: hintFor(e.code),
    });
  }
  return errorResult("network_error", `Unexpected: ${(e as Error).message ?? String(e)}`);
}

function hintFor(code: ErrorCode): string | undefined {
  switch (code) {
    case "auth_invalid":
      return "Issue a new key at https://catetus.com/dashboard.";
    case "insufficient_credits":
      return "Top up at https://catetus.com/dashboard/billing.";
    case "rate_limited":
      return "Back off and retry, or use an API key for higher limits.";
    case "modal_unavailable":
      return "Transient; retry with exponential backoff.";
    case "network_error":
      return "Check connectivity to api.catetus.com.";
    case "not_yet_hosted":
      return "This endpoint is not yet live on api.catetus.com; using local fallback.";
    default:
      return undefined;
  }
}

export const sharedApiClient = new ApiClient();
