// GET /api/v1/jobs/:id — Vercel function that proxies anonymous TryIt
// status polls to the auth-gated Fly API. Pairs with the POST /v1/jobs
// handler in ../jobs.ts; same demo Bearer is injected so the browser
// can poll its own anonymous job without holding a key.

import type { IncomingMessage, ServerResponse } from "node:http";

const UPSTREAM_BASE = "https://splatforge-api.fly.dev/v1/jobs/";

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(body));
}

// Vercel passes path params via req.url's query string when using
// `[param].ts` filename routing — the file is loaded as a serverless
// function and the dynamic segment is exposed as `?id=...` on req.url.
function extractId(req: IncomingMessage): string | null {
  if (!req.url) return null;
  const u = new URL(req.url, "http://x");
  const q = u.searchParams.get("id");
  if (q && /^[A-Za-z0-9_-]{1,128}$/.test(q)) return q;
  // Fallback: last path segment of the URL (handles bare-path routing).
  const parts = u.pathname.split("/").filter(Boolean);
  const tail = parts[parts.length - 1];
  if (tail && /^[A-Za-z0-9_-]{1,128}$/.test(tail)) return tail;
  return null;
}

export default async function handler(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  if (req.method !== "GET") {
    sendJson(res, 405, { error: "method not allowed" });
    return;
  }

  const id = extractId(req);
  if (!id) {
    sendJson(res, 400, { error: "invalid job id" });
    return;
  }

  const bearer = process.env.SPLATFORGE_DEMO_API_KEY;
  if (!bearer) {
    sendJson(res, 503, {
      error:
        "demo Bearer not configured on server — set SPLATFORGE_DEMO_API_KEY in Vercel env and include the same value in SPLATFORGE_API_KEYS on the Fly API.",
    });
    return;
  }

  let upstream: Response;
  try {
    upstream = await fetch(UPSTREAM_BASE + encodeURIComponent(id), {
      method: "GET",
      headers: { authorization: `Bearer ${bearer}` },
    });
  } catch (err) {
    sendJson(res, 502, {
      error: `upstream optimize API unreachable: ${(err as Error).message}`,
    });
    return;
  }

  const text = await upstream.text();
  res.statusCode = upstream.status;
  const ct = upstream.headers.get("content-type") ?? "application/json";
  res.setHeader("content-type", ct);
  res.end(text);
}
