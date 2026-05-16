// POST /api/v1/jobs — Vercel function that proxies anonymous TryIt
// drag-drop traffic to the auth-gated Fly API. Adds a server-side
// Bearer (SPLATFORGE_DEMO_API_KEY) the browser never sees. The Fly
// rate-limiter buckets by this key, so the demo lane shares one quota
// across all anonymous visitors — protective against abuse, and
// distinct from paid keys.
//
// Why this exists: vercel.json rewrites /api/v1/* to the Fly host
// for trivial pass-through routes (pricing previews, checkout, ratings —
// all unauth on Fly). But /v1/jobs is bearer-gated on Fly, so the
// anonymous browser fetch returns 401 ("could not start job"). This
// function takes precedence over the rewrite (Vercel functions win)
// and injects the demo Bearer server-side.

import type { IncomingMessage, ServerResponse } from "node:http";

const UPSTREAM = "https://splatforge-api.fly.dev/v1/jobs";

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(body));
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    total += buf.length;
    if (total > 64 * 1024) throw new Error("request body too large");
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString("utf8");
}

export default async function handler(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  if (req.method !== "POST") {
    sendJson(res, 405, { error: "method not allowed" });
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

  let raw: string;
  try {
    raw = await readBody(req);
  } catch (err) {
    sendJson(res, 413, { error: (err as Error).message });
    return;
  }

  const fwdHeaders: Record<string, string> = {
    "content-type": "application/json",
    authorization: `Bearer ${bearer}`,
  };
  // Forward client IP so the Fly side has a hope of per-IP rate
  // limiting later, even though today the bucket is keyed by Bearer.
  const xff = req.headers["x-forwarded-for"];
  if (typeof xff === "string") fwdHeaders["x-forwarded-for"] = xff;
  const ua = req.headers["user-agent"];
  if (typeof ua === "string") fwdHeaders["user-agent"] = ua;

  let upstream: Response;
  try {
    upstream = await fetch(UPSTREAM, {
      method: "POST",
      headers: fwdHeaders,
      body: raw,
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
