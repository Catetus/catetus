// Issue a short-lived presigned PUT URL so the browser can stream the
// uploaded splat scene **directly** to Cloudflare R2 — bypassing both
// Vercel's edge proxy (which has body limits) and our optimize API
// host (which is HTTP-only and would block on mixed content).
//
// Architecture:
//
//   browser  --POST /api/upload-url--->   this function (Vercel)
//   browser  <-- { uploadUrl, sourceUrl, key }  ---
//   browser  --PUT  bytes to uploadUrl ->  Cloudflare R2 (HTTPS)
//   browser  --POST /api/v1/jobs (source_url=sourceUrl)-->  Rust optimize API
//                                                 |
//                                                 v
//                                            GET sourceUrl  (server-side R2 fetch)
//
// R2 is S3-compatible with $0 egress, so this scales without bandwidth
// cost. The 5 GB-per-object limit applies; use multipart for larger
// files when we get there.
//
// Required Vercel env vars (configure in project settings):
//
//   R2_ACCOUNT_ID         - Cloudflare account ID (URL fragment in dashboard)
//   R2_BUCKET             - bucket name, e.g. "splatforge-uploads"
//   R2_ACCESS_KEY_ID      - S3-compatible API token, "Object Read & Write"
//   R2_SECRET_ACCESS_KEY  - secret half of the token
//
// Required R2 bucket CORS (set once via the dashboard or wrangler):
//
//   AllowedOrigins: ["https://splatforge.dev", "http://localhost:4321"]
//   AllowedMethods: ["PUT", "GET"]
//   AllowedHeaders: ["*"]
//   MaxAgeSeconds: 3600
//
// Bucket lifecycle (recommended): purge `uploads/` after 24 hours so
// stale unconsumed uploads don't accumulate.

import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const ALLOWED_EXTS = [".ply", ".spz", ".gltf", ".glb", ".splat"];
const MAX_SIZE = 5 * 1024 * 1024 * 1024; // 5 GB (R2 single-PUT cap)

interface SignBody {
  filename?: unknown;
  contentType?: unknown;
  size?: unknown;
}

function badRequest(msg: string): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status: 400,
    headers: { "content-type": "application/json" },
  });
}

function r2Client(): S3Client {
  const accountId = process.env.R2_ACCOUNT_ID;
  const accessKeyId = process.env.R2_ACCESS_KEY_ID;
  const secretAccessKey = process.env.R2_SECRET_ACCESS_KEY;
  if (!accountId || !accessKeyId || !secretAccessKey) {
    throw new Error(
      "R2 credentials missing: set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in Vercel env",
    );
  }
  return new S3Client({
    region: "auto",
    endpoint: `https://${accountId}.r2.cloudflarestorage.com`,
    credentials: { accessKeyId, secretAccessKey },
  });
}

function safeExt(filename: string): string | null {
  const dot = filename.lastIndexOf(".");
  if (dot <= 0) return null;
  const ext = filename.slice(dot).toLowerCase();
  return ALLOWED_EXTS.includes(ext) ? ext : null;
}

function randomKeyId(): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  if (uuid) return uuid.replace(/-/g, "").slice(0, 16);
  // Fallback if crypto.randomUUID is unavailable (shouldn't happen on Node 18+).
  return Math.random().toString(36).slice(2, 18);
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method !== "POST") {
    return new Response("method not allowed", { status: 405 });
  }

  let body: SignBody;
  try {
    body = (await req.json()) as SignBody;
  } catch {
    return badRequest("body must be JSON");
  }
  const { filename, contentType, size } = body;
  if (typeof filename !== "string" || filename.length === 0 || filename.length > 256) {
    return badRequest("filename: required string, ≤256 chars");
  }
  if (
    typeof size !== "number" ||
    !Number.isFinite(size) ||
    size <= 0 ||
    size > MAX_SIZE
  ) {
    return badRequest(`size: required positive number ≤ ${MAX_SIZE}`);
  }
  const ext = safeExt(filename);
  if (!ext) {
    return badRequest(
      `unsupported extension; allowed: ${ALLOWED_EXTS.join(" ")}`,
    );
  }

  const bucket = process.env.R2_BUCKET;
  if (!bucket) {
    return new Response(
      JSON.stringify({ error: "R2_BUCKET not configured on server" }),
      { status: 500, headers: { "content-type": "application/json" } },
    );
  }
  const key = `uploads/${Date.now().toString(36)}-${randomKeyId()}${ext}`;
  const ctype =
    typeof contentType === "string" && contentType.length > 0
      ? contentType
      : "application/octet-stream";

  try {
    const client = r2Client();
    const uploadUrl = await getSignedUrl(
      client,
      new PutObjectCommand({ Bucket: bucket, Key: key, ContentType: ctype }),
      { expiresIn: 60 * 60 }, // 1 h to upload
    );
    const sourceUrl = await getSignedUrl(
      client,
      new GetObjectCommand({ Bucket: bucket, Key: key }),
      { expiresIn: 60 * 60 * 6 }, // 6 h for the API to fetch + process
    );
    return new Response(
      JSON.stringify({
        uploadUrl,
        sourceUrl,
        key,
        contentType: ctype,
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  } catch (err) {
    return new Response(
      JSON.stringify({ error: (err as Error).message }),
      { status: 500, headers: { "content-type": "application/json" } },
    );
  }
}

export const config = { runtime: "nodejs" };
