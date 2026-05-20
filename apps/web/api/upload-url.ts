// Issue a short-lived presigned PUT URL so the browser can stream the
// uploaded splat scene directly to Cloudflare R2 — bypassing both
// Vercel's edge proxy (which has body limits) and our optimize API
// host (which is HTTP-only and would block on mixed content).
//
// Required env vars: R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID,
// R2_SECRET_ACCESS_KEY. R2 bucket CORS must allow PUT+GET from the
// site origin.

import type { IncomingMessage, ServerResponse } from "node:http";
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const ALLOWED_EXTS = [
  ".ply", ".spz", ".gltf", ".glb", ".splat",
  // Bundle uploads: hacpp-lzma (Scaffold-GS .tar bundle) + qat-bundle (PLY +
  // cameras.txt + images.txt + image folder, tar.gz) + ios-capture (.zip
  // of phone photos for the server-side COLMAP → Scaffold → QAT path).
  ".tar", ".tar.gz", ".tgz", ".zip",
];
const MAX_SIZE = 5 * 1024 * 1024 * 1024; // 5 GB (R2 single-PUT cap)

interface SignBody {
  filename?: unknown;
  contentType?: unknown;
  size?: unknown;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(body));
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
  return Math.random().toString(36).slice(2, 18);
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    total += buf.length;
    if (total > 8 * 1024) throw new Error("request body too large");
    chunks.push(buf);
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw) return {};
  return JSON.parse(raw);
}

export default async function handler(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  if (req.method !== "POST") {
    res.statusCode = 405;
    res.end("method not allowed");
    return;
  }

  let body: SignBody;
  try {
    body = (await readJsonBody(req)) as SignBody;
  } catch (err) {
    sendJson(res, 400, { error: `invalid JSON body: ${(err as Error).message}` });
    return;
  }

  const { filename, contentType, size } = body;
  if (typeof filename !== "string" || filename.length === 0 || filename.length > 256) {
    sendJson(res, 400, { error: "filename: required string, <=256 chars" });
    return;
  }
  if (
    typeof size !== "number" ||
    !Number.isFinite(size) ||
    size <= 0 ||
    size > MAX_SIZE
  ) {
    sendJson(res, 400, { error: `size: required positive number <= ${MAX_SIZE}` });
    return;
  }
  const ext = safeExt(filename);
  if (!ext) {
    sendJson(res, 400, {
      error: `unsupported extension; allowed: ${ALLOWED_EXTS.join(" ")}`,
    });
    return;
  }

  const bucket = process.env.R2_BUCKET;
  if (!bucket) {
    sendJson(res, 500, { error: "R2_BUCKET not configured on server" });
    return;
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
      { expiresIn: 60 * 60 },
    );
    const sourceUrl = await getSignedUrl(
      client,
      new GetObjectCommand({ Bucket: bucket, Key: key }),
      { expiresIn: 60 * 60 * 6 },
    );
    sendJson(res, 200, { uploadUrl, sourceUrl, key, contentType: ctype });
  } catch (err) {
    sendJson(res, 500, { error: (err as Error).message });
  }
}
