#!/usr/bin/env node
// upload-cesium-ion.mjs
//
// Uploads a Catetus-generated 3D Tiles tileset (with KHR_gaussian_splatting)
// to Cesium ion, then writes the resulting asset id to
// apps/web/src/data/cesium-ion-asset.json so vs-cesium-ion.astro can embed a
// live ion viewer iframe.
//
// Flow follows https://cesium.com/learn/ion/rest-api/ asset-creation:
//   1. POST /v1/assets   sourceType=3D_TILES + geolocation in options
//   2. PUT  archive.tar.gz to the S3 URL returned in `uploadLocation`
//   3. POST /v1/assets/{id}/uploadComplete  to trigger ion processing
//   4. GET  /v1/assets/{id}  poll until status=COMPLETE
//
// Idempotency: if `cesium-ion-asset.json` already exists with an `assetId`,
// the script no-ops unless `--force` is passed.
//
// Env: CESIUM_ION_TOKEN must be set with `assets:write` scope.
//
// Usage:
//   CESIUM_ION_TOKEN=... node apps/web/scripts/upload-cesium-ion.mjs \
//     --tileset apps/web/scripts/tmp/bonsai-tileset \
//     --name "Catetus bonsai_real demo" \
//     --lon -122.4194 --lat 37.7749
//
// Constraints honoured per repo conventions:
//   - No /tmp paths in checked-in code; tmp artifacts live under
//     apps/web/scripts/tmp/ (gitignored).
//   - Script is self-contained: only depends on Node 20+ built-ins
//     (fetch, fs/promises, child_process).
//
// If CESIUM_ION_TOKEN is unset, the script exits 0 with a clear message
// explaining how the operator can run it locally — the site already has a
// fallback path for when no asset id is present.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { promises as fs } from "node:fs";
import path from "node:path";
import process from "node:process";

const execFileP = promisify(execFile);

const REPO_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../../..");
const WEB_ROOT = path.join(REPO_ROOT, "apps", "web");
const ASSET_JSON = path.join(WEB_ROOT, "src", "data", "cesium-ion-asset.json");
const DEFAULT_TILESET = path.join(WEB_ROOT, "scripts", "tmp", "bonsai-tileset");
const DEFAULT_TMP = path.join(WEB_ROOT, "scripts", "tmp");

function parseArgs(argv) {
  const args = {
    tileset: DEFAULT_TILESET,
    name: "Catetus bonsai_real demo",
    description:
      "Live KHR_gaussian_splatting tileset emitted by Catetus from the Mip-NeRF 360 bonsai scene (iter7000, 1.06M splats, 4-level LOD).",
    lon: -122.4194,
    lat: 37.7749,
    height: 0,
    force: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case "--tileset":
        args.tileset = path.resolve(next());
        break;
      case "--name":
        args.name = next();
        break;
      case "--description":
        args.description = next();
        break;
      case "--lon":
        args.lon = parseFloat(next());
        break;
      case "--lat":
        args.lat = parseFloat(next());
        break;
      case "--height":
        args.height = parseFloat(next());
        break;
      case "--force":
        args.force = true;
        break;
      case "-h":
      case "--help":
        console.log(
          "usage: upload-cesium-ion.mjs [--tileset DIR] [--name STR] [--lon F] [--lat F] [--height F] [--force]",
        );
        process.exit(0);
        break;
      default:
        console.error(`unknown arg: ${a}`);
        process.exit(2);
    }
  }
  return args;
}

async function readJsonIfExists(p) {
  try {
    return JSON.parse(await fs.readFile(p, "utf8"));
  } catch (err) {
    if (err.code === "ENOENT") return null;
    throw err;
  }
}

async function tarballTileset(tilesetDir, outTar) {
  // Use `tar` from the OS. The archive must NOT contain a top-level directory;
  // ion expects to find tileset.json at the root of the archive.
  // We cd into tilesetDir and tar `.` so paths are relative.
  await fs.mkdir(path.dirname(outTar), { recursive: true });
  await execFileP("tar", ["-czf", outTar, "-C", tilesetDir, "."]);
  const st = await fs.stat(outTar);
  return st.size;
}

async function ionRequest(token, method, urlPath, body) {
  const url = urlPath.startsWith("http") ? urlPath : `https://api.cesium.com${urlPath}`;
  const headers = { Authorization: `Bearer ${token}` };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(url, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ion ${method} ${urlPath} -> ${res.status}: ${text}`);
  }
  if (res.status === 204) return null;
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

async function s3PutTarball(uploadLocation, tarballPath) {
  // ion returns an AWS S3 prefix + temporary credentials. The simplest path is
  // to shell out to `aws s3 cp`, which respects the supplied credentials via
  // env vars. We avoid the AWS SDK to keep this script dep-free.
  const { endpoint, bucket, prefix, accessKey, secretAccessKey, sessionToken } = uploadLocation;
  const dest = `s3://${bucket}/${prefix}archive.tar.gz`.replace(/([^:])\/\//g, "$1/");
  const env = {
    ...process.env,
    AWS_ACCESS_KEY_ID: accessKey,
    AWS_SECRET_ACCESS_KEY: secretAccessKey,
    AWS_SESSION_TOKEN: sessionToken,
  };
  if (endpoint) {
    env.AWS_ENDPOINT_URL = endpoint;
  }
  console.log(`s3 cp ${tarballPath} -> ${dest}`);
  await execFileP("aws", ["s3", "cp", tarballPath, dest], { env, maxBuffer: 1024 * 1024 * 16 });
}

async function pollUntilComplete(token, assetId, { timeoutMs = 30 * 60 * 1000, intervalMs = 5000 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const asset = await ionRequest(token, "GET", `/v1/assets/${assetId}`);
    const status = asset.status || asset.state;
    console.log(`asset ${assetId} status=${status} progress=${asset.percentComplete ?? "?"}`);
    if (status === "COMPLETE") return asset;
    if (status === "ERROR" || status === "DATA_ERROR") {
      throw new Error(`ion processing failed: ${JSON.stringify(asset)}`);
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`ion processing timed out after ${timeoutMs}ms`);
}

async function main() {
  const args = parseArgs(process.argv);

  // Idempotency: skip if we already have an asset id, unless --force.
  const existing = await readJsonIfExists(ASSET_JSON);
  if (existing && existing.assetId && !args.force) {
    console.log(`asset already uploaded (id=${existing.assetId}); pass --force to re-upload.`);
    return;
  }

  const token = process.env.CESIUM_ION_TOKEN;
  if (!token) {
    console.log(
      "CESIUM_ION_TOKEN not set — skipping live upload.\n" +
        "To upload, run locally:\n" +
        "  CESIUM_ION_TOKEN=<ion-token> node apps/web/scripts/upload-cesium-ion.mjs\n" +
        "The site will fall back to a placeholder viewer until cesium-ion-asset.json is written.",
    );
    return;
  }

  // 1. Verify tileset exists and looks valid.
  const tilesetJson = path.join(args.tileset, "tileset.json");
  const stat = await fs.stat(tilesetJson).catch(() => null);
  if (!stat) {
    throw new Error(
      `tileset.json not found at ${tilesetJson}.\n` +
        `Generate it first with:\n` +
        `  target/release/catetus optimize --preset geospatial \\\n` +
        `    --output-dir ${args.tileset} benches/scenes/real/bonsai_iter7000.ply`,
    );
  }
  console.log(`packing tileset ${args.tileset}`);
  const tarballPath = path.join(DEFAULT_TMP, "bonsai-tileset.tar.gz");
  const bytes = await tarballTileset(args.tileset, tarballPath);
  console.log(`tarball ${tarballPath} (${bytes} bytes)`);

  // 2. Create asset on ion.
  const createBody = {
    name: args.name,
    description: args.description,
    type: "3DTILES",
    sourceType: "3D_TILES",
    options: {
      sourceType: "3D_TILES",
      position: [args.lon, args.lat, args.height],
    },
  };
  console.log(`POST /v1/assets ${JSON.stringify(createBody)}`);
  const created = await ionRequest(token, "POST", "/v1/assets", createBody);
  const assetId = created.assetMetadata?.id ?? created.id;
  const upload = created.uploadLocation;
  const onComplete = created.onComplete;
  if (!assetId || !upload) {
    throw new Error(`unexpected ion response: ${JSON.stringify(created)}`);
  }
  console.log(`ion asset id=${assetId}`);

  // 3. Upload tarball to the temporary S3 prefix.
  await s3PutTarball(upload, tarballPath);

  // 4. Tell ion the upload is complete so it starts processing.
  if (onComplete) {
    const { method, url, fields } = onComplete;
    console.log(`POST ${url}`);
    await ionRequest(token, method || "POST", url, fields || {});
  } else {
    await ionRequest(token, "POST", `/v1/assets/${assetId}/uploadComplete`, {});
  }

  // 5. Poll until processing finishes.
  const asset = await pollUntilComplete(token, assetId);
  console.log(`asset ready: ${JSON.stringify({ id: asset.id, type: asset.type })}`);

  // 6. Write the asset metadata file the site reads.
  const out = {
    assetId,
    name: args.name,
    description: args.description,
    position: { lon: args.lon, lat: args.lat, height: args.height },
    uploadedAt: new Date().toISOString(),
  };
  await fs.mkdir(path.dirname(ASSET_JSON), { recursive: true });
  await fs.writeFile(ASSET_JSON, JSON.stringify(out, null, 2) + "\n", "utf8");
  console.log(`wrote ${ASSET_JSON}`);
}

main().catch((err) => {
  console.error(err.stack || err.message || String(err));
  process.exit(1);
});
