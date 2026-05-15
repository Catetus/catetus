# Vercel Functions for the SplatForge web app

These functions handle the few server-side tasks the static Astro site
can't do in the browser:

| function          | path                  | role                                       |
|-------------------|----------------------|--------------------------------------------|
| `upload-url.ts`   | `POST /api/upload-url` | Signs a Cloudflare R2 PUT URL so the browser uploads splat scenes directly to object storage (no Vercel-proxy body limits, no mixed-content blocking). |

Static-Astro deployments on Vercel auto-detect `api/*.ts` siblings of
the project root as Node.js serverless functions. Default runtime is
Node 24 on Fluid Compute.

## Required environment variables

Set these in **Project Settings → Environment Variables** on Vercel
(all three "Environments" boxes ticked: Production / Preview / Dev):

| name                 | description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `R2_ACCOUNT_ID`      | Cloudflare account ID — see the URL fragment in the R2 dashboard            |
| `R2_BUCKET`          | Bucket name, e.g. `splatforge-uploads`                                       |
| `R2_ACCESS_KEY_ID`   | S3-compatible API token, scoped to **Object Read & Write** on this bucket    |
| `R2_SECRET_ACCESS_KEY` | The secret half of the API token                                          |

The function authenticates to R2 via SigV4 (S3-compatible). No
Cloudflare Wrangler runtime, no extra binding setup needed.

## One-time R2 bucket configuration

In the Cloudflare dashboard:

1. **Create the bucket.** R2 → Create bucket → pick the same name you
   wrote into `R2_BUCKET`. Location: Automatic.
2. **CORS policy.** Bucket → Settings → CORS. Allow `PUT` + `GET` from
   the web origins:
   ```json
   [{
     "AllowedOrigins": [
       "https://splatforge.dev",
       "http://localhost:4321"
     ],
     "AllowedMethods": ["PUT", "GET"],
     "AllowedHeaders": ["*"],
     "ExposeHeaders": ["ETag"],
     "MaxAgeSeconds": 3600
   }]
   ```
3. **Lifecycle.** Bucket → Settings → Object lifecycle. Add a rule:
   delete objects under prefix `uploads/` older than `1 day`. Stale
   unconsumed uploads disappear automatically.
4. **API token.** Right rail → "Manage R2 API Tokens" → Create token →
   permissions: *Object Read & Write*, *Apply to specific buckets*:
   `splatforge-uploads`. Copy the access key + secret into the Vercel
   env vars above.

## Local development

`PUT` from a localhost dev server works against the same R2 bucket if
you whitelist `http://localhost:4321` in CORS (item 2 above). The
function reads R2 env vars from `apps/web/.env.local` when running
`astro dev` — copy `.env.example` (when added) or set them in your
shell.

## Why R2 and not Vercel Blob

R2 has $0 egress fees and is S3-compatible; the same Vercel Function +
client code works against AWS S3 or any S3-compatible store with a
config change. Vercel Blob would have been simpler but caps single
objects at 5 GB and bills egress, both of which matter at the scale
SplatForge targets.
