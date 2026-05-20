# `catetus.com` — Deploy Runbook

> **Status: stealth.** Nothing in this document gets executed until the user
> explicitly says "ship it". The repo is private. There is no public origin.
> Treat every section below as a recipe, not a button.

This runbook makes the website + viewer-app deployable to either **Vercel**
or **Cloudflare Pages** with the same artifact (`packages/website/dist/`)
produced by `packages/website/build-composite.sh`. Pick one, not both.

---

## 0. Pre-deploy checklist

Run through this **in order**. Do not skip — a public `catetus.com` is a
one-way door.

- [ ] **Stealth lifted.** Founder has explicitly approved going public.
      Today's date is recorded in `tasks/launch-log.md`.
- [ ] **Repo visibility.** Decide if the GitHub repo is going public at the
      same time, or staying private behind a Releases-only surface.
- [ ] **Demo scene quality gate.** `packages/website/public/demos/splatbench_lowlight.glb`
      is currently 305 KB — that is a **placeholder-grade** asset, not a real
      bonsai/garden/etc. Replace with a real public scene (≤ 50 MB ideally;
      Cloudflare Pages caps individual files at 25 MB, so chunk or host on
      R2/S3 for anything larger) before launch.
- [ ] **Robots flip.** `packages/website/public/robots.txt` is currently
      `Disallow: /`. Replace per "Robots strategy" below.
- [ ] **Meta robots flip.** `packages/website/index.html` ships with
      `<meta name="robots" content="noindex, nofollow" />`. Remove or change
      to `index, follow` before launch.
- [ ] **OG image + Twitter card.** Add `public/og.png` (1200×630) and the
      corresponding `<meta property="og:*" />` + `<meta name="twitter:*" />`
      tags to `index.html`. No public share until this exists.
- [ ] **Defensive publication PDF.** Render
      `experiments/defensive-publication/V5_2_PUBLIC.md` to PDF, place it at
      `packages/website/public/V5_2_PUBLIC.pdf`, and update the link in
      `src/pages/home.ts`.
- [ ] **Page audit.** Read `src/pages/{home,viewer,about,notFound}.ts` end
      to end. Nothing about team members, funding, customers, or unreleased
      benchmarks ships in v1.
- [ ] **Lint + build clean.** `pnpm -C packages/website lint` and
      `bash packages/website/build-composite.sh` both exit 0.
- [ ] **CDN budget.** Read the "Cost estimates" section. Confirm card on
      file with chosen provider. Set a billing alert at $25/mo.
- [ ] **Domain.** `__DOMAIN__` (e.g. `catetus.com`) is registered and
      DNS is delegated to the chosen platform's nameservers (Cloudflare
      Registrar → Pages is the path of least resistance).
- [ ] **Rollback plan rehearsed.** See "Rollback" below. You have screen-
      shotted the dashboard "Promote previous deployment" button so you can
      find it under stress.

---

## 1. Build locally (always do this first)

```sh
# from repo root
bash packages/website/build-composite.sh
```

Expected output (last 5 lines):

```
==> verifying composite layout
  OK  packages/website/dist/index.html
  OK  packages/website/dist/viewer/index.html
  OK  packages/website/dist/demos/splatbench_lowlight.glb
==> composite build OK
```

Smoke-test the artifact before any push:

```sh
pnpm -C packages/website preview   # http://localhost:5174
# In a browser: open /, click "Open viewer", verify the iframe loads
# /viewer/ and the demo scene at /demos/splatbench_lowlight.glb renders.
```

If `preview` doesn't serve `/viewer/` (it's a different Vite project), use
a generic static server instead:

```sh
npx --yes serve packages/website/dist -p 5174
```

---

## 2. Option A — Cloudflare Pages (recommended)

**Why recommended.** Cheaper for static + WebGL traffic, no Functions cost
since we have none, and the free tier (500 builds / 100 GB egress per month
on the free plan; unlimited bandwidth on Pro for $20/mo) covers any pre-PMF
launch volume. Sits naturally next to Cloudflare Registrar + R2 for large
demo scenes later.

### One-time setup

1. **Registrar.** Move `__DOMAIN__` to Cloudflare Registrar (at-cost) or
   delegate NS records to Cloudflare DNS. Verify in dashboard.
2. **Create Pages project** in the Cloudflare dashboard:
   - Connect GitHub → select this repo.
   - **Production branch:** `main`.
   - **Framework preset:** None.
   - **Build command:** `bash packages/website/build-composite.sh`
   - **Build output directory:** `packages/website/dist`
   - **Root directory:** *(leave blank — composite script handles paths)*
   - **Environment variables:** none required. (`CF_PAGES=1` is set by
     the platform — the composite script already keys off it.)
   - **Node version:** 20 (set via `NODE_VERSION=20` env var, or commit
     `.nvmrc` at repo root).
3. **First deploy** runs automatically. Inspect the build log; confirm the
   "composite build OK" line is present and total dist size matches local.
4. **Custom domain.** Pages → project → Custom domains → "Set up a custom
   domain" → enter `__DOMAIN__` → Cloudflare creates the DNS records and
   provisions the TLS cert (LetsEncrypt or Google Trust Services). Allow
   ~2 min for propagation.
5. **Verify headers.** `curl -I https://__DOMAIN__/assets/index-*.js` should
   show `Cache-Control: public, max-age=31536000, immutable`. If not, the
   `_headers` file isn't in `dist/` — re-run the composite locally and check.

### Per-deploy

Pushing to `main` triggers a build. Preview deploys are created for every
PR. To deploy a non-`main` commit manually:

```sh
# wrangler is installed via the GitHub Action runner or `pnpm dlx wrangler`.
# This command is DOCUMENTED but NOT TO BE RUN in stealth.
#   pnpm dlx wrangler pages deploy packages/website/dist \
#     --project-name=catetus-website --branch=main
```

---

## 3. Option B — Vercel

**Why pick this instead.** Strong DX, instant preview deploys, good
analytics if you ever turn them on. Slightly more expensive at scale because
bandwidth is metered on the Pro plan.

### One-time setup

1. **Create project** in Vercel dashboard → Import → select this repo.
   - **Framework Preset:** Other.
   - **Root Directory:** *(leave as repo root — `vercel.json` lives in
     `packages/website/` but Vercel reads it from any path provided to
     `vercel link`; for a monorepo, set Root Directory to
     `packages/website` so Vercel auto-detects `vercel.json`).*
   - **Build / Output settings:** leave blank — `vercel.json` overrides.
   - **Install Command:** `pnpm install --frozen-lockfile` (already in
     `vercel.json`).
2. **Environment variables:** none required.
3. **First deploy.** Push to `main` or click "Deploy" in the dashboard.
4. **Custom domain.** Project → Settings → Domains → add `__DOMAIN__` →
   Vercel issues TLS via LetsEncrypt and supplies the A/CNAME values for
   your DNS provider.

### Per-deploy

Pushing to `main` deploys to production. PRs get preview URLs at
`catetus-website-<hash>.vercel.app`. Manual deploys (NOT in stealth):

```sh
#   pnpm dlx vercel --prod      # interactive, prompts for project link
```

---

## 4. Cost estimates

Rough monthly cost at three traffic tiers. Assumes ~5 MB average page weight
once the real demo scene is wired (1 MB landing + 4 MB viewer + demo loaded
on demand). Numbers in USD, **before** any volume discount.

| Visitors/mo | Pageviews | Egress | Cloudflare Pages | Vercel |
|---|---|---|---|---|
| 1 K       | ~3 K   | ~15 GB  | **$0** (free)             | **$0** (Hobby) |
| 10 K      | ~30 K  | ~150 GB | **$0** (free) or **$20** Pro | **$20** Pro + ~$0 BW |
| 100 K     | ~300 K | ~1.5 TB | **$20** Pro (unlimited BW) | **$20** Pro + ~$60 BW = **~$80** |

Caveats:

- These numbers ignore the demo scene itself if hosted on R2/S3 separately.
  A real bonsai GLB at ~30 MB pulled by half of visitors at 100 K traffic =
  ~1.5 TB just for demos. Host large assets on **R2** (free egress) regardless
  of which platform serves HTML/JS.
- Vercel adds Function execution costs if any route becomes dynamic. We have
  none today — keep it that way.
- Build minutes: Cloudflare gives 500 builds/mo free; Vercel gives 6000 build
  min/mo on Hobby, 24000 on Pro. Composite build is ~10 s end-to-end, so this
  is never the binding constraint.

**Verdict:** Cloudflare Pages until traffic > 1M/mo or you need Vercel-
exclusive features (Edge Functions, ISR, etc., which we don't).

---

## 5. Custom domain (`__DOMAIN__`) — first hookup

Same steps on both platforms in 2026:

1. Add the domain in the platform dashboard.
2. Platform shows DNS records to create (usually one CNAME + one TXT for
   verification, or one A + one AAAA for apex domains).
3. Add records at your DNS provider. If using Cloudflare DNS for both
   registrar and DNS, this is one click.
4. Wait for verification (typically <2 min, occasionally 10–30 min for
   apex domains depending on propagation).
5. Force HTTPS (default on both platforms).
6. Test: `curl -I https://__DOMAIN__/` → 200 + cache headers per §2.
7. Test viewer route: `curl -I https://__DOMAIN__/viewer/` → 200, content
   should be the viewer-app's `index.html`.
8. Test SPA fallback: `curl -I https://__DOMAIN__/viewer/some/deep/path` →
   200, same content as `/viewer/index.html` (proves the rewrite rule).

---

## 6. Rollback

A bad deploy is recoverable in **under 60 seconds** on either platform.

### Cloudflare Pages

1. Dashboard → project → Deployments tab.
2. Find the last good deployment (green check).
3. Click "⋯" → **Rollback to this deployment** → confirm.
4. New production URL serves the previous build within ~10 s.

### Vercel

1. Dashboard → project → Deployments tab.
2. Find the last good deployment.
3. Click "⋯" → **Promote to Production** → confirm.
4. Aliases swap atomically; live within ~5 s.

### Backstop (both platforms)

If the dashboard itself is unreachable: `git revert <bad-commit> && git push`
re-triggers a build of the previous tree. ~2 min worst case. This is why
every release must be a single squashed commit on `main` — never a merge
commit with 30 sub-commits.

---

## 7. SharedArrayBuffer / COOP+COEP (future)

The viewer-app's `sort-worker` currently uses **Transferable** ArrayBuffers,
not SharedArrayBuffer. Therefore we do **not** need
`Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp` headers today.

If SAB becomes required (e.g., zero-copy sort across N workers, WASM
threading, multi-core radix):

1. Audit every cross-origin asset for `Cross-Origin-Resource-Policy:
   cross-origin`. This includes demo scenes on R2, fonts, analytics scripts.
   Anything missing CORP will be **blocked** under COEP.
2. Uncomment the COOP/COEP block in `public/_headers` (CF Pages).
3. Add the equivalent to `vercel.json → headers` for Vercel.
4. Test in incognito + verify `crossOriginIsolated === true` in DevTools
   console before deploying.
5. Expect collateral damage: third-party embeds, OAuth popups, and some
   analytics will break. Plan accordingly.

---

## 8. Robots strategy

| Phase | `robots.txt` | `<meta name="robots">` |
|---|---|---|
| Stealth (now) | `Disallow: /` | `noindex, nofollow` |
| Soft launch (private beta link sharing) | `Disallow: /` | `noindex, nofollow` (only people with the URL find it) |
| Public launch | `Allow: /` + `Sitemap: https://__DOMAIN__/sitemap.xml` | remove the meta entirely (or `index, follow`) |
| Post-launch (block AI scrapers selectively) | add per-bot `Disallow` blocks for GPTBot / CCBot / etc. — see https://darkvisitors.com for the current list | unchanged |

Sitemap generation is deferred until launch. When needed, drop a static
`public/sitemap.xml` listing `/`, `/#/about`, `/#/viewer` (hash routes are
NOT crawlable — switch to history routing first if SEO matters for those).

---

## 9. What this runbook does NOT cover

- Email (`hello@__DOMAIN__`, etc.) — set up via Cloudflare Email Routing
  or Google Workspace; orthogonal to Pages/Vercel.
- Status page — Cloudflare has built-in. Add when SLA matters.
- Error tracking (Sentry) — not wired. Add post-launch if signups start.
- Analytics — see top-level project decisions, not deploy plumbing.
- The viewer-app's own development. Refer to `packages/viewer-app/README.md`.
- Public asset hosting for large demo scenes — use R2 (Cloudflare) or
  S3+CloudFront; do not commit GBs of GLBs to git.
