# `@catetus/website`

Static landing page for **catetus.com**. Vite + TypeScript, no framework,
no SSR. Output is a plain `dist/` directory deployable to any CDN.

> **Stealth.** The repository is currently private and the project is in
> stealth mode. Do **not** deploy this site to a public origin without
> explicit approval. There is no `vercel.json`, no GitHub Pages action, no
> `wrangler.toml` in this package. Adding one is a separate decision.

---

## What's in scope

- `/` — landing page with hero, headline +6.54 dB SOG-sidecar result,
  canonical-11 11/11 leaderboard summary, link to the V5.2 defensive
  publication, viewer CTA.
- `/viewer` — embeds `packages/viewer-app/` via `<iframe>` from the same
  origin (default `/viewer/`). Override with `window.__CATETUS_VIEWER_URL__`.
- `/about` — methodology + prior-art acknowledgments (gsplat, Inria 3DGS,
  PlayCanvas SOG, GoDe, MPEG G-PCC). No team info.
- Hash-based router (`#/viewer`, `#/about`). Host-agnostic — works on any
  static CDN without rewrites.
- Dark mode only.

## What's deferred (stub or omitted)

- **Public hosting / deploy config** — no `vercel.json`, no GitHub Pages
  workflow, no Cloudflare Pages config. Add when stealth comes off.
- **Real test scene** on `/viewer` — currently iframes the unpopulated
  viewer-app. A small bonsai PLY served from a CDN should be wired before
  public launch (see "Pre-populating the viewer" below).
- **Analytics** — none. No Plausible, no GA, no Vercel Analytics. Add a
  privacy-respecting option (Plausible self-host or Cloudflare Web Analytics)
  if/when needed.
- **Contact form / mailing list** — none. Stealth.
- **`robots.txt` + sitemap** — `index.html` ships with `noindex, nofollow`
  meta. Flip that meta and add `robots.txt` + `sitemap.xml` at deploy time.
- **Open Graph / Twitter card images** — none. Add before any public share.
- **PDF artifact** of the defensive publication — currently links to the
  Markdown source on GitHub. Render to PDF before public link-out.

---

## Develop

```sh
pnpm install         # from repo root
pnpm -C packages/website dev      # http://localhost:5174
pnpm -C packages/website build    # → packages/website/dist/
pnpm -C packages/website preview  # serves dist/
pnpm -C packages/website lint     # tsc --noEmit
```

## Co-hosting the viewer

The viewer-app is a separate Vite project. To make `/viewer` actually load
the viewer in the iframe, build both packages and stitch them at the CDN:

```sh
pnpm -C packages/viewer-app build         # → packages/viewer-app/dist/
pnpm -C packages/website build            # → packages/website/dist/

# Then on your static host, mount:
#   packages/website/dist/      at /
#   packages/viewer-app/dist/   at /viewer/
```

Alternatively, copy the viewer build into the website's `dist/viewer/`
during your deploy step:

```sh
rsync -a packages/viewer-app/dist/ packages/website/dist/viewer/
```

To point the iframe at a totally separate origin (e.g., a sandbox
subdomain), inject a global at build time:

```ts
// vite.config.ts
define: { 'window.__CATETUS_VIEWER_URL__': JSON.stringify('https://viewer.catetus.com/') }
```

## Pre-populating the viewer with a test scene

The viewer-app accepts a `?src=` URL parameter (see `packages/viewer-app/src/`
for the loader entry point). Once a public CDN-hosted bonsai PLY exists,
update `src/pages/viewer.ts` to pass `?src=<url>` on the iframe `src`.
Not wired yet — the public sample asset hasn't been uploaded.

---

## Deploy (when stealth comes off)

The output is a plain SPA. Any of these work — pick one, don't pick all:

- **Cloudflare Pages** — point at `packages/website` with build command
  `pnpm -C packages/website build` and output dir `packages/website/dist`.
  Add a `/*  /index.html  200` rewrite if you migrate off hash routing.
- **Vercel** — same. Add `vercel.json` with a SPA rewrite if needed.
- **GitHub Pages** — gh-actions workflow that runs the build and uploads
  `dist/`. Hash routing means no `404.html` SPA hack required.
- **S3 + CloudFront** — copy `dist/` to a bucket, point a distribution at
  it, set the default root object to `index.html`.

Before any public push:

1. Flip `<meta name="robots">` in `index.html` from `noindex` to allow.
2. Add `public/robots.txt` and `public/sitemap.xml`.
3. Add an OG image and `<meta property="og:*" />` tags.
4. Render `experiments/defensive-publication/V5_2_PUBLIC.md` to a PDF and
   host it; update the link in `src/pages/home.ts`.
5. Co-host the viewer (above).
6. Audit `src/pages/` for anything that shouldn't be public yet.

---

## License

Apache-2.0 (matches the rest of the repo).
