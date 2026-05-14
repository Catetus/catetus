# @splatforge/web

Public landing page for SplatForge. Static site built with [Astro](https://astro.build/) — zero client JS by default plus one tiny inline island for the preset toggle.

## Develop

```bash
# from repo root
pnpm install
pnpm -F @splatforge/web dev
```

## Build

```bash
pnpm -F @splatforge/web build
# output → apps/web/dist
```

The build step reads `benches/reports/splatbench-v0.json` at compile time via a typed loader (`src/lib/splatbench.ts`). The leaderboard fidelity column auto-activates as soon as that JSON gains a `fidelity: { deltaE94, ssim, psnr }` field per scene (planned for v0.1.1).

## Deploy

- **Vercel** — `Framework Preset: Astro`, no env vars required, root: `apps/web`.
- **Cloudflare Pages** — build cmd `pnpm -F @splatforge/web build`, output `apps/web/dist`.
- **GitHub Pages** — push `apps/web/dist` to `gh-pages`.

No upload backend exists yet; the "Try it" section is a placeholder for the design-partner pipeline.
