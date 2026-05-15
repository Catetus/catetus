# `splatforge/optimize-action` — Launch-Week Plan

## Goal

Make `splatforge/optimize-action@v1` the default way a Babylon/Three/Cesium
team gates Gaussian-splat PRs. Concretely, in one week we want:

- **300 GitHub stars** on the action repo
- **30 unique repos** running the action (queryable via GitHub Code Search for
  `splatforge/optimize-action`)
- **5 logos** in the README "trusted by" strip
- **3 outbound conversion paths** open: HN, Awesome lists, dev forums

These numbers are deliberately modest — we'd rather hit 30 real installs than
3000 vanity stars.

## Pre-flight (T-2 days)

- [ ] Publish `splatforge/optimize-action` repo (mirror `apps/optimize-action/`).
- [ ] Tag `v1.0.0` + create a `v1` major-version moving tag.
- [ ] Submit listing to the **GitHub Marketplace** (verified-publisher flow).
- [ ] Land a "Using the action" section in the main SplatForge README that
      links here.
- [ ] Land 2 logo cards on `splatforge.dev` (any two of: a museum demo, an
      e-comm shop, an architecture-viz studio that's already in the design
      partner program).
- [ ] Stand up `splatforge.dev/optimize-action` landing page with the same
      `uses:` snippet front and centre.
- [ ] Pre-record a 45-second screen capture: open PR → action runs → badge
      appears. Host as MP4 + GIF.

## Launch day (T+0, Tuesday — best HN day, avoids end-of-week slump)

### Hacker News — Show HN

**Time:** 9:00 AM ET (peaks for technical audiences).

**Title:** `Show HN: SplatForge Optimize – a GitHub Action that compresses 3D splat PRs`

**Body strawman:**

> Hi HN, I've been working on SplatForge — an optimizer that turns raw Gaussian
> splat captures (often 500 MB – 2 GB `.ply`) into web-ready 30-100 MB `.glb`
> for Babylon.js / Three.js / Cesium scenes.
>
> Up to today you had to run the CLI locally and commit the optimized binary.
> The new GitHub Action wraps the hosted optimizer — drop a few lines in a
> workflow file and every PR that touches a `.ply` gets:
>
>   - the optimized `.glb` published behind a public URL
>   - a sticky PR comment with a fidelity badge (PSNR/SSIM coming, byte-savings
>     today)
>   - a configurable regression gate (`regression-threshold: 0.6` = fail if
>     output is bigger than 60% of input)
>
> Free tier is 100 jobs/month per key — covers most OSS repos. The pipeline
> behind it is a Rust API + Modal CPU worker (free) + Modal A100 differentiable
> repack (paid).
>
> Code: github.com/splatforge/optimize-action
> Demo screencap: <link>
> API source + design notes: github.com/splatforge/splatforge
>
> Would love feedback on the workflow ergonomics and what other
> render-engine ecosystems would want this for.

### Awesome list PRs

Open one PR per list, on the same day so the cross-references compound. Use
the same one-liner format each list expects.

- [ ] **awesome-actions** — under "Code Quality" or "Static Analysis"
- [ ] **awesome-github-actions** — under "Build & Test"
- [ ] **awesome-3d-gaussian-splatting** (active community list) — under
      "Tools"
- [ ] **awesome-cesium**, **awesome-threejs**, **awesome-babylonjs** — each
      has a "Tools" or "Integrations" section

PR body template:

> Adds [`splatforge/optimize-action`](https://github.com/splatforge/optimize-action),
> a GitHub Action that compresses Gaussian-splat PRs to web-ready GLB via the
> SplatForge Cloud API and posts a fidelity-badge PR check. Useful for any
> repo that vendor-commits captured 3D scenes.

### Dev-forum cross-posts

Same day, in this order (gives HN a head start):

- [ ] **Babylon.js Forum** — *Showcase* → "Compressing splats in CI"
      Link the example workflow + screencap.
- [ ] **Three.js Discourse** — *Showcase* → same.
- [ ] **Cesium Community Forum** — *Showcase / Real World* → same, with a
      paragraph on Cesium-specific tiling integration.
- [ ] **r/GaussianSplatting** — link post.
- [ ] **r/threejs** — link post.

**Forum-post strawman (shorter than HN):**

> If you're vendor-committing `.ply` / `.splat` files in CI, here's a one-liner
> that compresses them on every PR and gates the build on the result.
>
> Workflow:
> ```yaml
> - uses: splatforge/optimize-action@v1
>   with:
>     api-key: ${{ secrets.SPLATFORGE_API_KEY }}
> ```
>
> Posts a sticky comment with the fidelity badge. Free tier is 100 jobs/month.
> Source + design rationale: github.com/splatforge/optimize-action

### Twitter / X

A single thread, posted within 30 min of the HN submission going up. Keep
each post under 240 chars so they don't truncate in embeds.

> 1/ Just shipped: a GitHub Action that takes the raw `.ply` you committed to
> your three.js / babylon.js / cesium repo, runs it through the SplatForge
> optimizer, and posts a fidelity badge on the PR. Free tier covers most OSS
> repos.
>
> 2/ One liner: `uses: splatforge/optimize-action@v1`
> See it live: [link to a public demo PR]
>
> 3/ Behind the scenes: deterministic Rust optimizer (CPU, free) + gsplat A100
> differentiable repack (paid). Same hosted API, same PR badge — you don't
> pick a tier, the platform does.

## Day +1 to +3

- Cross-link from the splatforge.dev homepage to top HN comment if it lands
  on the front page.
- Reply to every HN comment within 30 min — the algorithm rewards engagement
  in the first 4 hours.
- DM 5 design-partner contacts and ask them to install the action on a
  public PR before close of day +1, so the GH Code Search count rises early.

## Day +4 to +7

- Land a deeper write-up on the SplatForge blog comparing PSNR/SSIM scores
  on the SplatBench v0 corpus before vs after the optimizer — this is the
  "social proof" piece for arms-length skeptics.
- Submit to **dev.to** and **hashnode** as a follow-up.
- If HN landed: write a "what we learned" retro post.

## Anti-patterns to avoid

- **No "Generated with Claude" / "Made with AI" framing** in any of the
  launch copy. The signal is performance and ergonomics, not novelty.
- **No paid-tier upsell in the launch.** The free tier is the entire pitch.
  Paid lands two weeks after when adoption is real.
- **No "join the waitlist" friction** anywhere on the path from HN to
  a working PR. The action must work on a fresh repo with a freshly-issued
  free key in <60 seconds.

## Metrics to track

| Metric | Source | Target (day 7) |
| --- | --- | --- |
| GitHub stars on action repo | GH API | 300 |
| Unique repos using action | `gh search code "uses: splatforge/optimize-action"` | 30 |
| HN points (peak) | news.ycombinator.com | 200 |
| Free-tier signup-to-first-PR conversion | API analytics | ≥40% |
| API jobs created via Action | API analytics (label prefix `gh:`) | 200 |
