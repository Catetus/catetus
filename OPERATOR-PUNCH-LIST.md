# Operator Punch List

Living queue of human-credential / external-system actions accumulated by autonomous agents. Engineering work never blocks on these; this list gets executed in batches when convenient.

**Format:** each item has owner-action verb, the agent / branch that produced the artifact, what it unblocks. Strike-through when done.

## 🚨 Tier 0 — Security (do FIRST)

- [ ] **Rotate the leaked Stripe live secret key** — `sk_live_51TXM7f…` was pasted into chat at 2026-05-15. The secret is in the conversation transcript and may persist in Claude logs/telemetry. Stripe Dashboard → Developers → API keys → roll the secret. Takes 5 minutes; eliminates exposure window. Then set the new secret as the Stripe MCP env + Fly secret `STRIPE_SECRET_KEY` for prod. **Never paste secret keys in chat again; use env vars + Fly secrets only.**
- [ ] **Verify Michaelmontalbano account has no orphan SplatForge products** — before the MCP switched to the SplatForge account I accidentally created 3 products in your main "Michaelmontalbano" Stripe account (LIVE mode). I archived one (`prod_UWOzLKCfiYnFkz` "SplatForge Repack Runs"); the other two creates returned IDs (`prod_UWOz97C8pb56YI`, `prod_UWOzCTTZrVGqjo`) that the API can't find for follow-up archive. Likely they didn't persist (transient MCP state mid-swap), but worth a 30-second dashboard check on the Michaelmontalbano account to confirm.

## Stripe SplatForge account state (acct_1TXM7fIZCvZOU40b, LIVE mode)

Created via MCP during this session — all in LIVE mode (`livemode:true`). Operator should verify the prices look right and meter event names match what `apps/api/src/billing.rs` emits:

- **Product `prod_UWP0tXyEhEtjVU`** — SplatForge Repack Runs (usage meter target)
- **Product `prod_UWP0MEwMVKmF9U`** — SplatForge Repack Seconds (usage meter target)
- **Product `prod_UWP01FtK3Czpyl`** — SplatForge Team Seat → **price `price_1TXMD8IZCvZOU40bzQUA2OWX`** ($99/mo USD recurring, live)
- **Meters NOT yet created** — Stripe MCP `stripe_api_search` doesn't surface the meter-events API in this session's tool set. Operator runs `STRIPE_SECRET_KEY=<rotated-new-key> tasks/scripts/stripe-bootstrap.sh --mode live` to create them (`splatforge_repack_runs` + `splatforge_repack_seconds` meter events tied to the products above).
- Then set Fly secret `STRIPE_TEAM_PRICE_ID=price_1TXMD8IZCvZOU40bzQUA2OWX` so `apps/api/src/checkout.rs` mints Team Checkout sessions against the right price.

---

## Tier 1 — Launch-day blockers (do before v0.1.2 publish)

- [x] ~~**Review + merge PR #1**~~ — DONE (commit `c64b107`). Consolidation of 14 branches; 149 tests + 7-page web build. Downstream Bet-0 PRs #5, #7, #8, #9, #10, #12 all merged on top. PR #11 (bet9-revenue) force-pushed clean after rebase-corruption fix; awaiting CI as of 2026-05-15 17:05Z.
- [ ] **`gh issue edit 2580`** — fix `hello@splatforge.dev` placeholder in the Khronos glTF issue body (issue is live, agent submitted via `cbdc83e`, body has a stray placeholder).
- [ ] **Vercel prod deploy + git tag `v0.1.2`** — `apps/web` is publish-ready on `docs/v0.1.2-blog-polish` (commit `1b55210`). 4 pages green in Astro build. Refresh blog after Bet 0 lands (see Tier 3).
- [ ] **OG image generation** — `/og-image.png` shared fallback exists; per-job OG cards need a Vercel Edge function at `/og/report/<id>.png` using `@vercel/og`/Satori. Hook point is `Base.astro`'s `ogImage` field.
- [ ] **Strip-attribution sanity grep** — operator runs `git log --all --grep='Claude' --grep='Anthropic'` before publish to catch any agent slip-up. Should return empty.

## Neural codec (Bet 1) — M3 gate result

- [x] ~~**[Bet 1] Neural codec M3 gate (50× lossless on bonsai+bicycle)**~~ — **MISSED**. Honest plateau ~4-5× neural compression with ΔPSNR > 0; far from 50× target.
  - bonsai @ target 10×: median **3.39× / +0.97 dB** (N=2, 1000 iters)
  - bonsai @ target 20×: median **4.35× / +1.06 dB** (N=3)
  - **Diagnosis**: hyperprior too small (4 lvls × 2-dim × 65k entries → 64-hidden MLP). Min σ ~3-5 levels → 6-8 bits/channel floor → ~380-500 bits/splat hard cap. 50× would need ~80 bits/splat.
  - **Branch**: `splatforge-private/research/neural-codec-v0.1-m3` (commits `8b9bc43` + `d6667dd`)
  - **Modal cost**: ~$1 of $10 cap
  - **Next attempt if revisited**: 10× bigger hyperprior + HEMGS-style spatial context + mixed-bit quantization. Hold for now — v0.2 wins more cheaply.

## Tier 2 — Credentialed actions (any time after Bet 0 lands)

- [ ] **Stripe live-mode bootstrap** — scripts now READY on `feat/stripe-live-mode-prep` (commit `9677611`): `tasks/scripts/stripe-bootstrap.sh --mode {test,live}` (rejects mismatched key prefixes), `tasks/scripts/stripe-bootstrap-team-tier.sh --mode live` ($99/seat/mo USD recurring, idempotent), then `tasks/scripts/stripe-smoke-live.sh` (5 fail-fast contract checks). `apps/api/CHECKOUT.md` "Launch day runbook" section has step-by-step. After: set Fly secrets `STRIPE_SECRET_KEY` (sk_live_), `STRIPE_WEBHOOK_SECRET`, `STRIPE_TEAM_PRICE_ID`.
- [ ] **Cesium ion upload** — `CESIUM_ION_TOKEN=... node apps/web/scripts/upload-cesium-ion.mjs` (built tileset at `apps/web/scripts/tmp/bonsai-tileset`, idempotent). Optional `PUBLIC_CESIUM_ION_VIEWER_TOKEN` on Vercel for live iframe. Branch `feat/cesium-ion-upload`.
- [ ] **Khronos WG email** — paste `docs/standards-outreach/khronos-wg-email.md` body to `standards@khronos.org`. Notifies the glTF WG of issue #2580 + the 23-clause conformance crate.
- [ ] **OpenUSD forum post** — sign in at `https://forum.aousd.org/`, paste title + body from `docs/standards-outreach/openusd-forum-post-READY.md`. Comment the resulting forum URL back on `KhronosGroup/glTF#2580`.

## Tier 3 — Post-Bet-0-merge refreshes

- [ ] **Web viewer attribute-layout migration** — KHR writer aligned to RC on `feat/khr-writer-rc-alignment` (commit `01d465d`); the Astro web viewer at `apps/web/public/viewer/manifest.js` still keys off pre-RC location. Reader auto-detects both, so existing artifacts work; new writes from the web viewer need migration. Pre-RC fixtures at `apps/web/public/compare-scenes/**/scene.gltf` need regen to the new SH layout (per-coef VEC3 FLOAT, not SCALAR-of-45). Tracked as task #71.
- [ ] **DC quantization regression note** — under the RC, DC color is non-negotiably VEC3 FLOAT (ACC_SH_COEF clause), so web-mobile / size-min presets **lose ~9 bytes per splat** vs pre-RC. POSITION + SCALE + OPACITY remain quantized. Cite this in the v0.1.2 blog refresh — it's an honest tradeoff: -9 bytes/splat for RC conformance and KHR ratification credibility.
- [ ] **Add v0.3.1 LPIPS kill to blog** — clean kill discipline result worth surfacing. Single-seed bonsai probe showed +4.281 dB at λ_lpips=0.25; N=3×3 validation revealed it was the high tail of σ=0.77 bonsai variance (true mean +3.337). Outdoor scenes (stump -0.6 dB, bicycle -0.5 dB vs λ=0) trade PSNR for LPIPS-drop honestly but lose the PSNR race. Verdict: no clean v0.3.1 PSNR ship; possible "perceptual-track preset" with both metrics if user wants both axes. Cost: $0.42 of $1.50 Modal budget. Stacks with F-3DGS / CodecGS-Lite / VCR / SaliencyPrune-v1 / Splat-Δ in the kill ledger.
- [ ] **Refresh v0.1.2 blog** — original draft on `docs/v0.1.2-blog-polish` is the floor (16 scenes, 23/10 KHR, 1.38× bonsai real, M-series WebGPU). After PR #5 (cluster.fly, correct numbers) lands, refresh to pick up: stump_real (1.25× outdoor advantage), cluster.fly LOD ladder showing SplatForge LEAD widens on dense indoor (size-min XXL: 59.64× vs splat-transform 15.79× = +278%), KHR-SPZ extension (28/13 clause count, RC-aligned writer on PR #3), 4090 WebGPU bench (1M @ 70 fps / 10M @ 6.5 fps), fidelity-ml v0.4 grouped-R²=-0.10 honesty, on-prem Pro license framework, SaliencyPrune kill citation. The honest pitch: "SplatForge SPZ + size-min wins every measured cell; advantage grows from 1.25× on real outdoor → 3.78× on dense indoor close-up." Real-data lead is HIGHER than the synthetic lead originally claimed (3.78× outdoor → ~3.78× indoor median).
- [ ] **SOG fidelity column on `/bench`** — wire `feat/api-production-hardening`'s fidelity output into the leaderboard once that branch merges.
- [ ] **bicycle_real splat-transform run** — bench did land (commit `cf6e075` on `bench/bicycle-real`, splat-transform 18.48× on 897 MB). Pull row into the leaderboard once branch merges.
- [ ] **Re-run `apps/web/scripts/sync-data.mjs`** — after Bet 0 merges to pick up all the new benches into `apps/web/src/data/`.

## Tier 4 — Hardware-credentialed (4090 box)

- [x] ~~**4090 single-tenant WebGPU re-bench**~~ — DONE on `bench/4090-clean-single-tenant` (commit `34cdc61`). 3 trials, median ± std: **1M @ 70.52 fps ± 0.30 / 10M @ 6.52 fps ± 0.03**. **Contention hypothesis rejected** — single-tenant within trial-to-trial noise of the dual-agent run. The 10M @ 6.5 fps finding is solid; 60 fps @ 10M needs the cs_project + cs_gather fusion unblock.
- [ ] **`schtasks` cleanup** — `schtasks /Delete /TN sf_rebench /F` over SSH on montespc to remove the leftover scheduled task.

## Tier 5 — Recruiting + GTM (v3.2 Bet 4)

- [ ] **LinkedIn recruiter brand activation** — once v0.1.2 publishes, the launch artifact compounds into the recruiter brand. Operator updates LinkedIn job posts + reaches out to the saved-warm-intro list (will exist once Bet 4 task #70 runs).
- [ ] **5-vertical warm-intro outreach** — held until Bet 0 lands per "focus all efforts on Bet 0." Then Bet 4 agent maps the contact graph + drafts first-touch emails for Bentley/Cesium, Apple Vision Pro, Adobe Substance, Autodesk Forma, World Labs / Spark.

---

## Maintenance

- Agents append to this file via direct Edit when they produce operator-actionables. Item count caps at ~50 — older completed items move to `OPERATOR-PUNCH-LIST-DONE.md` archive when this grows past that.
- Tier 1 items block the next public launch artifact. Tier 2-5 are queue-anywhere.
- This list lives in the public repo so it's also a public-facing transparency artifact for the v3.2 plan execution.
