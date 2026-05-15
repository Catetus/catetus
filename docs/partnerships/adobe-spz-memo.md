# SplatForge × Adobe: Production-grade optimization for the Photoshop / Substance 3D Gaussian-Splat pipeline

**Status:** Confidential — BD/partnership draft. Not for public distribution.
**From:** SplatForge (Monte Abbananto, founder/operator)
**To:** Adobe Spatial-3D / Substance 3D leadership, via the Babylon.js maintainer team (Adobe-funded) or Khronos 3D Formats WG (mutual membership).
**Date:** May 2026
**One-line ask:** Ship `splatforge optimize` as the engine behind Photoshop and Substance 3D's "Export 3D Gaussian Splat → optimize for web" path, and license `splatforge-pro` for an "Optimize Pro" tier inside Creative Cloud.

---

## 1. Why now

Three things became true in the last 90 days:

1. **SPZ is winning the wire-format war.** Niantic released SPZ 4 on 2026-05-11 with a vendor-extension system, parallel ZSTD streams, and removed the 10M-point cap. The first registered vendor extension is Adobe's (`recommended camera bounds for orbit-style viewers`). SPZ is the format SplatForge has shipped as a first-class output since v0.1.0.
2. **Adobe is publicly all-in on splats.** Photoshop's "Rotate Object" feature shipped on SPZ, and Adobe users have generated ~800K SPZ files in the trailing two months. Adobe-funded Babylon.js 8.0 (March 2025) shipped first-party SPZ + compressed-PLY readers, spherical-harmonic support, and static-texture optimizations that took complex scenes from ~15 FPS to 60 FPS. The viewer story is solved.
3. **The optimizer side is not.** Adobe ships great capture and authoring; the bytes-on-the-wire problem — fidelity-graded compression, deterministic outputs, browser-ready glTF/SPZ pairing, CI-grade regression gates — is currently a per-customer DIY job. PlayCanvas's `splat-transform` is the open-source default. Our 13-scene SplatBench corpus says it gives up 3.78× of compression vs the bar.

SplatForge is the only team that has built the production layer of this stack as a first-class product: a deterministic Rust pipeline, a benchmark corpus with ΔE94/SSIM/ML-Score gates, a hosted API on Fly, a KHR_gaussian_splatting conformance suite that has been submitted upstream to Khronos, and a research line (PostHAC + Differentiable Repack + SwVQ) that stacks on top of SPZ instead of replacing it.

We want to put that pipeline behind Adobe's splat-export button.

## 2. The two shapes of the partnership

We're proposing the partnership in two layers — one that helps every Photoshop user immediately, one that becomes a Creative Cloud SKU.

### Shape A — `splatforge optimize` as the engine behind Photoshop's "Optimize for web" toggle

When a Photoshop or Substance 3D user clicks "Export 3D Gaussian Splat" with **Optimize for web** enabled, the export goes through our deterministic pipeline:

- Output format selectable: SPZ 4 (Adobe's native), `.glb + KHR_gaussian_splatting`, or both.
- Median compression on the public corpus: **21.88× (web-mobile preset) vs splat-transform's 5.91×** on the same scenes.
- BLAKE3-stable bytes — same input + same SplatForge version produces identical output, which means Adobe can cache by content hash inside CC.
- Fidelity report (ΔE94, SSIM, ML-Score across 8 deterministic orbit poses) emitted alongside the asset. The export dialog can show "quality: 99.4 / 100" instead of a binary "Done."
- Zero new CC infra: this runs as either an in-process Rust library linked into Photoshop's spatial-3D module, or as a hosted call to `splatforge-api.fly.dev` for users whose Photoshop is online. Both shapes ship from the same codebase.

This is the **free / default** tier. The user doesn't see SplatForge; they see "Optimize for web," it works, and their files are 4× smaller than what splat-transform would produce. Adobe's brand on the surface; our engine in the basement.

### Shape B — `splatforge-pro` licensed as a Creative Cloud "Optimize Pro" tier

The same export dialog gets a second toggle: **Optimize Pro** (or whatever Adobe wants to call the SKU). That route runs the private compression engine:

- **PostHAC** — splat-aware hierarchical attention entropy coding. Stacks on SPZ for an extra 1.57× on the public corpus and **34.5× net on 2.8M-splat bonsai-class scenes**. Architecturally distinct from SwVQ — the two are incompatible by design.
- **Differentiable Repack** — RDO-optimized compression. **+6.4 dB PSNR over opacity-prune at 50% byte budget on bonsai**, at $0.05–$0.12 of GPU per scene. The cost target is set deliberately so a $9.99/month CC add-on is comfortably above unit-cost.
- **Point-Cloud Codec Adapter** — Draco-based final-codec stage, **1.80× orthogonal at 0 dB ΔPSNR** on bonsai. Composes with PostHAC.
- **Saliency-weighted vector quantization (SwVQ)** — 37% lower swMSE vs uniform 8-bit, already public-readable via the SPZ flag-bit-0 extension we've reserved.

Optimize Pro is the version that ships large, museum-grade, or AR-target scenes — the use cases that justify a per-scene fee. The license to Adobe is for the encoder; the SPZ output stays readable in any standard SPZ 4 reader, including the open-source one.

## 3. The technical fit (why we're the right partner, not just *a* partner)

| Layer | Adobe today | What SplatForge adds |
|---|---|---|
| Capture | Scaniverse / Rotate Object / Substance 3D scan tools | — (not our lane) |
| Authoring | Photoshop spatial-3D, Substance 3D Painter | — |
| Wire format | SPZ 4 (Niantic), with Adobe vendor extension | First-class SPZ writer in SplatForge core; PostHAC stacks on SPZ rather than replacing it |
| Web runtime | Babylon.js 8.0 (Adobe-funded SPZ reader) | KHR_gaussian_splatting glTF output for non-Babylon runtimes; SOG read-back for PlayCanvas interop |
| Optimization / encoding | Currently DIY per customer | **SplatForge: 21.88× median compression, deterministic, fidelity-gated, hosted-or-embedded** |
| Standards | Adobe seat on Khronos 3D Formats WG (Stefano Corazza et al.) | KHR_gaussian_splatting conformance suite drafted by SplatForge — 23 clauses, 10 golden fixtures, CI validator binary; submission pending Khronos acceptance |

SPZ 4's vendor-extension byte (which Adobe now owns for camera bounds) and the SPZ header `flags` byte (bit 0 reserved for our SwVQ chunk; bits 1–7 unallocated) sit next to each other in the same 16-byte header. There is a clean, conflict-free path to adding a SplatForge optimizer-marker extension and an Adobe-specific extension in the same file without either side stepping on the other. We'd like to standardize that registry with Adobe rather than fork it.

## 4. The numbers (cite-ready, all from the public SplatBench corpus)

- **Compression — SplatForge vs the open-source bar:** 21.88× median (web-mobile preset) vs splat-transform v2.1.1's 5.91× median on the same 13 SplatBench scenes — a **3.78× advantage** measured on identical inputs.
- **Compression — stacked private path:** PostHAC on top of SPZ delivers **34.5× on the 2.8M-splat bonsai scene** — net-residual improves with scene size, which is the regime Photoshop / Substance 3D will increasingly produce.
- **Fidelity:** Differentiable Repack delivers **+6.4 dB PSNR over opacity-prune at 50% byte budget** on bonsai, at $0.05–$0.12 of GPU cost per scene.
- **Determinism:** BLAKE3-stable canonical IR; same input + same version = identical output bytes. Required for cache-by-hash inside CC and for Adobe's audit/provenance requirements.
- **Runtime:** WebGPU compute decode at **127 fps @ 1M splats** on M-series Macs; streaming-tile viewer with 1.3 ms cold-start and 512 MB resident LRU — drop-in for any CC web-preview surface that doesn't already use Babylon.
- **Standards:** SplatForge's `crates/splatforge-khr-conformance` ships 23 clauses + 10 golden fixtures + a validator binary, intended for upstream Khronos acceptance as the conformance suite for KHR_gaussian_splatting.

## 5. The strategic frame

There are two axes forming in the splat-optimization market:

- **The free-CLI axis:** PlayCanvas/Snap shipping `splat-transform` and SOG. Excellent for one-off conversions on a developer's laptop. Structurally unable to ship hosted infrastructure, CI integrations, fidelity gates, or enterprise SLAs — those are commercial offerings PlayCanvas is not in the business of selling.
- **The production-grade axis:** Adobe Creative Cloud (creator-facing, billion-dollar distribution, but no internal optimizer team) + SplatForge (deterministic Rust pipeline, hosted infra, perceptual fidelity QA, research compression engine).

We're not asking Adobe to take a side against PlayCanvas. PlayCanvas doesn't ship hosted, doesn't ship fidelity gates, and doesn't ship inside Creative Cloud — they are a complementary, not competitive, asset for Adobe's runtime story. We're asking Adobe to fill the production-grade hole with the partner that is already 3.78× ahead of the open-source default and is the only team shipping the fidelity layer.

## 6. What we want from Adobe

1. **An intro** to the Spatial-3D / Substance 3D leadership and the engineer(s) who owned the Photoshop SPZ-export path. (Sequence proposal: Babylon.js maintainer team → Adobe-internal intro → working session. Backup path: Khronos 3D Formats WG co-membership via Stefano Corazza.)
2. **A 90-minute working-session demo** of `splatforge optimize` running against an Adobe-supplied corpus (or our SplatBench corpus, if Adobe prefers to evaluate against public scenes first). PostHAC + DiffRepack live. Fidelity report rendered side-by-side with the source PLY.
3. **An evaluation license trial** — 90 days, scoped to Adobe Spatial-3D / Substance 3D internal use. Free. No exclusivity strings. The goal of the trial is to put numbers on Adobe's own scenes so the decision to license is not theoretical.

## 7. What we'd give Adobe

1. **A private engineering channel for SPZ-format extensions.** Bit 0 of the SPZ flags byte is already reserved for our SwVQ extension; bits 1–7 are open. We'd collaborate on a future flag for any Adobe-specific extension (an "optimizer-marker" byte is the obvious one — letting any SPZ reader identify the encoder lineage of a file is a thing both organizations want).
2. **Early access to all pre-public research algorithms** (PostHAC, DifferentiableRepack, SwVQ, the fidelity-ML scoring model, the Point-Cloud Codec Adapter) under NDA. Adobe sees our roadmap before Khronos does.
3. **A co-authored SIGGRAPH 2027 submission** if the joint work produces a publishable result — *"Production-grade Gaussian Splat encoding for the Adobe Creative Cloud pipeline"* writes itself.
4. **Conformance leadership in Khronos** — Adobe is one of the largest voting members of the WG; we are the team that wrote the conformance suite. A coordinated submission lands faster than two parallel ones.

## 8. What we are deliberately NOT asking

- **No brand placement** on the Photoshop or Substance 3D UI. The "Powered by SplatForge" footer is the wrong battle. The right battle is: Adobe users get a 4× better default, attribution lives in `chunk.metadata.encoder = "splatforge/$version"` inside the SPZ file.
- **No exclusivity.** Adobe values neutrality across the ecosystem. We are not asking Adobe to choose us over PlayCanvas, or to refuse to ship `splat-transform` interoperability. We are asking Adobe to add SplatForge where there is no production-grade choice today.
- **No request for investment.** This memo is about distribution, not capital.

## 9. The honest scope statement

SplatForge today is a six-person team. Adobe is a 30,000-person company. Below is what we can credibly deliver in the first six months of partnership, and what we can't:

**Can deliver in 6 months (with a yes from Adobe on the embedding shape):**
- Production-grade Rust static library or hosted endpoint, exposing the public `splatforge optimize` surface, integrated into a Photoshop or Substance 3D prerelease build. (We embed; Adobe ships.)
- SLA on the hosted API (99.5% in the first quarter, 99.9% target within 12 months). Today we're at 99.5% with a single Fly region; geographic expansion is a known, costed line item.
- Conformance-suite handoff to the joint Adobe + SplatForge Khronos working track.
- Determinism guarantees and source-input audit logging suitable for Adobe's compliance review.

**Cannot deliver in 6 months:**
- Adobe-grade enterprise security review certification (SOC 2 Type II, ISO 27001) without a phased plan. We are *targeting* SOC 2 Type I in Q4 2026; Type II in Q2 2027. This will be a gating item for production CC integration and we are being explicit about it now rather than at the contract stage.
- Localization to all 30+ Creative Cloud locales — error messages today are English-only.
- 100-region GPU capacity. Differentiable Repack today runs on a single 4090; we have a scaling plan with three credible cloud-GPU providers, but it is not lit up.

We'd rather under-promise in writing than discover the gap at the integration meeting.

## 10. The decision

We are asking Adobe for a 90-minute working session in the next eight weeks. If the demo lands and the numbers replicate on Adobe's own scenes, the next step is an evaluation-license trial under a mutual NDA, followed by a Term Sheet for the embedding shape (A, B, or both). If the demo doesn't land — if Adobe's internal team has measurements that contradict ours, or strategic reasons to keep optimization in-house — we want to know that early too, so we can re-cast our relationship as "Adobe is an upstream library consumer of `splatforge-spz` and KHR_gaussian_splatting conformance" and move on without consuming each other's time.

The full outreach sequence and the contact map are in `outreach-sequence.md` and `contact-map.md` in this same folder.

---

*Prepared by SplatForge for confidential Adobe BD use. All numbers cited are from the public SplatBench corpus, committed at `benches/reports/splatbench-v0.json` in the SplatForge repository. Internal research numbers are reproducible in `splatforge-private` under NDA.*
